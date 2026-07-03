"""
augment.py — BSM Memory Augmentation pipeline.

Generic memory layer for any LLM.

Architecture:
    Input
      → LLM (any model)
        → Hidden states
          → LLMAdapter → binary state
            → MemoryEngine → retrieve nearest experiences
          → Augment LLM logits with memory
      → Augmented output

Example:
    from bsm.augment import BSMAugment
    
    augmenter = BSMAugment(llm_hidden_dim=768)
    augmenter.connect_memory(memory_engine)
    
    # Augment each prediction
    prediction = augmenter.predict(input_ids, llm_model)
"""

import torch
import numpy as np
from collections import Counter
from typing import Optional, List, Any, Callable
from .llm_adapter import LLMAdapter
from .memory_engine import MemoryEngine


class BSMAugment:
    """
    BSM Memory Augmentation pipeline.
    Wraps any LLM with a geometric memory layer.
    """
    
    def __init__(self,
                 llm_hidden_dim: int = 768,
                 state_dim: int = 128,
                 hid_dim: int = 384,
                 alpha: float = 0.3,
                 min_confidence: float = 0.2):
        """
        Args:
            llm_hidden_dim: hidden dimension of the LLM
            state_dim: binary state dimension (D)
            hid_dim: projection hidden dimension
            alpha: memory influence weight (0..1)
            min_confidence: minimum memory confidence to override LLM
        """
        self.adapter = LLMAdapter(llm_hidden_dim, state_dim, hid_dim)
        self.memory = None
        self.alpha = alpha
        self.min_confidence = min_confidence
        self.state_dim = state_dim
        
        # Metrics
        self.n_predictions = 0
        self.n_memory_used = 0
        self.n_llm_only = 0
    
    def connect_memory(self, memory: MemoryEngine):
        """Attach a MemoryEngine instance."""
        self.memory = memory
        return self
    
    def create_memory(self,
                      capacity: int = 100000,
                      n_neighbors: int = 4) -> MemoryEngine:
        """Create and attach a fresh MemoryEngine."""
        self.memory = MemoryEngine(
            state_dim=self.state_dim,
            capacity=capacity,
            n_neighbors=n_neighbors,
        )
        return self.memory
    
    def observe(self, 
                llm_model,
                input_ids: torch.Tensor,
                target_token: int,
                value: float = 1.0):
        """
        Store an LLM observation in memory.
        
        Args:
            llm_model: any HuggingFace-like model
            input_ids: [B, L] token IDs (context)
            target_token: int, the token that followed
            value: confidence value for this memory
        """
        if self.memory is None:
            return
        
        # Extract binary state from LLM hidden state
        state = self.adapter.extract_state(llm_model, input_ids)
        
        # Store (state, target) pair
        self.memory.observe(state, target_token, value=value)
    
    def predict(self,
                llm_model,
                input_ids: torch.Tensor,
                return_details: bool = False) -> Any:
        """
        Predict next token with BSM memory augmentation.
        
        Args:
            llm_model: any HuggingFace-like model
            input_ids: [B, L] context tokens
            return_details: if True, return (token, {metadata})
            
        Returns:
            predicted token (int) or (token, metadata) tuple
        """
        self.n_predictions += 1
        metadata = {"memory_used": False}
        
        with torch.no_grad():
            # 1. Get LLM output
            outputs = llm_model(input_ids)
            llm_logits = outputs.logits[:, -1, :]  # [B, V]
            llm_token = llm_logits.argmax(dim=-1).item()
            
            # 2. Get binary state from LLM hidden
            state = self.adapter.extract_state(llm_model, input_ids)
            
            # 3. Query memory
            if self.memory and self.memory.size > 0:
                experiences = self.memory.recall(
                    state[0], top_k=self.memory.n_neighbors
                )
                
                if experiences:
                    # 4. Augment with memory
                    augmented_logits = self.adapter.augment(
                        llm_logits, experiences, alpha=self.alpha
                    )
                    aug_token = augmented_logits.argmax(dim=-1).item()
                    
                    metadata["memory_used"] = True
                    metadata["llm_token"] = llm_token
                    metadata["memory_token"] = aug_token
                    
                    if aug_token != llm_token:
                        self.n_memory_used += 1
                    else:
                        self.n_llm_only += 1
                    
                    if return_details:
                        return aug_token, metadata
                    return aug_token
        
        self.n_llm_only += 1
        if return_details:
            return llm_token, metadata
        return llm_token
    
    def train_adapter(self,
                      llm_model,
                      dataloader,
                      n_steps: int = 1000,
                      lr: float = 1e-3):
        """
        Train the adapter to produce good binary states.
        Uses contrastive loss: similar context → similar binary states.
        
        Args:
            llm_model: the LLM to extract hidden states from
            dataloader: yields (input_ids, target_tokens) batches
            n_steps: number of training steps
            lr: learning rate
        """
        self.adapter.train()
        opt = torch.optim.AdamW(self.adapter.parameters(), lr=lr)
        
        for step in range(n_steps):
            try:
                batch = next(dataloader)
            except StopIteration:
                dataloader = iter(dataloader)
                batch = next(dataloader)
            
            input_ids, targets = batch
            
            with torch.no_grad():
                hidden = llm_model(input_ids, output_hidden_states=True)
                if hasattr(hidden, 'hidden_states'):
                    h = hidden.hidden_states[-1][:, -1, :]
                else:
                    h = hidden.last_hidden_state[:, -1, :]
            
            # Encode
            raw, binary = self.adapter(h)
            
            # Contrastive loss: same target → similar states
            # (simplified: MSE between states of same target)
            loss = torch.tensor(0.0)
            unique_targets = targets.unique()
            for t in unique_targets:
                mask = targets == t
                if mask.sum() > 1:
                    states_t = binary[mask]
                    # Encourage states with same target to be similar
                    pairwise_dist = (states_t.unsqueeze(0) != states_t.unsqueeze(1)).float().mean()
                    loss = loss + pairwise_dist
            
            # Also encourage binary states to be ±1 (regularization)
            loss = loss + 0.01 * (binary.abs() - 1).pow(2).mean()
            
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        self.adapter.eval()
    
    def stats(self) -> dict:
        """Return augmentation statistics."""
        return {
            "predictions": self.n_predictions,
            "memory_overrides": self.n_memory_used,
            "llm_only": self.n_llm_only,
            "memory_rate": self.n_memory_used / max(self.n_predictions, 1),
            "alpha": self.alpha,
            "min_confidence": self.min_confidence,
            "adapter_params": sum(p.numel() for p in self.adapter.parameters()),
        }
