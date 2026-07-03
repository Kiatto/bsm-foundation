"""
llm_adapter.py — BSM Memory Layer adapter for any LLM.

Converts any language model's hidden states to binary states,
enabling geometric memory augmentation.

Usage:
    from bsm.llm_adapter import LLMAdapter
    
    adapter = LLMAdapter(state_dim=128)
    
    # During inference:
    hidden = llm.get_hidden_states(input_ids)  # [B, L, H]
    context, memory_state = adapter.encode(hidden)  # [B, D]
    
    # Retrieve similar experiences
    experiences = mem.recall(memory_state, top_k=4)
    
    # Augment LLM output with memory
    augmented_logits = adapter.augment(llm_logits, experiences, alpha=0.3)
"""

import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from typing import Optional, List, Dict, Any, Tuple, Union


class LLMAdapter(nn.Module):
    """
    BSM Memory Layer adapter.
    Maps any LLM's hidden states to binary memory states.
    
    Architecture:
        LLM hidden state [H]
          → Linear(H, hid_dim)
            → Tanh
              → Linear(hid_dim, state_dim)
                → sign() → binary state [D]
    """
    
    def __init__(self,
                 llm_hidden_dim: int = 768,
                 state_dim: int = 128,
                 hid_dim: int = 384):
        super().__init__()
        self.state_dim = state_dim
        self.encoder = nn.Sequential(
            nn.Linear(llm_hidden_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, state_dim),
        )
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode LLM hidden states to binary memory states.
        
        Args:
            hidden_states: [B, H] or [B, L, H] from LLM
            
        Returns:
            (raw_state, binary_state) both [B, D]
        """
        # If 3D, use last token's hidden state
        if hidden_states.dim() == 3:
            hidden = hidden_states[:, -1, :]  # [B, H]
        else:
            hidden = hidden_states  # [B, H]
        
        raw = self.encoder(hidden)  # [B, D]
        binary = torch.sign(raw)    # [B, D]
        return raw, binary
    
    def augment(self,
                llm_logits: torch.Tensor,
                memory_experiences: List[Any],
                token_decoder: Optional[callable] = None,
                alpha: float = 0.3) -> torch.Tensor:
        """
        Augment LLM logits with retrieved memory experiences.
        
        Args:
            llm_logits: [B, V] logits from LLM
            memory_experiences: list of (experience, distance, weight) from MemoryEngine
            token_decoder: function(token_id) → token_votes (optional)
            alpha: weight given to memory (0 = LLM only, 1 = memory only)
            
        Returns:
            augmented_logits: [B, V]
        """
        if not memory_experiences:
            return llm_logits
        
        # Memory vote distribution
        mem_votes = Counter()
        for exp in memory_experiences:
            if isinstance(exp, tuple) and len(exp) >= 2:
                token_id, weight = exp[0], exp[1]
            elif token_decoder:
                token_id = token_decoder(exp)
                weight = 1.0
            else:
                continue
            mem_votes[token_id] += weight
        
        if not mem_votes:
            return llm_logits
        
        # Convert memory votes to logit adjustment
        # For each token that appeared in memory, boost its logit
        V = llm_logits.shape[-1]
        adjustment = torch.zeros_like(llm_logits)
        for token_id, weight in mem_votes.most_common(min(10, len(mem_votes))):
            if token_id < V:
                adjustment[0, token_id] = weight * alpha * 10.0  # scale factor
        
        return llm_logits + adjustment
    
    def extract_state(self, llm_model, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Convenience: get hidden state from any HuggingFace-like LLM and encode.
        
        Args:
            llm_model: model with .get_input_embeddings() or base_model
            input_ids: [B, L] token IDs
            
        Returns:
            binary_state [B, D]
        """
        # Get hidden states — this is model-specific
        # For HuggingFace models:
        with torch.no_grad():
            outputs = llm_model(input_ids, output_hidden_states=True)
            # Use last hidden state
            if hasattr(outputs, 'hidden_states'):
                hidden = outputs.hidden_states[-1]  # [B, L, H]
            else:
                hidden = outputs.last_hidden_state  # [B, L, H]
        
        _, binary = self.forward(hidden)
        return binary
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)
    
    def load(self, path: str):
        self.load_state_dict(torch.load(path))
        return self
