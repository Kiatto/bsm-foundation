"""
llm_rag.py — BSM + LLM RAG integration.

Pattern: encode text chunks → BSM store → retrieve by query → prepend as context.

This is the CORRECT architecture for LLM augmentation (not hidden-state override).
"""

import torch
import numpy as np
from typing import List, Optional

from bsm import BSM


class BSMRAG:
    """Retrieval-Augmented Generation powered by BSM memory.

    Usage:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        rag = BSMRAG(model, tokenizer)
        rag.index_text("The Eiffel Tower is in Paris, France.")
        answer = rag.generate("Where is the Eiffel Tower?")
    """

    def __init__(self,
                 llm,
                 tokenizer,
                 bsm: Optional[BSM] = None,
                 chunk_size: int = 200,
                 max_context_chunks: int = 3):
        self.llm = llm
        self.tokenizer = tokenizer
        self.bsm = bsm or BSM(encoder="hash", state_dim=256)
        self.chunk_size = chunk_size
        self.max_context_chunks = max_context_chunks

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_text(self, text: str, source: str = ""):
        """Split *text* into chunks and store each in BSM memory."""
        chunks = self._chunk(text)
        for i, chunk in enumerate(chunks):
            state = self.bsm.encode(chunk)
            self.bsm.observe(state, {
                "text": chunk,
                "source": source,
                "index": i,
            })

    def index_documents(self, docs: List[dict]):
        """Index a list of {"text": str, "source": str} documents."""
        for doc in docs:
            self.index_text(doc["text"], doc.get("source", ""))

    def _chunk(self, text: str) -> List[str]:
        """Split text into roughly equal-sized chunks by words."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
        return chunks

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: Optional[int] = None) -> List[str]:
        """Return top-k chunk texts relevant to *query*."""
        k = k or self.max_context_chunks
        state = self.bsm.encode(query)
        results = self.bsm.recall(state, k=k)
        return [r[0]["text"] for r in results]

    def retrieve_with_distances(self, query: str, k: Optional[int] = None
                                ) -> List[tuple]:
        """Return [(chunk_text, hamming_distance), ...]."""
        k = k or self.max_context_chunks
        state = self.bsm.encode(query)
        results = self.bsm.recall(state, k=k)
        return [(r[0]["text"], r[1]) for r in results]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def build_prompt(self, query: str, k: Optional[int] = None) -> str:
        """Build a prompt with retrieved context prepended."""
        k = k or self.max_context_chunks
        contexts = self.retrieve(query, k=k)
        context_block = "\n".join(f"Context: {c}" for c in contexts)
        return f"{context_block}\n\nQuestion: {query}\nAnswer:"

    def generate(self,
                 query: str,
                 k: Optional[int] = None,
                 max_new_tokens: int = 60,
                 **gen_kwargs) -> str:
        """Generate answer using RAG: retrieve context + LLM generation."""
        prompt = self.build_prompt(query, k=k)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=1024)

        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                **gen_kwargs,
            )

        full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated part after the prompt
        answer = full[len(prompt):].strip()
        return answer

    # ------------------------------------------------------------------
    # Memory introspection
    # ------------------------------------------------------------------

    def memory_stats(self) -> dict:
        return self.bsm.info()

    def memory_health(self) -> dict:
        return self.bsm.health()

    # ------------------------------------------------------------------

    def __repr__(self):
        return (f"BSMRAG(llm={type(self.llm).__name__}, "
                f"chunks={self.bsm._store.size()})")


# Alias for convenience
RAG = BSMRAG
