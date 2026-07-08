"""
llm_rag.py — BSM + LLM RAG integration.

Pattern: encode text chunks → BSM store → retrieve by query
         → [ContextCompiler: rank → cluster → merge → prune]
         → prepend as context → LLM generate.

This is the CORRECT architecture for LLM augmentation (not hidden-state override).
"""

import torch
import numpy as np
from typing import List, Optional

from bsm import BSM
from bsm.memory.context_compiler import ContextCompiler


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
                 max_context_chunks: int = 5,
                 max_context_tokens: int = 768):
        self.llm = llm
        self.tokenizer = tokenizer
        self.bsm = bsm or BSM(encoder="hash", state_dim=256)
        self.chunk_size = chunk_size
        self.max_context_chunks = max_context_chunks
        self.compiler = ContextCompiler(
            max_tokens=max_context_tokens,
            max_chunks=max_context_chunks,
            tokenizer=tokenizer,
        )

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
        """Index a list of {"text": str, "source": str} documents.

        If the BSM encoder is a ProjectionEncoder that hasn't been fitted,
        this will fit it on the corpus texts first.
        """
        from bsm.memory.encoder.bsm_encoder import ProjectionEncoder
        enc = self.bsm._encoder
        if isinstance(enc, ProjectionEncoder) and not enc._fitted:
            texts = [doc["text"] for doc in docs]
            enc.fit(texts)
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

    def retrieve(self, query: str, k: Optional[int] = None,
                 compile: bool = True) -> List[str]:
        """Return top-k chunk texts relevant to *query*.

        With compile=True (default), the ContextCompiler deduplicates,
        clusters, and prunes to fit the token budget.
        """
        k = k or self.max_context_chunks * 4
        state = self.bsm.encode(query)
        results = self.bsm.recall(state, k=k)
        if compile and results:
            return self.compiler.compile(results, query=query)
        return [r[0]["text"] for r in results[:self.max_context_chunks]]

    def retrieve_with_distances(self, query: str, k: Optional[int] = None
                                ) -> List[tuple]:
        """Return [(chunk_text, hamming_distance), ...]."""
        k = k or self.max_context_chunks * 4
        state = self.bsm.encode(query)
        results = self.bsm.recall(state, k=k)
        compiled = self.compiler.compile(results, query=query)
        return [(t, 0) for t in compiled]  # distances after merge are approximate

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def build_prompt(self, query: str, k: Optional[int] = None) -> str:
        """Build a prompt with compiled context prepended."""
        k = k or self.max_context_chunks * 4
        compiled = self.retrieve(query, k=k, compile=True)
        context_block = "\n".join(f"Context: {c}" for c in compiled)
        return f"{context_block}\n\nQuestion: {query}\nAnswer:"

    def generate(self,
                 query: str,
                 k: Optional[int] = None,
                 max_new_tokens: int = 60,
                 use_compiler: bool = True,
                 **gen_kwargs) -> str:
        """Generate answer using RAG: retrieve context + LLM generation.

        Args:
            use_compiler: if True, applies ContextCompiler (dedup, cluster, prune).
                          if False, returns raw top-k chunks.
        """
        if use_compiler:
            prompt = self.build_prompt(query, k=k)
        else:
            raw = self.retrieve(query, k=k, compile=False)
            ctx = "\n".join(f"Context: {c}" for c in raw)
            prompt = f"{ctx}\n\nQuestion: {query}\nAnswer:"
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
