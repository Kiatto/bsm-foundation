"""
rag_demo.py — BSM + LLM RAG integration demo.

Downloads GPT-2 (or uses cached), indexes a small knowledge base,
and answers questions using BSM-retrieved context.

Usage:
    python examples/rag_demo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bsm.integrations.llm_rag import BSMRAG


KNOWLEDGE_BASE = [
    {"text": "The Eiffel Tower is located in Paris, France. "
             "It was built in 1889 for the World's Fair.",
     "source": "geography"},
    {"text": "The Great Wall of China is a ancient fortification "
             "stretching over 13,000 miles. Construction began "
             "in the 7th century BC.",
     "source": "geography"},
    {"text": "Photosynthesis is the process by which plants convert "
             "sunlight into chemical energy. It produces oxygen "
             "as a byproduct.",
     "source": "science"},
    {"text": "Water freezes at 0 degrees Celsius (32 degrees Fahrenheit) "
             "and boils at 100 degrees Celsius (212 degrees Fahrenheit) "
             "at standard atmospheric pressure.",
     "source": "science"},
    {"text": "Python is a high-level programming language created by "
             "Guido van Rossum in 1991. It emphasizes code readability.",
     "source": "programming"},
    {"text": "The human brain contains approximately 86 billion neurons. "
             "It consumes about 20% of the body's energy.",
     "source": "biology"},
]

QUERIES = [
    "Where is the Eiffel Tower?",
    "How does photosynthesis work?",
    "What is Python?",
    "At what temperature does water boil?",
]


def main():
    print("=" * 60)
    print("BSM RAG Demo — GPT-2 with BSM Memory")
    print("=" * 60)

    # Load model (smallest available)
    print("\n[1] Loading GPT-2...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Create RAG
    print("[2] Initialising BSMRAG...")
    rag = BSMRAG(model, tokenizer)

    # Index knowledge base
    print(f"[3] Indexing {len(KNOWLEDGE_BASE)} documents...")
    rag.index_documents(KNOWLEDGE_BASE)
    print(f"    Stored {rag.bsm._store.size()} chunks in BSM memory")
    print(f"    Memory: {rag.bsm.info()}")

    # Test retrieval
    print(f"\n[4] Retrieval test:")
    for q in QUERIES:
        contexts = rag.retrieve(q, k=2)
        print(f"\n  Q: {q}")
        for c in contexts:
            print(f"    → {c[:80]}...")

    # Generate answers
    print(f"\n[5] Generation with RAG context:\n")
    for q in QUERIES:
        print(f"  Q: {q}")
        answer = rag.generate(q, k=2, max_new_tokens=40)
        print(f"  A: {answer}\n")

    # Compare: generation WITHOUT RAG
    print(f"[6] Generation WITHOUT RAG (baseline):\n")
    for q in QUERIES:
        prompt = f"Question: {q}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=40,
                pad_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer[len(prompt):].strip()
        print(f"  Q: {q}")
        print(f"  A: {answer}\n")

    print("=" * 60)
    print("RAG health:", rag.memory_health())
    print("=" * 60)


if __name__ == "__main__":
    import torch
    main()
