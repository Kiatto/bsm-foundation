"""
rag_demo.py — BSM + LLM RAG integration demo with ContextCompiler.

Downloads GPT-2, indexes a small knowledge base, retrieves with
ContextCompiler (dedup + cluster + prune), then generates answers.

Usage:
    python examples/rag_demo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bsm.integrations.llm_rag import BSMRAG


KNOWLEDGE_BASE = [
    {"text": "The Eiffel Tower is located in Paris, France. "
             "It was built in 1889 for the World's Fair and stands "
             "at 330 meters tall.",
     "source": "geography"},
    {"text": "The Great Wall of China is an ancient fortification "
             "stretching over 13,000 miles. Construction began "
             "as early as the 7th century BC.",
     "source": "geography"},
    {"text": "The Louvre Museum in Paris houses the Mona Lisa. "
             "It is the world's largest art museum.",
     "source": "geography"},
    {"text": "Photosynthesis is the process by which plants convert "
             "sunlight into chemical energy. It produces glucose "
             "and oxygen as byproducts.",
     "source": "science"},
    {"text": "Chlorophyll, the green pigment in plants, absorbs "
             "light energy for photosynthesis. It reflects green "
             "light, which is why leaves appear green.",
     "source": "science"},
    {"text": "Water freezes at 0 degrees Celsius (32 degrees "
             "Fahrenheit) and boils at 100 degrees Celsius "
             "(212 degrees Fahrenheit) at sea level.",
     "source": "science"},
    {"text": "Python is a high-level programming language created by "
             "Guido van Rossum in 1991. It emphasizes readability "
             "and is widely used in data science.",
     "source": "programming"},
    {"text": "Variables in Python are dynamically typed. A variable "
             "can hold any type of value, and its type can change "
             "during program execution.",
     "source": "programming"},
    {"text": "The human brain contains approximately 86 billion "
             "neurons. It consumes about 20% of the body's energy "
             "despite being only 2% of body weight.",
     "source": "biology"},
    {"text": "DNA (deoxyribonucleic acid) carries genetic information "
             "in all living organisms. It consists of four nucleotide "
             "bases: adenine, guanine, cytosine, and thymine.",
     "source": "biology"},
]

QUERIES = [
    "Where is the Eiffel Tower?",
    "How does photosynthesis work?",
    "What is Python?",
    "At what temperature does water boil?",
    "What is DNA?",
]


def main():
    print("=" * 65)
    print("BSM RAG Demo — with ContextCompiler")
    print("=" * 65)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("\n[1] Loading GPT-2...")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print("[2] Initialising BSMRAG with ContextCompiler...")
    rag = BSMRAG(
        model, tokenizer,
        max_context_chunks=3,
        max_context_tokens=512,
    )

    print(f"[3] Indexing {len(KNOWLEDGE_BASE)} documents...")
    rag.index_documents(KNOWLEDGE_BASE)
    print(f"    Stored {rag.bsm._store.size()} chunks")
    print(f"    Memory: {rag.bsm.info()}")

    print(f"\n[4] ContextCompiler in action (retrieve → compile):\n")
    for q in QUERIES:
        state = rag.bsm.encode(q)
        raw_results = rag.bsm.recall(state, k=20)
        compiled = rag.compiler.compile(raw_results, query=q)
        stats = rag.compiler.stats()

        print(f"  Q: {q}")
        print(f"     Raw chunks: {stats['chunks_in']}, "
              f"Clusters: {stats['clusters']}, "
              f"Compiled: {stats['chunks_out']} chunks, "
              f"~{stats['tokens']} tokens")
        for c in compiled:
            print(f"       → {c[:90]}...")
        print()

    print(f"[5] Generation — WITH ContextCompiler:\n")
    for q in QUERIES:
        answer = rag.generate(q, max_new_tokens=50)
        print(f"  Q: {q}")
        print(f"  A: {answer}\n")

    print(f"[6] Generation — WITHOUT ContextCompiler (raw top-3):\n")
    for q in QUERIES:
        answer = rag.generate(q, max_new_tokens=50, use_compiler=False)
        print(f"  Q: {q}")
        print(f"  A: {answer}\n")

    print("=" * 65)
    print("RAG health:", rag.memory_health())
    print("Compiler stats (last query):", rag.compiler.stats())
    print("=" * 65)


if __name__ == "__main__":
    import torch
    main()
