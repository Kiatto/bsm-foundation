"""
bench_improved.py — Confronto: HashEncoder vs ProjectionEncoder + query reranking.

Misura l'impatto delle due ottimizzazioni sulla accuracy finale.
"""

import sys, time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bsm import BSM
from bsm.integrations.llm_rag import BSMRAG


QA_PAIRS = [
    ("Where is the Eiffel Tower?", "Paris"),
    ("When was the Eiffel Tower built?", "1889"),
    ("What is the Great Wall of China?", "fortification"),
    ("How long is the Great Wall of China?", "13,000"),
    ("What does photosynthesis produce?", "glucose"),
    ("What pigment absorbs light in plants?", "chlorophyll"),
    ("What temperature does water freeze at?", "0"),
    ("What temperature does water boil at?", "100"),
    ("Who created Python?", "van Rossum"),
    ("When was Python created?", "1991"),
    ("How many neurons are in the human brain?", "86 billion"),
    ("What percentage of energy does the brain use?", "20%"),
    ("What does DNA stand for?", "deoxyribonucleic acid"),
    ("What are the four DNA bases?", "adenine"),
    ("What is the Louvre?", "museum"),
]

KNOWLEDGE_BASE = [
    "The Eiffel Tower is located in Paris, France. "
    "It was built in 1889 for the World's Fair and stands 330 meters tall.",
    "The Great Wall of China is an ancient fortification "
    "stretching over 13,000 miles. Construction began in the 7th century BC.",
    "The Louvre Museum in Paris houses the Mona Lisa. "
    "It is the world's largest art museum.",
    "Photosynthesis is the process by which plants convert "
    "sunlight into chemical energy. It produces glucose and oxygen.",
    "Chlorophyll, the green pigment in plants, absorbs "
    "light energy for photosynthesis. Leaves appear green because "
    "chlorophyll reflects green light.",
    "Water freezes at 0 degrees Celsius (32 degrees Fahrenheit) "
    "and boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
    "Python is a high-level programming language created by "
    "Guido van Rossum in 1991. It emphasizes readability.",
    "Variables in Python are dynamically typed and can change type.",
    "The human brain contains approximately 86 billion neurons. "
    "It consumes about 20% of the body's energy.",
    "DNA (deoxyribonucleic acid) carries genetic information. "
    "Its four nucleotide bases are adenine, guanine, cytosine, and thymine.",
]


def check(answer, keyword):
    return keyword.lower() in answer.lower()


def run_config(label, bsm_encoder, model, tokenizer):
    """Run one config: returns (accuracy, recall@5, latency_ms)."""
    print(f"\n  --- {label} ---")
    rag = BSMRAG(model, tokenizer, bsm=BSM(encoder=bsm_encoder, state_dim=256))
    rag.index_documents([{"text": d} for d in KNOWLEDGE_BASE])

    # Retrieval recall
    recall_hits = 0
    for q, kw in QA_PAIRS:
        results = rag.retrieve(q, k=5, compile=True)
        if any(kw.lower() in r.lower() for r in results):
            recall_hits += 1
    recall = recall_hits / len(QA_PAIRS)

    # Generation accuracy
    correct = 0
    latencies = []
    for q, kw in QA_PAIRS:
        t0 = time.perf_counter()
        ans = rag.generate(q, max_new_tokens=40)
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)
        ok = check(ans, kw)
        if ok:
            correct += 1
        print(f"    {'✓' if ok else '✗'} {q[:40]:40s} → {ans[:45]:45s} ({lat:.0f}ms)")

    acc = correct / len(QA_PAIRS)
    print(f"  Accuracy: {acc:.0%} ({correct}/{len(QA_PAIRS)})")
    print(f"  Recall@5: {recall:.0%} ({recall_hits}/{len(QA_PAIRS)})")
    print(f"  Latency: {np.mean(latencies):.0f}ms mean, {np.percentile(latencies, 99):.0f}ms p99")
    return acc, recall, np.mean(latencies)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print("Loading SmolLM2-135M-Instruct...")
    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M-Instruct", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M-Instruct", trust_remote_code=True)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Params: {params_m:.0f}M\n")

    # Config A: HashEncoder (originale)
    acc_a, rec_a, lat_a = run_config("HashEncoder (originale)", "hash", model, tokenizer)

    # Config B: ProjectionEncoder (fitted sul KB) + query reranking
    model.cpu()
    import gc; gc.collect()
    acc_b, rec_b, lat_b = run_config("ProjectionEncoder + rerank", "projection", model, tokenizer)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  CONFRONTO")
    print(f"{'=' * 60}")
    print(f"  {'Config':<35} {'Accuracy':<10} {'Recall@5':<10} {'Lat(ms)':<8}")
    print(f"  {'─' * 35} {'─' * 10} {'─' * 10} {'─' * 8}")
    print(f"  {'HashEncoder (originale)':<35} {acc_a:<10.0%} {rec_a:<10.0%} {lat_a:<8.0f}")
    print(f"  {'ProjectionEncoder + rerank':<35} {acc_b:<10.0%} {rec_b:<10.0%} {lat_b:<8.0f}")
    print(f"  {'Δ':<35} {acc_b - acc_a:<+10.0%} {rec_b - rec_a:<+10.0%} {lat_b - lat_a:<+8.0f}")
    print()

    del model, tokenizer


if __name__ == "__main__":
    import torch
    main()
