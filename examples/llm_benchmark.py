"""
llm_benchmark.py — BSM RAG benchmark: GPT-2 vs SmolLM2-Instruct.

Metriche concrete:
  - Answer accuracy (keyword match su 15 domande factuali)
  - Latenza per query
  - BSM retrieval Recall@5
  - Compressione ContextCompiler
"""

import sys, time, json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def _build_prompt_no_rag(question, model_type):
    if model_type == "instruct":
        return f"<|user|>\n{question}\n<|assistant|>\n"
    return f"Question: {question}\nAnswer:"


def _build_prompt_rag(question, contexts, model_type):
    ctx = "\n".join(contexts)
    if model_type == "instruct":
        return f"<|user|>\nContext:\n{ctx}\n\nQuestion: {question}\n<|assistant|>\n"
    return f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"


def check_answer(answer: str, keyword: str) -> bool:
    return keyword.lower() in answer.lower()


def run_benchmark(model_name, label, model_type="base", max_new=40):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print(f"{'=' * 65}")

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model_params = sum(p.numel() for p in model.parameters()) / 1e6
    load_s = time.perf_counter() - t0
    print(f"  Params: {model_params:.0f}M | Load: {load_s:.1f}s")

    # ---- No RAG ----
    print(f"\n  [No RAG]")
    no_rag_correct = 0
    no_rag_lat = []
    for q, kw in QA_PAIRS:
        prompt = _build_prompt_no_rag(q, model_type)
        inp = tokenizer(prompt, return_tensors="pt")
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new,
                                 pad_token_id=tokenizer.eos_token_id)
        ms = (time.perf_counter() - t0) * 1000
        no_rag_lat.append(ms)
        ans = tokenizer.decode(out[0], skip_special_tokens=True)
        ans = ans[len(prompt):].strip()
        ok = check_answer(ans, kw)
        if ok: no_rag_correct += 1
        print(f"    {'✓' if ok else '✗'} {q[:42]:42s} → {ans[:48]:48s} ({ms:.0f}ms)")

    acc_no = no_rag_correct / len(QA_PAIRS)
    lat_no_mu = float(np.mean(no_rag_lat))
    lat_no_p99 = float(np.percentile(no_rag_lat, 99))

    # ---- With BSM RAG ----
    print(f"\n  [BSM RAG]")
    rag = BSMRAG(model, tokenizer)
    for doc in KNOWLEDGE_BASE:
        rag.index_text(doc)

    # Retrieval recall
    recall_hits = 0
    for q, kw in QA_PAIRS:
        results = rag.retrieve(q, k=5, compile=True)
        combined = " ".join(results)
        if kw.lower() in combined.lower():
            recall_hits += 1
    recall = recall_hits / len(QA_PAIRS)

    with_rag_correct = 0
    with_rag_lat = []
    compiler_stats = []
    for q, kw in QA_PAIRS:
        contexts = rag.retrieve(q, k=5, compile=True)
        prompt = _build_prompt_rag(q, contexts, model_type)
        inp = tokenizer(prompt, return_tensors="pt")
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new,
                                 pad_token_id=tokenizer.eos_token_id)
        ms = (time.perf_counter() - t0) * 1000
        with_rag_lat.append(ms)
        ans = tokenizer.decode(out[0], skip_special_tokens=True)
        ans = ans[len(prompt):].strip()
        ok = check_answer(ans, kw)
        if ok: with_rag_correct += 1
        compiler_stats.append(rag.compiler.stats())
        print(f"    {'✓' if ok else '✗'} {q[:42]:42s} → {ans[:48]:48s} ({ms:.0f}ms)")

    acc_with = with_rag_correct / len(QA_PAIRS)
    lat_with_mu = float(np.mean(with_rag_lat))
    lat_with_p99 = float(np.percentile(with_rag_lat, 99))

    avg_in = np.mean([s.get("chunks_in", 0) for s in compiler_stats])
    avg_out = np.mean([s.get("chunks_out", 0) for s in compiler_stats])
    avg_tok = np.mean([s.get("tokens", 0) for s in compiler_stats])

    # Summary table
    print(f"\n  {'─' * 55}")
    print(f"  {'Metric':<30} {'No RAG':<12} {'BSM RAG':<12}")
    print(f"  {'─' * 30} {'─' * 12} {'─' * 12}")
    print(f"  {'Accuracy':<30} {acc_no:<12.0%} {acc_with:<12.0%}")
    print(f"  {'Δ':<30} {'—':<12} {acc_with - acc_no:<+12.0%}")
    print(f"  {'Latency mean (ms)':<30} {lat_no_mu:<12.0f} {lat_with_mu:<12.0f}")
    print(f"  {'Latency p99 (ms)':<30} {lat_no_p99:<12.0f} {lat_with_p99:<12.0f}")
    print(f"  {'BSM Recall@5':<30} {'—':<12} {recall:<12.0%}")
    print(f"  {'BSM Memory (KB)':<30} {'—':<12} {rag.bsm._store.size() * 32 / 1024:<12.1f}")
    print(f"  {'ContextCompiler':<30} {'—':<12} f'{avg_in:.0f}→{avg_out:.0f} ch, {avg_tok:.0f} tok'")
    print(f"  {'Params (M)':<30} {model_params:<12.0f} {model_params:<12.0f}")

    model.cpu()
    del model, tokenizer, rag
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": label,
        "params_m": model_params,
        "type": model_type,
        "no_rag_accuracy": acc_no,
        "with_rag_accuracy": acc_with,
        "delta": acc_with - acc_no,
        "no_rag_latency_mean_ms": lat_no_mu,
        "with_rag_latency_mean_ms": lat_with_mu,
        "no_rag_latency_p99_ms": lat_no_p99,
        "with_rag_latency_p99_ms": lat_with_p99,
        "bsm_recall_at_5": recall,
        "bsm_memory_kb": rag.bsm._store.size() * 32 / 1024 if 'rag' in dir() else 0,
    }


if __name__ == "__main__":
    import torch
    results = []
    results.append(run_benchmark("openai-community/gpt2", "GPT-2 (base)", "base"))
    results.append(run_benchmark("HuggingFaceTB/SmolLM2-135M-Instruct",
                                 "SmolLM2-135M-Instruct", "instruct"))

    print(f"\n\n{'=' * 65}")
    print(f"  FINAL COMPARISON")
    print(f"{'=' * 65}")
    print(f"  {'Model':<25} {'No RAG':<8} {'BSM':<8} {'Δ':<7} {'Recall@5':<10} {'Lat(ms)':<8}")
    print(f"  {'─' * 25} {'─' * 8} {'─' * 8} {'─' * 7} {'─' * 10} {'─' * 8}")
    for r in results:
        d = r["with_rag_accuracy"] - r["no_rag_accuracy"]
        print(f"  {r['model']:<25} {r['no_rag_accuracy']:<8.0%} "
              f"{r['with_rag_accuracy']:<8.0%} {d:<+7.0%} "
              f"{r['bsm_recall_at_5']:<10.0%} "
              f"{r['with_rag_latency_mean_ms']:<8.0f}")

    print(f"\n  Saved: llm_benchmark_results.json")
    with open("llm_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
