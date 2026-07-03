"""
Cross-domain validation of DSG invariants.
Tests on: TinyStories, WikiText, Python Code, E. coli DNA, Random.
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, time, math, json
from collections import Counter
from tokenizers import Tokenizer

torch.manual_seed(42); np.random.seed(42)

IN_DIM = 48; C = 4
STEPS = 2000; BATCH_SIZE = 16; SEQ_LEN = 64; LR = 1e-3
D = 128; H = 192
N_BUCKET_BITS = 12; N_BUCKETS = 1 << N_BUCKET_BITS

# ============ Models ============
class DSGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = nn.Linear(IN_DIM, H)
        self.e2 = nn.Linear(H, D)
        self.decoder = nn.Linear(D, 12)
    def forward(self, bits):
        h = torch.tanh(self.e1(bits))
        sp = self.e2(h); sb = torch.sign(sp)
        return self.decoder(sp), sb, sp
    def predict_token(self, logits):
        bits = (logits > 0).float()
        return (bits * (2**torch.arange(12, device=logits.device)).float()).sum(-1).long()

def int_to_bits(x, bits=12):
    return ((x.unsqueeze(-1) >> torch.arange(bits, device=x.device)) & 1).float()

# ============ Intrinsic Dimension ============
def participation_ratio(X):
    C = np.cov(X.T) if X.shape[1] <= 4096 else None
    if C is None: return None, None
    ev = np.linalg.eigvalsh(C)[::-1]
    ev = ev[ev > 1e-12]
    tv = np.sum(ev)
    pr = tv**2 / np.sum(ev**2) if np.sum(ev**2) > 0 else 0
    return pr, ev

# ============ TGAM ============
class TGAM:
    def __init__(self, simhash):
        self.simhash = simhash
        self.max_states = 50000
        self._states = torch.zeros(self.max_states, D)
        self._tokens = np.zeros(self.max_states, dtype=np.int32)
        self._n = 0
        self.buckets = [[] for _ in range(N_BUCKETS)]
    def bucket(self, s):
        proj = s @ self.simhash.T; b = 0
        for i in range(N_BUCKET_BITS):
            if proj[i] > 0: b |= 1 << i
        return b
    def add(self, s, tok):
        idx = self._n
        if idx >= self.max_states: return
        self._states[idx] = s.cpu(); self._tokens[idx] = tok
        self.buckets[self.bucket(s)].append(idx); self._n += 1
    def build(self, model, make_batch, max_examples=50000):
        model.eval(); n = 0
        with torch.no_grad():
            while n < max_examples:
                bits, tgt, nbits = make_batch(return_nbits=True)
                _, sb, _ = model(bits)
                for i in range(bits.shape[0]):
                    self.add(sb[i], tgt[i].item()); n += 1
                    if n >= max_examples: break
    def query_batch(self, sb, cand=200, neigh=4):
        results = []
        for i in range(sb.shape[0]):
            s = sb[i]; b = self.bucket(s)
            cs = set()
            for idx in self.buckets[b]: cs.add(idx)
            if len(cs) < cand:
                for bb in range(N_BUCKET_BITS):
                    nb = b ^ (1 << bb)
                    for idx in self.buckets[nb]: cs.add(idx)
                    if len(cs) >= cand: break
            cs = list(cs)[:cand]
            if not cs: results.append(Counter()); continue
            d = (s.unsqueeze(0) != self._states[cs].to(s.device)).float().sum(1).cpu()
            top = sorted(enumerate(d.tolist()), key=lambda x: x[1])[:neigh]
            votes = Counter()
            for wi, dd in top:
                votes[self._tokens[cs[wi]]] += 1.0 / (1.0 + dd / (D * 2))
            results.append(votes)
        return results

# ============ DataLoaders per domain ============

def make_tinystories_loader():
    tok = Tokenizer.from_file("data/tokenizer.json")
    with open("data/tinystories_train.txt") as f: text = f.read()
    lines = [l.strip() for l in text.split("\n") if l.strip() and not l.startswith("<|endoftext|>")]
    all_text = " ".join(lines)
    words = all_text.split()
    stories_text = []
    for i in range(0, len(words), 2000):
        chunk = " ".join(words[i:i+2000])
        if len(chunk) > 200: stories_text.append(chunk)
    return TextLoader(tok, stories_text)

def make_wikitext_loader():
    tok = Tokenizer.from_file("data/tokenizer.json")
    with open("/tmp/dsg_datasets/wikitext_train.txt") as f:
        text = f.read()
    # Split into paragraphs
    paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 200]
    return TextLoader(tok, paras)

def make_code_loader():
    tok = Tokenizer.from_file("data/tokenizer.json")
    with open("/tmp/dsg_datasets/python_code/all_code.py") as f:
        text = f.read()
    # Split into chunks of ~2000 chars
    chunks = []
    for i in range(0, len(text), 2000):
        chunk = text[i:i+2000].strip()
        if len(chunk) > 200: chunks.append(chunk)
    return TextLoader(tok, chunks)

def make_dna_loader():
    """DNA as 4-mers → 256 possible tokens (IDs 0-255)."""
    with open("/tmp/dsg_datasets/dna_sequence.txt") as f:
        seq = f.read().strip().upper()
    # Filter only A,C,G,T
    seq = ''.join(c for c in seq if c in 'ACGT')
    # 4-mer mapping
    bases = ['A', 'C', 'G', 'T']
    kmer_to_id = {}
    for i, a in enumerate(bases):
        for j, b in enumerate(bases):
            for k, c in enumerate(bases):
                for l, d in enumerate(bases):
                    kmer = a + b + c + d
                    kmer_to_id[kmer] = (i*64 + j*16 + k*4 + l)
    # Convert sequence to token IDs
    tokens = []
    for i in range(0, len(seq) - 3, 1):  # sliding 4-mers
        kmer = seq[i:i+4]
        if kmer in kmer_to_id:
            tokens.append(kmer_to_id[kmer])
    print(f"  DNA tokens: {len(tokens):,} unique: {len(set(tokens))}")
    return TokenLoader(tokens)

def make_random_loader(vocab_size=4096, n_tokens=2000000):
    tokens = np.random.randint(0, vocab_size, size=n_tokens).tolist()
    return TokenLoader(tokens)

class TextLoader:
    def __init__(self, tok, texts):
        self.tok = tok; self.texts = texts; self.tokens = []
        for s in texts:
            ids = tok.encode(s).ids
            if len(ids) >= 65: self.tokens.append(ids)
        print(f"  {len(self.tokens)} sequences", end="")
        # Pack all tokens into one long sequence
        self.all_tokens = []
        for t in self.tokens:
            self.all_tokens.extend(t)
        print(f", {len(self.all_tokens):,} tokens", end="")
    def get_batch(self, batch_size, seq_len):
        n = len(self.all_tokens)
        starts = np.random.randint(0, n - seq_len - 1, batch_size)
        x = torch.zeros(batch_size, seq_len, dtype=torch.long)
        y = torch.zeros(batch_size, seq_len, dtype=torch.long)
        for bi, s in enumerate(starts):
            x[bi] = torch.tensor(self.all_tokens[s:s+seq_len])
            y[bi] = torch.tensor(self.all_tokens[s+1:s+seq_len+1])
        return x, y

class TokenLoader:
    def __init__(self, tokens):
        self.tokens = tokens
        print(f"  {len(tokens):,} tokens")
    def get_batch(self, batch_size, seq_len):
        n = len(self.tokens)
        starts = np.random.randint(0, n - seq_len - 1, batch_size)
        x = torch.zeros(batch_size, seq_len, dtype=torch.long)
        y = torch.zeros(batch_size, seq_len, dtype=torch.long)
        for bi, s in enumerate(starts):
            x[bi] = torch.tensor(self.tokens[s:s+seq_len])
            y[bi] = torch.tensor(self.tokens[s+1:s+seq_len+1])
        return x, y

# ============ Unified experiment ============
def run_domain(name, data_loader):
    print(f"\n{'='*60}")
    print(f"  Domain: {name}")
    print(f"{'='*60}")
    t_start = time.time()

    model = DSGModel()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    for step in range(STEPS):
        x, y = data_loader.get_batch(BATCH_SIZE, SEQ_LEN)
        B, T = x.shape
        ctx_list, tgt_list = [], []
        for t in range(C, T):
            ctx_list.append(x[:, t-C:t])
            tgt_list.append(x[:, t])
        ctx = torch.stack(ctx_list, dim=1).reshape(-1, C)
        tgt = torch.stack(tgt_list, dim=1).reshape(-1)
        bits = int_to_bits(ctx, 12).reshape(-1, IN_DIM)

        logits, _, _ = model(bits)
        loss = F.binary_cross_entropy_with_logits(
            logits.reshape(-1, 12),
            ((int_to_bits(tgt, 12) + 1) // 2).float().reshape(-1, 12),
        )
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 1000 == 0:
            print(f"    step {step}: loss={loss.item():.4f}", flush=True)

    # ---- Decoder eval ----
    model.eval()
    cor, tot = 0, 0
    with torch.no_grad():
        for _ in range(100):
            x, y = data_loader.get_batch(BATCH_SIZE, SEQ_LEN)
            B, T = x.shape
            ctx_list, tgt_list = [], []
            for t in range(C, T):
                ctx_list.append(x[:, t-C:t])
                tgt_list.append(x[:, t])
            ctx = torch.stack(ctx_list, dim=1).reshape(-1, C)
            tgt = torch.stack(tgt_list, dim=1).reshape(-1)
            bits = int_to_bits(ctx, 12).reshape(-1, IN_DIM)
            logits, _, _ = model(bits)
            pred = model.predict_token(logits)
            cor += (pred == tgt).sum().item()
            tot += tgt.shape[0]
    dec_acc = cor / tot * 100
    print(f"  Decoder: {dec_acc:.2f}%", flush=True)

    # ---- Collect states ----
    bin_list = []
    n_col = 0
    with torch.no_grad():
        while n_col < 2000:
            x, y = data_loader.get_batch(BATCH_SIZE, SEQ_LEN)
            B, T = x.shape
            ctx_list = []
            for t in range(C, T):
                ctx_list.append(x[:, t-C:t])
            ctx = torch.stack(ctx_list, dim=1).reshape(-1, C)
            bits = int_to_bits(ctx, 12).reshape(-1, IN_DIM)
            _, sb, _ = model(bits)
            for i in range(bits.shape[0]):
                bin_list.append(sb[i].cpu().to(torch.int8))
                n_col += 1
                if n_col >= 2000: break
    X = torch.stack(bin_list).numpy()
    print(f"  States: {X.shape}", flush=True)

    # ---- Intrinsic dimension ----
    pr, ev = participation_ratio(X)
    pr_rand, _ = participation_ratio(np.random.choice([-1, 1], size=X.shape).astype(np.float64))
    n90 = int(np.searchsorted(np.cumsum(ev) / np.sum(ev), 0.9) + 1) if ev is not None else 0
    var = np.var(X.astype(np.float32), axis=0)
    frozen = int(np.sum(var < 0.01))
    print(f"  PR: {pr:.1f}  (random: {pr_rand:.1f})  n90: {n90}  frozen: {frozen}/{D}", flush=True)

    # ---- Build & evaluate GAM ----
    def make_batch(return_nbits=False):
        x, y = data_loader.get_batch(BATCH_SIZE, SEQ_LEN)
        B, T = x.shape
        ctx_list, tgt_list, nctx_list = [], [], []
        for t in range(C, T - 1):
            ctx_list.append(x[:, t-C:t])
            tgt_list.append(x[:, t])
            nctx_list.append(x[:, t+1-C:t+1])
        ctx = torch.stack(ctx_list, dim=1).reshape(-1, C)
        tgt = torch.stack(tgt_list, dim=1).reshape(-1)
        nctx = torch.stack(nctx_list, dim=1).reshape(-1, C)
        bits = int_to_bits(ctx, 12).reshape(-1, IN_DIM)
        nbits = int_to_bits(nctx, 12).reshape(-1, IN_DIM)
        if return_nbits:
            return bits, tgt, nbits
        return bits, tgt

    simhash = torch.randn(N_BUCKET_BITS, D)
    gam = TGAM(simhash)
    gam.build(model, make_batch, max_examples=30000)
    # Evaluate GAM
    cor_d, cor_g, tot = 0, 0, 0
    with torch.no_grad():
        for _ in range(75):
            bits, tgt = make_batch()
            logits, sb, _ = model(bits)
            votes_list = gam.query_batch(sb)
            for i in range(bits.shape[0]):
                t = tgt[i].item()
                pd = model.predict_token(logits[i:i+1]).item()
                pg = votes_list[i].most_common(1)[0][0] if votes_list[i] else pd
                cor_d += pd == t; cor_g += pg == t; tot += 1
    gam_acc = cor_g / tot * 100
    print(f"  GAM: {gam_acc:.2f}%  (Decoder: {cor_d/tot*100:.2f}%)", flush=True)

    elapsed = (time.time() - t_start) / 60
    print(f"  Time: {elapsed:.1f}m", flush=True)

    return {
        "domain": name,
        "decoder": round(dec_acc, 2),
        "gam": round(gam_acc, 2),
        "pr": round(pr, 1),
        "pr_random": round(pr_rand, 1),
        "n90": n90,
        "frozen_bits": frozen,
        "pr_d_ratio": round(pr / D, 3),
        "time_min": round(elapsed, 1),
    }


# ============ Run all domains ============
print("Preparing datasets...", flush=True)
print("  TinyStories:", end=""); ts_loader = make_tinystories_loader(); print()
print("  WikiText:", end=""); wt_loader = make_wikitext_loader(); print()
print("  Python Code:", end=""); cd_loader = make_code_loader(); print()
print("  DNA:", end=""); dna_loader = make_dna_loader()
print("  Random:", end=""); rnd_loader = make_random_loader()

results = {}
results["TinyStories"] = run_domain("TinyStories", ts_loader)
results["WikiText"] = run_domain("WikiText", wt_loader)
results["Python"] = run_domain("Python Code", cd_loader)
results["DNA"] = run_domain("E. coli DNA", dna_loader)
results["Random"] = run_domain("Random", rnd_loader)

# ============ Final table ============
print(f"\n{'='*70}")
print(f"CROSS-DOMAIN INTRINSIC DIMENSION — FINAL TABLE")
print(f"{'='*70}")

headers = ["Quantity"] + list(results.keys())
row_fmt = "{:20s}" + " {:>10s}" * len(results)
print(row_fmt.format(*headers))
print("-" * 70)

rows = [
    ("Decoder %", "decoder", "{:>10.2f}"),
    ("GAM %", "gam", "{:>10.2f}"),
    ("PR", "pr", "{:>10.1f}"),
    ("PR (random baseline)", "pr_random", "{:>10.1f}"),
    ("PR / D", "pr_d_ratio", "{:>10.3f}"),
    ("Eigen n90", "n90", "{:>10d}"),
    ("Frozen bits", "frozen_bits", "{:>10d}"),
]

for name, key, fmt in rows:
    line = f"{name:20s}"
    for d in results:
        v = results[d].get(key)
        if v is None: line += f" {'--':>10s}"
        else: line += fmt.format(v)
    print(line)

# Invariant analysis over non-random domains
text_results = {k: v for k, v in results.items() if k != "Random"}
print(f"\nInvariant analysis (text-only domains: {', '.join(text_results.keys())}):")
for name, key, fmt in rows:
    vals = [text_results[d].get(key) for d in text_results]
    vals = [float(v) for v in vals if v is not None and v != 0]
    if len(vals) >= 3:
        mean = np.mean(vals); cv = np.std(vals) / max(mean, 1e-8)
        status = "STABLE" if cv < 0.15 else ("TRENDING" if cv < 0.3 else "SCALING")
        print(f"  [{status:>8s}] {name:20s} μ={mean:.4f} CV={cv:.3f}")

# Save results
with open("/tmp/dsg_results/cross_domain.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to /tmp/dsg_results/cross_domain.json")

total = sum(r["time_min"] for r in results.values())
print(f"Total: {total:.0f}m")
