"""
DSG Scaling Experiment: measure invariants as state dimension grows.
Vectorized operations for D=128..1024.
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import time, math
from collections import Counter

torch.manual_seed(42)
np.random.seed(42)

IN_DIM = 48
C = 4
STEPS = 6000
BATCH_SIZE = 16
SEQ_LEN = 64
LR = 1e-3
N_BUCKET_BITS = 12
N_BUCKETS = 1 << N_BUCKET_BITS

DIMS = [128, 256, 512, 1024]
HID_DIMS = {128: 192, 256: 384, 512: 768, 1024: 1536}

from tokenizers import Tokenizer

tok = Tokenizer.from_file("data/tokenizer.json")
with open("data/tinystories_train.txt") as f:
    text = f.read()
lines = [l.strip() for l in text.split("\n") if l.strip() and not l.startswith("<|endoftext|>")]
all_text = " ".join(lines)
words = all_text.split()
stories_text = []
for i in range(0, len(words), 2000):
    chunk = " ".join(words[i : i + 2000])
    if len(chunk) > 200:
        stories_text.append(chunk)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, tok, texts, seq_len=64):
        self.tok = tok
        self.seq_len = seq_len
        self.tokens = []
        for s in texts:
            ids = tok.encode(s).ids
            if len(ids) >= seq_len + 1:
                for start in range(0, len(ids) - seq_len, seq_len // 2):
                    self.tokens.append(torch.tensor(ids[start : start + seq_len + 1]))

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        t = self.tokens[i]
        return t[: self.seq_len], t[1 : self.seq_len + 1]


ds = TextDataset(tok, stories_text, seq_len=SEQ_LEN)
loader = torch.utils.data.DataLoader(
    ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0
)


def make_iter():
    return iter(loader)


def next_batch(it, use_dynamics=False):
    while True:
        try:
            x, y = next(it[0])
        except StopIteration:
            it[0] = iter(loader)
            continue
        break
    B, T = x.shape
    if not use_dynamics:
        ctx_list, tgt_list = [], []
        for t in range(C, T):
            ctx_list.append(x[:, t - C : t])
            tgt_list.append(x[:, t])
        ctx = torch.stack(ctx_list, dim=1).reshape(-1, C)
        tgt = torch.stack(tgt_list, dim=1).reshape(-1)
        bits = int_to_bits(ctx, 12).reshape(-1, IN_DIM)
        return bits, tgt
    else:
        ctx_list, tgt_list, nctx_list = [], [], []
        for t in range(C, T - 1):
            ctx_list.append(x[:, t - C : t])
            tgt_list.append(x[:, t])
            nctx_list.append(x[:, t + 1 - C : t + 1])
        ctx = torch.stack(ctx_list, dim=1).reshape(-1, C)
        tgt = torch.stack(tgt_list, dim=1).reshape(-1)
        nctx = torch.stack(nctx_list, dim=1).reshape(-1, C)
        bits = int_to_bits(ctx, 12).reshape(-1, IN_DIM)
        nbits = int_to_bits(nctx, 12).reshape(-1, IN_DIM)
        return bits, tgt, nbits


def int_to_bits(x, bits=12):
    return ((x.unsqueeze(-1) >> torch.arange(bits, device=x.device)) & 1).float()


def bits_to_int(bits, nb=12):
    bits = (bits > 0).float()
    return (bits * (2 ** torch.arange(bits.shape[-1], device=bits.device)).float()).sum(-1).long()


def mi_from_counts(counts, total):
    mi = 0.0
    for a in [0, 1]:
        for b in [0, 1]:
            p = counts[a, b] / total
            if p == 0:
                continue
            pa = counts[a, :].sum() / total
            pb = counts[:, b].sum() / total
            if pa > 0 and pb > 0:
                mi += p * math.log2(p / (pa * pb))
    return max(mi, 0.0)


def gini(x):
    s = np.sort(x)
    n = len(x)
    cum = np.cumsum(s)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n


class TGAM:
    """Tensor-based GAM — arbitrary D support."""

    def __init__(self, simhash, state_dim):
        self.simhash = simhash
        self.state_dim = state_dim
        self.max_states = 100000
        self._states = torch.zeros(self.max_states, state_dim)
        self._tokens = np.zeros(self.max_states, dtype=np.int32)
        self._n = 0
        self.buckets = [[] for _ in range(N_BUCKETS)]

    def bucket(self, s):
        proj = s @ self.simhash.T
        b = 0
        for i in range(N_BUCKET_BITS):
            if proj[i] > 0:
                b |= 1 << i
        return b

    def add(self, s, tok):
        idx = self._n
        if idx >= self.max_states:
            return
        self._states[idx] = s.cpu()
        self._tokens[idx] = tok
        self.buckets[self.bucket(s)].append(idx)
        self._n += 1

    def build(self, model, max_examples=100000):
        model.eval()
        it = [make_iter()]
        n = 0
        with torch.no_grad():
            while n < max_examples:
                bits, tgt, nbits = next_batch(it, use_dynamics=True)
                _, sb, _ = model(bits)
                for i in range(bits.shape[0]):
                    self.add(sb[i], tgt[i].item())
                    n += 1
                    if n >= max_examples:
                        break
        occ = sum(1 for bkt in self.buckets if bkt)
        print(f"    GAM: {n} states, {occ}/{N_BUCKETS} buckets", flush=True)

    def query_batch(self, sb, cand=200, neigh=4):
        """Query multiple states at once for speed."""
        B = sb.shape[0]
        results = []
        for i in range(B):
            s = sb[i]
            b = self.bucket(s)
            cs = set()
            for idx in self.buckets[b]:
                cs.add(idx)
            if len(cs) < cand:
                for bb in range(N_BUCKET_BITS):
                    nb = b ^ (1 << bb)
                    for idx in self.buckets[nb]:
                        cs.add(idx)
                    if len(cs) >= cand:
                        break
            cs = list(cs)[:cand]
            if not cs:
                results.append(Counter())
                continue
            cand_states = self._states[cs].to(s.device)
            dists = (s.unsqueeze(0) != cand_states).float().sum(1).cpu()
            top = sorted(enumerate(dists.tolist()), key=lambda x: x[1])[:neigh]
            votes = Counter()
            for wi, d in top:
                w = 1.0 / (1.0 + d / (self.state_dim * 2))
                votes[self._tokens[cs[wi]]] += w
            results.append(votes)
        return results


class DSGModel(nn.Module):
    def __init__(self, state_dim, hid_dim):
        super().__init__()
        self.state_dim = state_dim
        self.e1 = nn.Linear(IN_DIM, hid_dim)
        self.e2 = nn.Linear(hid_dim, state_dim)
        self.decoder = nn.Linear(state_dim, 12)

    def forward(self, bits):
        h = torch.tanh(self.e1(bits))
        sp = self.e2(h)
        sb = torch.sign(sp)
        return self.decoder(sp), sb, sp

    def predict_token(self, logits):
        return bits_to_int(torch.sign(logits), 12)


# =============================================
# MAIN EXPERIMENT LOOP
# =============================================
all_results = {}

for D in DIMS:
    H = HID_DIMS[D]
    print(f"\n{'='*70}")
    print(f"DSG Scaling: D={D}, H={H}")
    print("=" * 70)
    t_exp = time.time()

    # ---------- TRAIN ----------
    model = DSGModel(D, H)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    it = [make_iter()]
    for step in range(STEPS):
        bits, tgt = next_batch(it)
        logits, _, _ = model(bits)
        loss = F.binary_cross_entropy_with_logits(
            logits.reshape(-1, 12),
            ((int_to_bits(tgt, 12) + 1) // 2).float().reshape(-1, 12),
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 3000 == 0:
            print(f"    step {step}: loss={loss.item():.4f}", flush=True)

    # ---------- DECODER EVAL ----------
    model.eval()
    it_eval = [make_iter()]
    cor = tot = 0
    with torch.no_grad():
        for _ in range(100):
            bits, tgt = next_batch(it_eval)
            logits, _, _ = model(bits)
            pred = model.predict_token(logits)
            cor += (pred == tgt).sum().item()
            tot += tgt.shape[0]
    dec_acc = cor / tot * 100
    print(f"  Decoder: {dec_acc:.2f}%", flush=True)

    # ---------- COLLECT STATES FOR MI ----------
    states_list = []
    tgt_list_mi = []
    it_data = [make_iter()]
    n_mi = 0
    with torch.no_grad():
        while n_mi < 20000:
            bits, tgt = next_batch(it_data)
            _, sb, _ = model(bits)
            for i in range(bits.shape[0]):
                states_list.append(sb[i].cpu())
                tgt_list_mi.append(tgt[i].item())
                n_mi += 1
                if n_mi >= 20000:
                    break
    states_all = torch.stack(states_list)
    state_bits = ((states_all + 1) // 2).byte()
    tgt_arr = np.array(tgt_list_mi, dtype=np.int64)
    print(f"    {state_bits.shape[0]} states collected", flush=True)

    # ---------- VECTORIZED MI ----------
    print(f"  Computing MI...", flush=True)
    N = state_bits.shape[0]
    s_np = state_bits.numpy()  # [N, D]
    mi_counts = np.zeros((D, 12, 2, 2), dtype=np.float64)
    for k in range(12):
        t_bit = (tgt_arr >> k) & 1  # [N]
        s0_t0 = ((1 - s_np) * (1 - t_bit[None, :].T)).sum(0)  # [D]
        s0_t1 = ((1 - s_np) * t_bit[None, :].T).sum(0)
        s1_t0 = (s_np * (1 - t_bit[None, :].T)).sum(0)
        s1_t1 = (s_np * t_bit[None, :].T).sum(0)
        mi_counts[:, k, 0, 0] = s0_t0
        mi_counts[:, k, 0, 1] = s0_t1
        mi_counts[:, k, 1, 0] = s1_t0
        mi_counts[:, k, 1, 1] = s1_t1

    per_bit_mi = np.zeros(D)
    for j in range(D):
        mi_total = 0.0
        for k in range(12):
            mi_total += mi_from_counts(mi_counts[j, k], N)
        per_bit_mi[j] = mi_total

    mi_gini_val = gini(per_bit_mi)
    sorted_mi = np.sort(per_bit_mi)[::-1]
    cumsum = np.cumsum(sorted_mi)
    total_mi = cumsum[-1]
    pct_top25 = cumsum[D // 4 - 1] / total_mi * 100 if D >= 4 else 100
    pct_top50 = cumsum[D // 2 - 1] / total_mi * 100 if D >= 2 else 100
    print(f"    Total MI: {total_mi:.4f} bits")
    print(f"    Gini: {mi_gini_val:.4f}")
    print(f"    Top 25%: {pct_top25:.1f}%")
    print(f"    Top 50%: {pct_top50:.1f}%")

    # ---------- VECTORIZED GEOMETRY CURVE ----------
    print(f"  Geometry curve...", flush=True)
    n_geo = min(2000, N)
    geo_idx = np.random.choice(N, n_geo, replace=False)
    geo_bits = s_np[geo_idx]
    geo_tgt = tgt_arr[geo_idx]
    dists_all = []
    same_all = []
    chunk = 100
    for i in range(n_geo):
        end = min(i + chunk, n_geo)
        dd = (geo_bits[i] != geo_bits[i + 1 : end]).sum(axis=1)
        ss = geo_tgt[i] == geo_tgt[i + 1 : end]
        dists_all.extend(dd.tolist())
        same_all.extend(ss.tolist())

    geo = {}
    for d, s in zip(dists_all, same_all):
        if d not in geo:
            geo[d] = [0, 0]
        geo[d][0] += s
        geo[d][1] += 1

    p_d0 = geo.get(0, [0, 0])[0] / max(geo.get(0, [0, 0])[1], 1)
    d_half = D // 2
    half_vals = [
        geo.get(d, [0, 0]) for d in range(max(0, d_half - D // 20), d_half + D // 20)
    ]
    if half_vals:
        p_dhalf = sum(v[0] for v in half_vals) / max(sum(v[1] for v in half_vals), 1)
    else:
        p_dhalf = 0.5
    print(f"    P(d=0): {p_d0:.4f}, P(d~D/2): {p_dhalf:.4f}")

    # ---------- BUILD GAM ----------
    print(f"  Building GAM...", flush=True)
    simhash = torch.randn(N_BUCKET_BITS, D)
    gam = TGAM(simhash, D)
    gam.build(model)

    # ---------- EVALUATE GAM ----------
    print(f"  Evaluating GAM...", flush=True)
    it_gam = [make_iter()]
    cor_dec = cor_gam = 0
    tot = 0
    with torch.no_grad():
        for _ in range(75):
            bits, tgt = next_batch(it_gam)
            logits, sb, _ = model(bits)
            votes_list = gam.query_batch(sb)
            for i in range(bits.shape[0]):
                t = tgt[i].item()
                pd = model.predict_token(logits[i : i + 1]).item()
                pg = votes_list[i].most_common(1)[0][0] if votes_list[i] else pd
                cor_dec += pd == t
                cor_gam += pg == t
                tot += 1
    gam_acc = cor_gam / tot * 100
    print(f"    Decoder={cor_dec/tot*100:.2f}%  GAM={gam_acc:.2f}%", flush=True)

    # ---------- BANK ABLATION ----------
    n_banks = 4
    bank_size = D // n_banks
    print(f"  Bank ablation ({bank_size}-bit banks)...", flush=True)
    bank_drops = []
    for b_idx in range(n_banks):
        b_start = b_idx * bank_size
        b_end = (b_idx + 1) * bank_size if b_idx < n_banks - 1 else D
        it_ab = [make_iter()]
        cor = 0
        tot = 0
        with torch.no_grad():
            for _ in range(40):
                bits, tgt = next_batch(it_ab)
                logits, sb, _ = model(bits)
                for i in range(bits.shape[0]):
                    t = tgt[i].item()
                    ms = sb[i].clone()
                    ms[b_start:b_end] = 0
                    votes = gam.query_batch(ms.unsqueeze(0))[0]
                    pg = votes.most_common(1)[0][0] if votes else model.predict_token(logits[i : i + 1]).item()
                    cor += pg == t
                    tot += 1
        drop = gam_acc - cor / tot * 100
        bank_drops.append(drop)
        print(f"    Bank {b_idx} (bits {b_start}-{b_end-1}): drop={drop:+.2f}%", flush=True)

    max_drop = max(bank_drops)
    min_drop = max(min(bank_drops), 0.01)
    ratio = max_drop / min_drop
    print(f"    Max/min drop ratio: {ratio:.2f}x", flush=True)

    # ---------- STATE CAPACITY ----------
    n_unique = len(set(tuple(s.tolist()) for s in state_bits[:2000]))

    # ---------- RESULTS ----------
    t_elapsed = (time.time() - t_exp) / 60
    all_results[D] = {
        "decoder": round(dec_acc, 2),
        "gam": round(gam_acc, 2),
        "mi_gini": round(mi_gini_val, 4),
        "mi_total": round(total_mi, 4),
        "mi_top25_pct": round(pct_top25, 1),
        "mi_top50_pct": round(pct_top50, 1),
        "p_d0": round(p_d0, 4),
        "p_dhalf": round(p_dhalf, 4),
        "bank_drops": [round(x, 2) for x in bank_drops],
        "bank_ratio": round(ratio, 2),
        "unique_2k": n_unique,
        "time_min": round(t_elapsed, 1),
    }
    print(f"  Time: {t_elapsed:.0f}m", flush=True)

# =============================================
# FINAL TABLE
# =============================================
print(f"\n{'='*70}")
print("DSG SCALING — FINAL TABLE")
print("=" * 70)
header = f"{'Qty':20s}"
for D in DIMS:
    header += f" {D:>8d}"
print(header)
print("-" * 70)
rows = [
    ("Decoder %", "decoder"),
    ("GAM %", "gam"),
    ("MI Gini", "mi_gini"),
    ("MI total", "mi_total"),
    ("MI top25%", "mi_top25_pct"),
    ("MI top50%", "mi_top50_pct"),
    ("P(d=0)", "p_d0"),
    ("P(d~D/2)", "p_dhalf"),
    ("Bank ratio", "bank_ratio"),
    ("Unique/2k", "unique_2k"),
]
for name, key in rows:
    line = f"{name:20s}"
    for D in DIMS:
        val = all_results[D][key]
        if isinstance(val, float):
            line += f" {val:>8.2f}"
        elif isinstance(val, int):
            line += f" {val:>8d}"
        else:
            line += f" {str(val):>8s}"
    print(line)

print(f"\nScaling summary:")
for D in DIMS:
    r = all_results[D]
    gain = r["gam"] / max(r["decoder"], 1)
    print(f"  D={D:4d}: GAM/Dec={gain:.1f}x  MI/12={r['mi_total']/12:.2f}  Gini={r['mi_gini']:.3f}  Bank={r['bank_ratio']:.2f}x  P0={r['p_d0']:.3f}")

print(f"\nInvariant analysis:")
for name, key in rows:
    vals = [all_results[D][key] for D in DIMS]
    if isinstance(vals[0], (float, int)):
        vals_f = [float(v) for v in vals]
        mean = np.mean(vals_f)
        cv = np.std(vals_f) / mean if mean != 0 else 999
        note = "STABLE" if cv < 0.1 else ("TRENDING" if cv < 0.3 else "SCALING")
        print(f"  [{note:>8s}] {name:15s} μ={mean:.4f} CV={cv:.3f}  vals={[round(v,3) for v in vals_f]}")

total_time = sum(all_results[D]["time_min"] for D in DIMS)
print(f"\nTotal: {total_time:.0f}m")
