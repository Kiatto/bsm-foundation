"""
Mutual Information: quanto informazione ha ogni bit dello stato sul prossimo token?

Calcola MI(bit_j, token_k) per ogni bit j (0-255) e ogni bit del token (0-11),
poi ordina i 256 bit per MI totale.
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import time, math
from collections import Counter

torch.manual_seed(42)
np.random.seed(42)

IN_DIM = 48; STATE_DIM = 256; HID_DIM = 384
C = 4; STEPS = 6000; BATCH_SIZE = 16; SEQ_LEN = 64
LEARNING_RATE = 1e-3

from tokenizers import Tokenizer
tok = Tokenizer.from_file("data/tokenizer.json")

with open("data/tinystories_train.txt") as f: text = f.read()
lines = [l.strip() for l in text.split("\n") if l.strip() and not l.startswith("<|endoftext|>")]
all_text = " ".join(lines)
words = all_text.split()
stories_text = []
for i in range(0, len(words), 2000):
    chunk = " ".join(words[i:i+2000])
    if len(chunk) > 200: stories_text.append(chunk)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, tok, texts, seq_len=64):
        self.tok = tok; self.seq_len = seq_len
        self.tokens = []
        for s in texts:
            ids = tok.encode(s).ids
            if len(ids) >= seq_len + 1:
                for start in range(0, len(ids) - seq_len, seq_len // 2):
                    self.tokens.append(torch.tensor(ids[start:start+seq_len+1]))
    def __len__(self): return len(self.tokens)
    def __getitem__(self, i):
        t = self.tokens[i]; return t[:self.seq_len], t[1:self.seq_len+1]

ds = TextDataset(tok, stories_text, seq_len=SEQ_LEN)
loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)

def make_iter(): return iter(loader)

def next_batch(it):
    while True:
        try: x, y = next(it[0])
        except StopIteration: it[0] = iter(loader); continue
        break
    B, T = x.shape
    ctx_list, tgt_list = [], []
    for t in range(C, T):
        ctx_list.append(x[:, t-C:t]); tgt_list.append(x[:, t])
    ctx = torch.stack(ctx_list, dim=1).reshape(-1, C)
    tgt = torch.stack(tgt_list, dim=1).reshape(-1)
    bits = int_to_bits(ctx, 12).reshape(-1, IN_DIM)
    return bits, tgt

def next_batch_dynamics(it):
    while True:
        try: x, y = next(it[0])
        except StopIteration: it[0] = iter(loader); continue
        break
    B, T = x.shape
    ctx_list, tgt_list, next_ctx_list = [], [], []
    for t in range(C, T-1):
        ctx_list.append(x[:, t-C:t]); tgt_list.append(x[:, t])
        next_ctx_list.append(x[:, t+1-C:t+1])
    ctx = torch.stack(ctx_list, dim=1).reshape(-1, C)
    tgt = torch.stack(tgt_list, dim=1).reshape(-1)
    next_ctx = torch.stack(next_ctx_list, dim=1).reshape(-1, C)
    bits = int_to_bits(ctx, 12).reshape(-1, IN_DIM)
    next_bits = int_to_bits(next_ctx, 12).reshape(-1, IN_DIM)
    return bits, tgt, next_bits

def int_to_bits(x, bits=12):
    return ((x.unsqueeze(-1) >> torch.arange(bits, device=x.device)) & 1).float()

def bits_to_int(bits, num_bits=12):
    bits = (bits > 0).float()
    return (bits * (2 ** torch.arange(bits.shape[-1], device=bits.device)).float()).sum(-1).long()

class BSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = nn.Linear(IN_DIM, HID_DIM)
        self.e2 = nn.Linear(HID_DIM, STATE_DIM)
        self.decoder = nn.Linear(STATE_DIM, 12)
    def forward(self, bits):
        h = torch.tanh(self.e1(bits))
        state_pre = self.e2(h)
        state_bin = torch.sign(state_pre)
        logits = self.decoder(state_pre)
        return logits, state_bin, state_pre
    def predict_token(self, logits):
        return bits_to_int(torch.sign(logits), 12)

# ============ TRAIN ============
print("="*60, flush=True)
print("Training...", flush=True)
print("="*60, flush=True)

model = BSM()
opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
it = [make_iter()]

t0 = time.time()
for step in range(STEPS):
    bits, tgt = next_batch(it)
    logits, _, _ = model(bits)
    loss = F.binary_cross_entropy_with_logits(
        logits.reshape(-1, 12),
        ((int_to_bits(tgt, 12) + 1) // 2).float().reshape(-1, 12))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 2000 == 0:
        print(f"  step {step}: loss={loss.item():.4f}", flush=True)

# Evaluate
model.eval(); it_eval = [make_iter()]
cor = 0; tot = 0
with torch.no_grad():
    for _ in range(100):
        bits, tgt = next_batch(it_eval)
        logits, _, _ = model(bits)
        pred = model.predict_token(logits)
        cor += (pred == tgt).sum().item(); tot += tgt.shape[0]
print(f"  Decoder: {cor/tot*100:.2f}%", flush=True)


# ====================================================================
# PHASE 1: COLLECT JOINT COUNTS FOR MUTUAL INFORMATION
# ====================================================================
print(f"\n{'='*60}", flush=True)
print("Phase 1: Computing Mutual Information per bit", flush=True)
print("="*60, flush=True)

# MI(bit_j, token_bit_k) per ogni j=0..255, k=0..11
# We'll collect counts in a [256, 12, 2, 2] array:
# count[j][k][state_val][token_val]
# where state_val = 0 or 1, token_val = 0 or 1

counts = np.zeros((STATE_DIM, 12, 2, 2), dtype=np.float64)

n_samples = 50000
it_data = [make_iter()]
model.eval()

with torch.no_grad():
    collected = 0
    while collected < n_samples:
        bits, tgt = next_batch(it_data)
        logits, state_bin, _ = model(bits)
        for i in range(bits.shape[0]):
            t = tgt[i].item()
            token_bits = bits_to_int(torch.tensor([[t]]), 12)  # get 12-bit repr
            # Simple: get token bits directly
            token_bits_arr = [(t >> k) & 1 for k in range(12)]
            for j in range(STATE_DIM):
                s_val = int((state_bin[i, j].item() + 1) // 2)  # {-1,1} → {0,1}
                for k in range(12):
                    t_val = token_bits_arr[k]
                    counts[j][k][s_val][t_val] += 1
            collected += 1
            if collected >= n_samples: break
    print(f"  Collected {collected} samples", flush=True)

# Compute MI for each bit
def mutual_information_from_counts(c):
    """c is a 2x2 array: c[s][t] = count of (state=s, token=t)"""
    total = c.sum()
    if total == 0: return 0.0
    mi = 0.0
    for s in [0, 1]:
        for t in [0, 1]:
            p_st = c[s][t] / total
            if p_st == 0: continue
            p_s = c[s].sum() / total
            p_t = c[:, t].sum() / total
            if p_s > 0 and p_t > 0:
                mi += p_st * math.log2(p_st / (p_s * p_t))
    return max(mi, 0.0)

# Per-bit MI (total across all 12 token bits)
per_bit_mi = np.zeros(STATE_DIM)
for j in range(STATE_DIM):
    mi_total = 0.0
    for k in range(12):
        mi_total += mutual_information_from_counts(counts[j][k])
    per_bit_mi[j] = mi_total

# Also compute per-bit MI for token level (not per token-bit)
# Aggregate token bits: H(token) - H(token | bit)
# For this, group tokens into buckets by their 12-bit value
# Since we have the counts per token-bit, the total MI is just the sum
# (because MI is additive for independent dimensions, and the 12 bits are
# independent in the uniform token distribution)

# Sort bits by MI
sorted_indices = np.argsort(-per_bit_mi)  # descending
sorted_mi = per_bit_mi[sorted_indices]

print(f"\n  Per-bit MI distribution:", flush=True)
print(f"    Mean MI: {per_bit_mi.mean():.4f} bits", flush=True)
print(f"    Max MI:  {per_bit_mi.max():.4f} bits (bit {per_bit_mi.argmax()})", flush=True)
print(f"    Min MI:  {per_bit_mi.min():.4f} bits (bit {per_bit_mi.argmin()})", flush=True)
print(f"    Std MI:  {per_bit_mi.std():.4f} bits", flush=True)

# Top 10 bits
print(f"\n  Top-10 bits by MI:", flush=True)
for rank in range(10):
    b = sorted_indices[rank]
    print(f"    bit {b:3d}: MI = {per_bit_mi[b]:.4f} bits", flush=True)

# Bottom 10 bits
print(f"\n  Bottom-10 bits by MI:", flush=True)
for rank in range(10):
    b = sorted_indices[STATE_DIM - 1 - rank]
    print(f"    bit {b:3d}: MI = {per_bit_mi[b]:.4f} bits", flush=True)


# ====================================================================
# PHASE 2: HIERARCHY ANALYSIS
# ====================================================================
print(f"\n{'='*60}", flush=True)
print("Phase 2: Hierarchy Analysis", flush=True)
print("="*60, flush=True)

# Cumulative MI as we add more bits
cumulative = np.cumsum(sorted_mi)
total_mi = cumulative[-1]

print(f"\n  Total MI across all bits: {total_mi:.4f} bits", flush=True)
print(f"\n  Cumulative MI by percentile:", flush=True)
for pct in [5, 10, 25, 50, 75]:
    n_bits = int(STATE_DIM * pct / 100)
    cum = cumulative[n_bits - 1]
    print(f"    Top {pct:2d}% bits ({n_bits:3d} bits): {cum:.4f} bits ({cum/total_mi*100:.1f}% of total MI)", flush=True)

# Gini coefficient of MI distribution
def gini(x):
    sorted_x = np.sort(x)
    n = len(x)
    cumsum = np.cumsum(sorted_x)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

mi_gini = gini(per_bit_mi)
print(f"\n  Gini coefficient of MI distribution: {mi_gini:.4f}", flush=True)
print(f"    (1.0 = all MI in one bit, 0.0 = uniform)", flush=True)


# ====================================================================
# PHASE 3: ABLATION BY RANKED BANKS
# ====================================================================
print(f"\n{'='*60}", flush=True)
print("Phase 3: Ablation by Ranked Banks (Top64 / Mid64 / Bottom64)", flush=True)
print("="*60, flush=True)

# Build GAM
class GAM:
    def __init__(self, simhash):
        self.simhash = simhash
        self.states = []; self.next_states = []; self.tokens = []
        self.buckets = [[] for _ in range(16384)]
    def state_to_int(self, s):
        bits = ((s + 1) // 2).byte().cpu().numpy()
        return bits.dot(1 << np.arange(STATE_DIM, dtype=np.uint64))
    def state_to_bucket(self, s):
        proj = s @ self.simhash.T
        bucket = 0
        for i in range(14):
            if proj[i] > 0: bucket |= (1 << i)
        return bucket
    def add(self, s, ns, tok):
        sid = self.state_to_int(s); nsid = self.state_to_int(ns)
        b = self.state_to_bucket(s); idx = len(self.states)
        self.states.append(sid); self.next_states.append(nsid)
        self.tokens.append(tok); self.buckets[b].append(idx)
    def build(self, model, max_examples=100000):
        model.eval(); it = [make_iter()]; n = 0
        with torch.no_grad():
            while n < max_examples:
                bits, tgt, nbits = next_batch_dynamics(it)
                logits, sb, _ = model(bits); _, nsb, _ = model(nbits)
                for i in range(bits.shape[0]):
                    self.add(sb[i], nsb[i], tgt[i].item()); n += 1
                    if n >= max_examples: break
        occ = sum(1 for bkt in self.buckets if bkt)
        print(f"  GAM: {n} states, {occ}/16384 buckets", flush=True)
    def query(self, s, cand=200, neigh=4):
        sid = self.state_to_int(s); b = self.state_to_bucket(s)
        cs = set()
        for idx in self.buckets[b]: cs.add(idx)
        if len(cs) < cand:
            for bb in range(14):
                nb = b ^ (1 << bb)
                for idx in self.buckets[nb]: cs.add(idx)
                if len(cs) >= cand: break
        cs = list(cs)[:cand]
        if not cs: return Counter()
        dists = [(self.states[i] ^ sid).bit_count() for i in cs]
        top = sorted(zip(cs, dists), key=lambda x: x[1])[:neigh]
        votes = Counter()
        for idx, d in top:
            w = 1.0 / (1.0 + d / (STATE_DIM * 2))
            votes[self.tokens[idx]] += w
        return votes

print(f"\nBuilding GAM...", flush=True)
simhash = torch.randn(14, STATE_DIM)
gam = GAM(simhash)
gam.build(model)

# Define ranked banks
top64 = sorted_indices[:64]
mid64 = sorted_indices[64:128]
bot64 = sorted_indices[128:192]
low64 = sorted_indices[192:256]

ranked_banks = {"Top64": top64, "Mid64": mid64, "Bot64": bot64, "Low64": low64}

# Original consecutive banks
orig_banks = {"W0-63": np.arange(0, 64), "S64-127": np.arange(64, 128),
              "E128-191": np.arange(128, 192), "R192-255": np.arange(192, 256)}

def ablation(model, gam, bank_dict, name, n_batches=50):
    model.eval()
    # Full GAM first
    it = [make_iter()]; cor = 0; tot = 0
    with torch.no_grad():
        for _ in range(n_batches):
            bits, tgt = next_batch(it)
            logits, sb, _ = model(bits)
            for i in range(bits.shape[0]):
                t = tgt[i].item()
                votes = gam.query(sb[i])
                pg = votes.most_common(1)[0][0] if votes else model.predict_token(logits[i:i+1]).item()
                cor += (pg == t); tot += 1
    full = cor / tot * 100
    
    print(f"\n  {name}: Full GAM = {full:.2f}%", flush=True)
    res = {}
    for bn, idxs in bank_dict.items():
        cor = 0; tot = 0
        it2 = [make_iter()]
        with torch.no_grad():
            for _ in range(n_batches):
                bits, tgt = next_batch(it2)
                logits, sb, _ = model(bits)
                for i in range(bits.shape[0]):
                    t = tgt[i].item()
                    ms = sb[i].clone()
                    ms[idxs] = 0
                    votes = gam.query(ms)
                    pg = votes.most_common(1)[0][0] if votes else model.predict_token(logits[i:i+1]).item()
                    cor += (pg == t); tot += 1
        acc = cor / tot * 100
        res[bn] = (acc, full - acc)
        print(f"    Without {bn:10s}: GAM={acc:.2f}% (drop={full-acc:+.2f}%)", flush=True)
    return full, res

print(f"\n{'─'*60}", flush=True)
print("Original Consecutive Banks", flush=True)
print("─"*60, flush=True)
full_orig, res_orig = ablation(model, gam, orig_banks, "Original")

print(f"\n{'─'*60}", flush=True)
print("Ranked Banks (by MI)", flush=True)
print("─"*60, flush=True)
full_rank, res_rank = ablation(model, gam, ranked_banks, "Ranked")


# ====================================================================
# SUMMARY
# ====================================================================
print(f"\n{'='*60}", flush=True)
print("MUTUAL INFORMATION — SUMMARY", flush=True)
print("="*60, flush=True)

print(f"\nPer-bit MI distribution:", flush=True)
print(f"  Mean: {per_bit_mi.mean():.4f}  Max: {per_bit_mi.max():.4f}  Min: {per_bit_mi.min():.4f}  Std: {per_bit_mi.std():.4f}  Gini: {mi_gini:.4f}", flush=True)

print(f"\nCumulative MI:", flush=True)
for pct in [5, 10, 25, 50, 75, 100]:
    n = int(STATE_DIM * pct / 100)
    print(f"  Top {pct:2d}%: {cumulative[n-1]:.4f} bits ({cumulative[n-1]/total_mi*100:.1f}%)", flush=True)

print(f"\nAblation: Original Banks (consecutive):", flush=True)
for bn, (acc, drop) in res_orig.items():
    print(f"  Without {bn:10s}: drop={drop:+.2f}%", flush=True)

print(f"\nAblation: Ranked Banks (by MI):", flush=True)
for bn, (acc, drop) in res_rank.items():
    print(f"  Without {bn:10s}: drop={drop:+.2f}%", flush=True)

# If ranked ablation shows a clearer hierarchy than original, 
# then the bank structure is reducible to MI hierarchy
orig_max = max(res_orig.values(), key=lambda x: x[1])[1]
orig_min = min(res_orig.values(), key=lambda x: x[1])[1]
rank_max = max(res_rank.values(), key=lambda x: x[1])[1]
rank_min = min(res_rank.values(), key=lambda x: x[1])[1]

print(f"\nKey comparison:", flush=True)
print(f"  Original: max drop = {orig_max:.2f}%, min drop = {orig_min:.2f}%, ratio = {orig_max/orig_min:.2f}x", flush=True)
print(f"  Ranked:   max drop = {rank_max:.2f}%, min drop = {rank_min:.2f}%, ratio = {rank_max/rank_min:.2f}x", flush=True)

if orig_max > rank_max * 0.8:
    print(f"\n  → Consecutive banks explain the hierarchy about as well as ranked banks.", flush=True)
    print(f"  → The working bank effect is NOT reducible to just high-MI bits.", flush=True)
else:
    print(f"\n  → Ranked banks show a MUCH clearer hierarchy than consecutive banks.", flush=True)
    print(f"  → The working bank effect IS reducible to high-MI bits concentrated in early positions.", flush=True)

print(f"\nTotal time: {(time.time()-t0)/60:.0f}m", flush=True)
