"""
Shuffle Test: is bank specialization a property of the model or an artifact of bit ordering?

Experiment A: Train with random seed=123, run bank ablation
Experiment B: Train with permuted state bits, run bank ablation
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from collections import Counter
import time

torch.manual_seed(42)
np.random.seed(42)

IN_DIM = 48; STATE_DIM = 256; HID_DIM = 384
C = 4; STEPS = 6000; BATCH_SIZE = 16; SEQ_LEN = 64
N_BUCKET_BITS = 14; N_BUCKETS = 1 << N_BUCKET_BITS
LEARNING_RATE = 1e-3

BANK_CONFIGS = {
    "working":  (0, 64), "semantic": (64, 128),
    "episodic": (128, 192), "routing": (192, 256),
}

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

class GAM:
    def __init__(self, simhash):
        self.simhash = simhash
        self.states = []; self.next_states = []; self.tokens = []
        self.buckets = [[] for _ in range(N_BUCKETS)]
    def state_to_int(self, s):
        bits = ((s + 1) // 2).byte().cpu().numpy()
        return bits.dot(1 << np.arange(STATE_DIM, dtype=np.uint64))
    def state_to_bucket(self, s):
        proj = s @ self.simhash.T
        bucket = 0
        for i in range(N_BUCKET_BITS):
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
                logits, sb, _, _ = model(bits)
                _, nsb, _, _ = model(nbits)
                for i in range(bits.shape[0]):
                    self.add(sb[i], nsb[i], tgt[i].item()); n += 1
                    if n >= max_examples: break
        occ = sum(1 for bkt in self.buckets if bkt)
        print(f"  GAM: {n} states, {occ}/{N_BUCKETS} buckets", flush=True)
    def query(self, s, cand=200, neigh=4):
        sid = self.state_to_int(s); b = self.state_to_bucket(s)
        cs = set()
        for idx in self.buckets[b]: cs.add(idx)
        if len(cs) < cand:
            for bb in range(N_BUCKET_BITS):
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

class BSM(nn.Module):
    def __init__(self, state_perm=None):
        super().__init__()
        self.e1 = nn.Linear(IN_DIM, HID_DIM)
        self.e2 = nn.Linear(HID_DIM, STATE_DIM)
        self.decoder = nn.Linear(STATE_DIM, 12)
        self.dynamics = nn.Linear(STATE_DIM, STATE_DIM)
        self.register_buffer('simhash', torch.randn(N_BUCKET_BITS, STATE_DIM))
        if state_perm is not None:
            perm = torch.arange(STATE_DIM)[state_perm]
        else:
            perm = torch.arange(STATE_DIM)
        self.register_buffer('state_perm', perm)
    def forward(self, bits):
        h = torch.tanh(self.e1(bits))
        state_pre = self.e2(h)
        # Apply permutation
        state_pre = state_pre[:, self.state_perm]
        state_bin = torch.sign(state_pre)
        logits = self.decoder(state_pre)
        ns_pre = self.dynamics(state_pre)
        ns_bin = torch.sign(ns_pre)
        return logits, state_bin, ns_bin, ns_pre
    def predict_token(self, logits):
        return bits_to_int(torch.sign(logits), 12)

def train_model(model, seed_val=42):
    torch.manual_seed(seed_val)
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    it = [make_iter()]
    for step in range(STEPS):
        bits, tgt = next_batch(it)
        logits, _, _, _ = model(bits)
        loss = F.binary_cross_entropy_with_logits(
            logits.reshape(-1, 12),
            ((int_to_bits(tgt, 12) + 1) // 2).float().reshape(-1, 12))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 3000 == 0:
            print(f"  step {step}: loss={loss.item():.4f}", flush=True)
    # Evaluate
    model.eval(); it_eval = [make_iter()]
    cor = 0; tot = 0
    with torch.no_grad():
        for _ in range(100):
            bits, tgt = next_batch(it_eval)
            logits, _, _, _ = model(bits)
            pred = model.predict_token(logits)
            cor += (pred == tgt).sum().item(); tot += tgt.shape[0]
    acc = cor / tot * 100
    print(f"  Decoder: {acc:.2f}%", flush=True)
    return model, acc

def bank_ablation(model, gam, name="model", n_batches=100):
    """Run bank ablation, return dict of (bank_name, acc_without_bank)."""
    model.eval(); it = [make_iter()]
    
    # Full GAM accuracy first
    cor_gam = 0; total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            bits, tgt = next_batch(it)
            logits, sb, _, _ = model(bits)
            for i in range(bits.shape[0]):
                t = tgt[i].item()
                votes = gam.query(sb[i])
                pg = votes.most_common(1)[0][0] if votes else model.predict_token(logits[i:i+1]).item()
                cor_gam += (pg == t); total += 1
    full_acc = cor_gam / total * 100
    print(f"\n  {name}: Full GAM = {full_acc:.2f}%", flush=True)
    
    results = {}
    for bn, (st, en) in BANK_CONFIGS.items():
        cor = 0; tot = 0
        with torch.no_grad():
            it2 = [make_iter()]
            for _ in range(n_batches):
                bits, tgt = next_batch(it2)
                logits, sb, _, _ = model(bits)
                for i in range(bits.shape[0]):
                    t = tgt[i].item()
                    ms = sb[i].clone()
                    ms[st:en] = 0
                    votes = gam.query(ms)
                    pg = votes.most_common(1)[0][0] if votes else model.predict_token(logits[i:i+1]).item()
                    cor += (pg == t); tot += 1
        acc = cor / tot * 100
        drop = full_acc - acc
        results[bn] = (acc, drop)
        print(f"    Without {bn:10s} (bits {st:3d}-{en-1:3d}): GAM={acc:.2f}% (drop={drop:+.2f}%)", flush=True)
    
    return full_acc, results


# ============ BASELINE (seed=42, no permutation) ============
print("="*60, flush=True)
print("BASELINE: seed=42, no permutation", flush=True)
print("="*60, flush=True)
model_base = BSM()
train_model(model_base, 42)
gam_base = GAM(model_base.simhash)
gam_base.build(model_base)
full_base, res_base = bank_ablation(model_base, gam_base, "Baseline")


# ============ EXPERIMENT A: seed=123 ============
print(f"\n{'='*60}", flush=True)
print("EXPERIMENT A: seed=123, no permutation", flush=True)
print("="*60, flush=True)
model_a = BSM()
train_model(model_a, 123)
gam_a = GAM(model_a.simhash)
gam_a.build(model_a)
full_a, res_a = bank_ablation(model_a, gam_a, "Seed=123")


# ============ EXPERIMENT B: permuted bits ============
print(f"\n{'='*60}", flush=True)
print("EXPERIMENT B: seed=42, PERMUTED state bits", flush=True)
print("="*60, flush=True)
perm = torch.randperm(STATE_DIM)
model_b = BSM(state_perm=perm)
train_model(model_b, 42)
gam_b = GAM(model_b.simhash)
gam_b.build(model_b)
full_b, res_b = bank_ablation(model_b, gam_b, "Permuted")


# ============ SUMMARY ============
print(f"\n{'='*60}", flush=True)
print("SHUFFLE TEST — SUMMARY", flush=True)
print("="*60, flush=True)
print(f"\n{'Config':15s} {'FullGAM':8s} {'Working':10s} {'Semantic':10s} {'Episodic':10s} {'Routing':10s}", flush=True)
print("-"*63, flush=True)
print(f"{'Baseline':15s} {full_base:7.2f}%  " + "".join(f"{res_base[b][1]:+8.2f}%  " for b in BANK_CONFIGS), flush=True)
print(f"{'Seed=123':15s} {full_a:7.2f}%  " + "".join(f"{res_a[b][1]:+8.2f}%  " for b in BANK_CONFIGS), flush=True)
print(f"{'Permuted':15s} {full_b:7.2f}%  " + "".join(f"{res_b[b][1]:+8.2f}%  " for b in BANK_CONFIGS), flush=True)

# Analysis
print(f"\nAnalysis:", flush=True)
largest_drop_base = max(res_base.items(), key=lambda x: x[1][1])
largest_drop_a = max(res_a.items(), key=lambda x: x[1][1])
largest_drop_b = max(res_b.items(), key=lambda x: x[1][1])

print(f"  Baseline: largest drop from removing '{largest_drop_base[0]}' ({largest_drop_base[1][1]:+.2f}%)", flush=True)
print(f"  Seed=123: largest drop from removing '{largest_drop_a[0]}' ({largest_drop_a[1][1]:+.2f}%)", flush=True)
print(f"  Permuted: largest drop from removing '{largest_drop_b[0]}' ({largest_drop_b[1][1]:+.2f}%)", flush=True)

# Test: is the working bank always the most important?
working_always_wins = (
    res_base["working"][1] >= res_base["semantic"][1] and
    res_base["working"][1] >= res_base["episodic"][1] and
    res_base["working"][1] >= res_base["routing"][1] and
    res_a["working"][1] >= res_a["semantic"][1] and
    res_a["working"][1] >= res_a["episodic"][1] and
    res_a["working"][1] >= res_a["routing"][1] and
    res_b["working"][1] >= res_b["semantic"][1] and
    res_b["working"][1] >= res_b["episodic"][1] and
    res_b["working"][1] >= res_b["routing"][1]
)

print(f"\n  Working bank dominates in ALL runs: {working_always_wins}", flush=True)
if not working_always_wins:
    print(f"  → Implies: bank specialization is partially an artifact of bit ordering", flush=True)
else:
    print(f"  → Implies: bank specialization is a genuine property of training dynamics", flush=True)
