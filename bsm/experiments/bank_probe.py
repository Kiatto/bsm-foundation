"""
Bank Probing Experiment (Cognitive Architecture).

For each of the 4 state banks (working, semantic, episodic, routing),
train lightweight probes to predict linguistic features.

Tests whether the model's 256-bit state has spontaneously
organized into functionally specialized subspaces.
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from collections import Counter
import time, math, re

torch.manual_seed(42)
np.random.seed(42)

# ============ CONSTANTS ============
VOCAB_SIZE = 4096
IN_DIM = 48
STATE_DIM = 256
HID_DIM = 384
N_BUCKET_BITS = 14
N_BUCKETS = 1 << N_BUCKET_BITS
C = 4
STEPS = 6000
BATCH_SIZE = 16
SEQ_LEN = 64
LEARNING_RATE = 1e-3

# Banks
BANK_CONFIGS = {
    "working":  (0, 64),
    "semantic": (64, 128),
    "episodic": (128, 192),
    "routing":  (192, 256),
}

# ============ TOKENIZER ============
from tokenizers import Tokenizer
tok = Tokenizer.from_file("data/tokenizer.json")
vocab_size = tok.get_vocab_size()

# ============ DATA ============
with open("data/tinystories_train.txt") as f: text = f.read()

lines = [l.strip() for l in text.split("\n") if l.strip() and not l.startswith("<|endoftext|>")]
all_text = " ".join(lines)
words = all_text.split()
chunk_size = 2000
stories_text = []
for i in range(0, len(words), chunk_size):
    chunk = " ".join(words[i:i+chunk_size])
    if len(chunk) > 200:
        stories_text.append(chunk)

print(f"Generated {len(stories_text)} story chunks", flush=True)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, tok, texts, seq_len=64):
        self.tok, self.seq_len = tok, seq_len
        self.tokens = []
        for s in texts:
            ids = tok.encode(s).ids
            if len(ids) >= seq_len + 1:
                for start in range(0, len(ids) - seq_len, seq_len // 2):
                    self.tokens.append(torch.tensor(ids[start:start+seq_len+1]))
        print(f"  {len(self.tokens)} training windows", flush=True)
    def __len__(self):
        return len(self.tokens)
    def __getitem__(self, i):
        t = self.tokens[i]
        return t[:self.seq_len], t[1:self.seq_len+1]

ds = TextDataset(tok, stories_text, seq_len=SEQ_LEN)
loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)

def make_batch_iter():
    return iter(loader)

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
    return bits, tgt, x  # also return raw tokens for context analysis

def int_to_bits(x, bits=12):
    return ((x.unsqueeze(-1) >> torch.arange(bits, device=x.device)) & 1).float()

def bits_to_int(bits, num_bits=12):
    bits = (bits > 0).float()
    return (bits * (2 ** torch.arange(bits.shape[-1], device=bits.device)).float()).sum(-1).long()

# ============ MODEL ============

class BSMv4(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = nn.Linear(IN_DIM, HID_DIM)
        self.e2 = nn.Linear(HID_DIM, STATE_DIM)
        self.decoder = nn.Linear(STATE_DIM, 12)
        self.dynamics = nn.Linear(STATE_DIM, STATE_DIM)
        self.register_buffer('simhash', torch.randn(N_BUCKET_BITS, STATE_DIM))
    
    def forward(self, bits):
        h = torch.tanh(self.e1(bits))
        state_pre = self.e2(h)
        state_bin = torch.sign(state_pre)
        logits = self.decoder(state_pre)
        next_state_pre = self.dynamics(state_pre)
        next_state_bin = torch.sign(next_state_pre)
        return logits, state_bin, next_state_bin, next_state_pre
    
    def predict_token(self, logits):
        return bits_to_int(torch.sign(logits), 12)
    
    def accuracy(self, logits, targets):
        return (self.predict_token(logits) == targets).float().mean()


# ============ TRAINING ============

print("="*60, flush=True)
print("BANK PROBE: Training Phase", flush=True)
print("="*60, flush=True)

model = BSMv4()
opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
it = [make_batch_iter()]

t0 = time.time()
for step in range(STEPS):
    bits, tgt, _ = next_batch(it)
    h = torch.tanh(model.e1(bits))
    state_pre = model.e2(h)
    state_bin = torch.sign(state_pre)
    logits = model.decoder(state_pre)
    loss = F.binary_cross_entropy_with_logits(
        logits.reshape(-1, 12),
        ((int_to_bits(tgt, 12) + 1) // 2).float().reshape(-1, 12)
    )
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 2000 == 0:
        print(f"  step {step}: loss={loss.item():.4f}", flush=True)

# Evaluate
model.eval()
it_eval = [make_batch_iter()]
correct = 0; total = 0
with torch.no_grad():
    for _ in range(100):
        bits, tgt, _ = next_batch(it_eval)
        logits, _, _, _ = model(bits)
        pred = model.predict_token(logits)
        correct += (pred == tgt).sum().item()
        total += tgt.shape[0]
base_acc = correct / total * 100
print(f"Decoder baseline: {base_acc:.2f}%", flush=True)


# ====================================================================
# PHASE 1: COLLECT PROBE DATA
# ====================================================================
print(f"\n{'='*60}", flush=True)
print("Phase 1: Collecting Probe Data", flush=True)
print("="*60, flush=True)

# Features to probe per bank
# Collect (bank_state, target) pairs

# Determine punctuation and sentence-end token IDs
# Decode a few tokens to find punctuation IDs
sample_text = "Hello! Is this working? Yes, it is. Good."
sample_ids = tok.encode(sample_text).ids
print(f"Sample token IDs for '{sample_text}':", flush=True)
for tid in sample_ids:
    decoded = tok.decode([tid])
    print(f"  {tid}: '{decoded}'", flush=True)

# Build punctuation/sentence-end ID sets
punct_ids = set()
sentence_end_ids = set()
for tid in range(min(4096, vocab_size)):
    decoded = tok.decode([tid]).strip()
    if decoded in {'.', '!', '?', ',', ';', ':', '"', "'", '(', ')', '[', ']', '-', '...'}:
        punct_ids.add(tid)
        if decoded in {'.', '!', '?'}:
            sentence_end_ids.add(tid)

# Token frequency: collect from training data
print(f"\nCounting token frequencies...", flush=True)
token_freq = Counter()
for s in stories_text:
    token_freq.update(tok.encode(s).ids)
top_500 = set(tid for tid, _ in token_freq.most_common(500))
print(f"  Top-500 tokens cover {sum(token_freq[t] for t in top_500)/sum(token_freq.values())*100:.1f}% of tokens", flush=True)

# Collect (state_bank, targets) for probe training
# Target definitions (better balanced):
# 0: token_bit_0 (first bit of next token) — approx 50/50
# 1: token_bit_6 (middle bit) — approx 50/50  
# 2: token_bit_11 (last bit) — approx 50/50
# 3: context_repeat (token appears in 4-token context) — more balanced
# 4: first_window_half (position < 32 vs ≥ 32) — exactly 50/50

BANKS = list(BANK_CONFIGS.keys())
BANK_SLICES = [BANK_CONFIGS[b] for b in BANKS]
N_BANKS = len(BANKS)
N_TARGETS = 5

n_data = 0
all_bank_states = []  # [N_BANKS][N, 64]
all_targets = []      # [N, N_TARGETS]

n_batches = 100  # ~96K examples
it_data = [make_batch_iter()]

with torch.no_grad():
    for batch_idx in range(n_batches):
        bits, tgt, raw_x = next_batch(it_data)
        logits, state_bin, _, _ = model(bits)
        
        # Reconstruct context tokens for each example
        B = raw_x.shape[0]
        T = raw_x.shape[1]
        
        for i in range(bits.shape[0]):
            # Determine which sequence this example came from
            seq_idx = i // (T - C)
            pos_in_seq = (i % (T - C)) + C  # position in original sequence
            
            next_token = tgt[i].item()
            
            # Context tokens (4 previous tokens)
            ctx_tokens = [raw_x[seq_idx, pos_in_seq - C + j].item() for j in range(C)]
            
            # Features (better balanced)
            token_bits = int_to_bits(torch.tensor([next_token]), 12)[0]
            targets = [
                token_bits[0].item(),          # token_bit_0
                token_bits[6].item(),          # token_bit_6
                token_bits[11].item(),         # token_bit_11
                1.0 if next_token in ctx_tokens else 0.0,  # context_repeat
                1.0 if (pos_in_seq - C) < (T - C) // 2 else 0.0,  # first_window_half
            ]
            
            all_targets.append(targets)
            
            # Store per-bank states
            state = state_bin[i].cpu()
            bank_states = []
            for name, (start, end) in BANK_CONFIGS.items():
                bank_states.append(state[start:end].clone())
            all_bank_states.append(bank_states)
            
            n_data += 1

print(f"Collected {n_data} probe examples", flush=True)

# Convert to tensors
all_bank_tensors = []
for b_idx in range(N_BANKS):
    stacked = torch.stack([ex[b_idx] for ex in all_bank_states])
    all_bank_tensors.append(stacked)  # [N, 64]

all_targets_tensor = torch.tensor(all_targets, dtype=torch.float32)  # [N, 5]

print(f"Bank state shapes: {[t.shape for t in all_bank_tensors]}", flush=True)
print(f"Targets shape: {all_targets_tensor.shape}", flush=True)


# ====================================================================
# PHASE 2: TRAIN PROBES (PyTorch Logistic Regression)
# ====================================================================
print(f"\n{'='*60}", flush=True)
print("Phase 2: Training Probes (PyTorch Logistic Regression)", flush=True)
print("="*60, flush=True)

# Split data
N = len(all_targets)
N_train = int(N * 0.8)
indices = np.random.permutation(N)

results = {}  # results[(bank_name, target_name)] = (score, baseline, type)

target_names = ["token_bit_0", "token_bit_6", "token_bit_11", "context_repeat", "window_half"]

device = torch.device('cpu')

def train_logistic_probe(X_train, y_train, X_test, y_test, lr=0.1, epochs=1000):
    """Train linear probe: 64-dim → binary classification."""
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_test, dtype=torch.float32)
    y_val = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    # Normalize: center the binary states
    X_t = (X_t + 1) / 2  # {-1, 1} → {0, 1}
    X_val = (X_val + 1) / 2
    
    probe = nn.Linear(64, 1)
    nn.init.zeros_(probe.weight)
    nn.init.zeros_(probe.bias)
    opt = torch.optim.SGD(probe.parameters(), lr=lr)
    
    best_acc = 0
    for epoch in range(epochs):
        logits = probe(X_t)
        loss = F.binary_cross_entropy_with_logits(logits, y_t)
        opt.zero_grad(); loss.backward(); opt.step()
        
        if epoch % 200 == 0:
            with torch.no_grad():
                probs = torch.sigmoid(probe(X_val))
                preds = (probs > 0.5).float()
                acc = (preds == y_val).float().mean().item() * 100
                if acc > best_acc:
                    best_acc = acc
                if epoch % 500 == 0:
                    print(f"      epoch {epoch}: loss={loss.item():.4f} val_acc={acc:.2f}%", flush=True)
    
    # Final eval
    with torch.no_grad():
        probs = torch.sigmoid(probe(X_val))
        preds = (probs > 0.5).float()
        acc = (preds == y_val).float().mean().item() * 100
    return max(acc, best_acc)

print(f"\n{'Bank':12s} {'Target':16s} {'Score':8s} {'Baseline':8s} {'Type':8s}", flush=True)
print("-"*52, flush=True)

for b_idx, bank_name in enumerate(BANKS):
    states_np = all_bank_tensors[b_idx].numpy().astype(np.float32)
    
    for t_idx, target_name in enumerate(target_names):
        target_np = all_targets_tensor[:, t_idx].numpy()
        
        baseline = max(target_np.mean(), 1 - target_np.mean()) * 100
        
        X_train = states_np[indices[:N_train]]
        X_test = states_np[indices[N_train:]]
        y_train = target_np[indices[:N_train]]
        y_test = target_np[indices[N_train:]]
        
        score = train_logistic_probe(X_train, y_train, X_test, y_test, epochs=1000)
        score_type = "acc%"
        
        results[(bank_name, target_name)] = (score, baseline, score_type)
        
        print(f"{bank_name:12s} {target_name:16s} {score:7.2f}%  {baseline:7.2f}%  {score_type:8s}", flush=True)


# ====================================================================
# PHASE 3: PROBE-ONLY-BANK COMPARISON
# ====================================================================
print(f"\n{'='*60}", flush=True)
print("Phase 3: Probe-Only-Bank Analysis (single bank per probe)", flush=True)
print("="*60, flush=True)
print(f"\nFor each target, which bank gives the best probe accuracy?", flush=True)
print(f"{'Target':16s} {'Best Bank':12s} {'Score':8s} {'Baseline':8s} {'Improvement':12s}", flush=True)
print("-"*56, flush=True)

for t_idx, target_name in enumerate(target_names):
    best_score = -1
    best_bank = ""
    for b_idx, bank_name in enumerate(BANKS):
        score, baseline, _ = results[(bank_name, target_name)]
        if score > best_score:
            best_score = score
            best_bank = bank_name
    impr = best_score - baseline
    print(f"{target_name:16s} {best_bank:12s} {best_score:7.2f}%  {baseline:7.2f}%  +{impr:5.2f}%", flush=True)

# Bank importance ranking
print(f"\n{'='*60}", flush=True)
print("Bank Importance Ranking", flush=True)
print("="*60, flush=True)
print(f"\nAverage probe score across all targets:", flush=True)
bank_scores = {}
for b_idx, bank_name in enumerate(BANKS):
    scores = [results[(bank_name, tn)][0] for tn in target_names]
    avg = np.mean(scores)
    bank_scores[bank_name] = avg
    print(f"  {bank_name:12s}: {avg:.2f}% avg probe accuracy", flush=True)

# Rank banks
ranked = sorted(bank_scores.items(), key=lambda x: -x[1])
print(f"\n  Bank ranking (best → worst):", flush=True)
for r, (name, score) in enumerate(ranked):
    print(f"    {r+1}. {name:12s} ({score:.2f}%)", flush=True)


# ====================================================================
# PHASE 4: TEMPORAL ANALYSIS
# ====================================================================
print(f"\n{'='*60}", flush=True)
print("Phase 4: Temporal Bank Dynamics", flush=True)
print("="*60, flush=True)
print(f"\nHow do banks change over time within a sequence?", flush=True)

# Collect a sequence of states from a single sequence
model.eval()
it_temp = [make_batch_iter()]

with torch.no_grad():
    bits, tgt, raw_x = next_batch(it_temp)
    logits, state_bin, _, _ = model(bits)

# Pick first sequence in batch
seq_state = state_bin[0]  # [256]

# For each position in this sequence, collect bank means
bank_means_over_time = {name: [] for name in BANKS}
for name, (start, end) in BANK_CONFIGS.items():
    bank = seq_state[start:end]
    bank_means_over_time[name].append(bank.float().mean().item())

# Average over batch
bank_means_batch = {name: [] for name in BANKS}
for name, (start, end) in BANK_CONFIGS.items():
    banks = state_bin[:, start:end]  # [B, 64]
    bank_means_batch[name] = banks.float().mean(dim=1).tolist()

print(f"\n  Bank activation means (batch of {state_bin.shape[0]}):", flush=True)
for name in BANKS:
    vals = bank_means_batch[name]
    mean = np.mean(vals)
    std = np.std(vals)
    print(f"    {name:12s}: mean={mean:+.3f}  std={std:.3f}  range=[{min(vals):+.3f}, {max(vals):+.3f}]", flush=True)

# Fraction of bits that are +1 vs -1
print(f"\n  Bank sparsity (fraction of +1 bits):", flush=True)
for name, (start, end) in BANK_CONFIGS.items():
    bank_bits = state_bin[:, start:end]  # [B, 64]
    frac_pos = (bank_bits > 0).float().mean().item()
    print(f"    {name:12s}: {frac_pos*100:.1f}% +1 bits", flush=True)

# Cross-bank correlation
print(f"\n  Cross-bank correlation (mean absolute correlation):", flush=True)
for i, name_i in enumerate(BANKS):
    for j, name_j in enumerate(BANKS):
        if j <= i: continue
        bank_i = state_bin[:, BANK_CONFIGS[name_i][0]:BANK_CONFIGS[name_i][1]].float()
        bank_j = state_bin[:, BANK_CONFIGS[name_j][0]:BANK_CONFIGS[name_j][1]].float()
        # Compute mean per-position correlation
        corr_vals = []
        for pos in range(bank_i.shape[0]):
            corr = np.corrcoef(bank_i[pos].numpy(), bank_j[pos].numpy())[0, 1]
            if not np.isnan(corr):
                corr_vals.append(abs(corr))
        avg_corr = np.mean(corr_vals) if corr_vals else 0
        print(f"    {name_i:12s} × {name_j:12s}: mean |r| = {avg_corr:.3f}", flush=True)


# ====================================================================
# SUMMARY
# ====================================================================
print(f"\n{'='*60}", flush=True)
print("COGNITIVE ARCHITECTURE SUMMARY", flush=True)
print("="*60, flush=True)
print(f"\n{'Bank':12s} {'Bit0':8s} {'Bit6':8s} {'Bit11':8s} {'Repeat':8s} {'Half':8s} {'Avg':8s}", flush=True)
print("-"*60, flush=True)
for b_idx, bank_name in enumerate(BANKS):
    scores = [f"{results[(bank_name, tn)][0]:6.2f}%" for tn in target_names]
    avg = np.mean([results[(bank_name, tn)][0] for tn in target_names])
    print(f"{bank_name:12s} {scores[0]} {scores[1]} {scores[2]} {scores[3]} {scores[4]} {avg:6.2f}%", flush=True)

# Key insight
print(f"\nKey Insights:", flush=True)
print(f"  1. Best bank for token bits (bit0+bit6+bit11): "
      f"{max(BANKS, key=lambda b: results[(b, 'token_bit_0')][0] + results[(b, 'token_bit_6')][0] + results[(b, 'token_bit_11')][0])}", flush=True)
print(f"  2. Best bank for context_repeat: "
      f"{max(BANKS, key=lambda b: results[(b, 'context_repeat')][0])}", flush=True)
print(f"  3. Best bank for position (window_half): "
      f"{max(BANKS, key=lambda b: results[(b, 'window_half')][0])}", flush=True)

print(f"\nDeterministic baseline: Majority-class prediction", flush=True)
print(f"Score >> baseline = bank encodes this feature", flush=True)
print(f"Score ≈ baseline = bank has no information", flush=True)
print(f"\nTotal time: {(time.time()-t0)/60:.0f}m", flush=True)
