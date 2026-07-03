"""
BSM-X: Three experiments on state-centric retrieval.

1. Weighted Hamming — learn 256 per-bit importance weights
2. State → State retrieval — GAM returns S_{t+1}, not token
3. Structured state banks — split state into semantic groups

Builds on v40_state_dynamics.py infrastructure.
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from collections import Counter
import time, math

torch.manual_seed(42)
np.random.seed(42)

# ============ CONSTANTS ============
VOCAB_SIZE = 4096
IN_DIM = 48       # 4 tokens × 12 bits
STATE_DIM = 256
HID_DIM = 384
N_BUCKET_BITS = 14
N_BUCKETS = 1 << N_BUCKET_BITS
C = 4             # context length
STEPS = 6000
BATCH_SIZE = 16
SEQ_LEN = 64
LEARNING_RATE = 1e-3
ALPHA = 0.5       # token vs dynamics loss weight

# ============ TOKENIZER ============
from tokenizers import Tokenizer
tok = Tokenizer.from_file("data/tokenizer.json")
tok.no_truncation = True
vocab_size = tok.get_vocab_size()
print(f"Vocab size: {vocab_size}", flush=True)

# ============ DATA ============
with open("data/tinystories_train.txt") as f: text = f.read()
# TinyStories format: one story per line, with <|endoftext|> separators
# Build a proper dataset by splitting into paragraphs (~1 paragraph = 1 story snippet)
lines = [l.strip() for l in text.split("\n") if l.strip() and not l.startswith("<|endoftext|>")]
# Use full file: concatenate lines into chunks of ~200 tokens
all_text = " ".join(lines)
# Split into 512-token chunks (sliding window)
words = all_text.split()
chunk_size = 2000  # ~200 tokens in chars
stories_text = []
for i in range(0, len(words), chunk_size):
    chunk = " ".join(words[i:i+chunk_size])
    if len(chunk) > 200:  # at least a few words
        stories_text.append(chunk)

print(f"Generated {len(stories_text)} story chunks from {len(lines)} lines", flush=True)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, tok, texts, seq_len=64):
        self.tok, self.seq_len = tok, seq_len
        self.tokens = []
        for s in texts:
            ids = tok.encode(s).ids
            if len(ids) >= seq_len + 1:
                # Take as many non-overlapping seq_len+1 windows as possible
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
    return bits, tgt

def next_batch_dynamics(it):
    while True:
        try: x, y = next(it[0])
        except StopIteration: it[0] = iter(loader); continue
        break
    B, T = x.shape
    ctx_list, tgt_list, next_ctx_list = [], [], []
    for t in range(C, T - 1):
        ctx_list.append(x[:, t-C:t])
        tgt_list.append(x[:, t])
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

# ============ EVAL HELPERS ============
def evaluate_decoder(model, it, n_batches=100):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            bits, tgt = next_batch(it)
            logits, _, _, _ = model(bits)
            pred = model.predict_token(logits)
            correct += (pred == tgt).sum().item()
            total += tgt.shape[0]
    return correct / total * 100


# ============ GAM (base) ============

class GAM:
    """Graph Associative Memory — stores (S_t, S_{t+1}, token) triples."""
    def __init__(self, simhash):
        self.simhash = simhash
        self.states = []     # state integers
        self.next_states = []
        self.tokens = []
        self.buckets = [[] for _ in range(N_BUCKETS)]
        # Pre-allocate tensor state storage for fast weighted distance
        self.max_states = 200000
        self.state_tensor = torch.zeros(self.max_states, STATE_DIM)
        self.next_state_tensor = torch.zeros(self.max_states, STATE_DIM)
        self._n = 0
    
    def state_to_int(self, state_tensor):
        """Binary {-1,1} → integer."""
        bits = ((state_tensor + 1) // 2).byte().cpu().numpy()
        ints = bits.dot(1 << np.arange(STATE_DIM, dtype=np.uint64))
        return ints
    
    def state_to_bucket(self, state_tensor):
        """SimHash LSH: sign(state @ simhash.T) → bucket index."""
        proj = state_tensor @ self.simhash.T
        bucket = 0
        for i in range(N_BUCKET_BITS):
            if proj[i] > 0: bucket |= (1 << i)
        return bucket
    
    def add(self, state_tensor, next_state_tensor, token):
        state_int = self.state_to_int(state_tensor)
        next_state_int = self.state_to_int(next_state_tensor)
        bucket = self.state_to_bucket(state_tensor)
        idx = len(self.states)
        self.states.append(state_int)
        self.next_states.append(next_state_int)
        self.tokens.append(token)
        self.buckets[bucket].append(idx)
        # Store in pre-allocated tensor
        if self._n < self.max_states:
            self.state_tensor[self._n] = state_tensor.cpu()
            self.next_state_tensor[self._n] = next_state_tensor.cpu()
            self._n += 1
    
    def _weighted_dist(self, query_int, query_tensor, candidates, weights):
        """Compute weighted Hamming distance using tensor ops (fast)."""
        cand_tensor = self.state_tensor[candidates].to(query_tensor.device)
        diff = (query_tensor.unsqueeze(0) != cand_tensor).float()
        dists = (diff * weights.to(query_tensor.device).unsqueeze(0)).sum(1)
        return dists.cpu().tolist()
    
    def build(self, model, max_examples=100000):
        model.eval()
        it = [make_batch_iter()]
        n = 0
        with torch.no_grad():
            while n < max_examples:
                bits, tgt, next_bits = next_batch_dynamics(it)
                logits, state_bin, _, _ = model(bits)
                _, next_state_bin, _, _ = model(next_bits)
                for i in range(bits.shape[0]):
                    self.add(state_bin[i], next_state_bin[i], tgt[i].item())
                    n += 1
                    if n >= max_examples: break
        occ = sum(1 for bucket in self.buckets if bucket)
        print(f"  GAM: {n} states, {occ}/{N_BUCKETS} buckets, {n//max(occ,1):.0f} avg/bucket", flush=True)
    
    def query(self, state_tensor, n_candidates=200, n_neighbors=4, weights=None):
        """Query GAM with S_t, return token votes.
        
        weights: optional [STATE_DIM] array of per-bit importance.
        """
        state_int = self.state_to_int(state_tensor)
        bucket = self.state_to_bucket(state_tensor)
        
        candidate_set = set()
        for idx in self.buckets[bucket]:
            candidate_set.add(idx)
        if len(candidate_set) < n_candidates:
            for b in range(N_BUCKET_BITS):
                neighbor = bucket ^ (1 << b)
                for idx in self.buckets[neighbor]:
                    candidate_set.add(idx)
                    if len(candidate_set) >= n_candidates: break
                if len(candidate_set) >= n_candidates: break
        
        candidates = list(candidate_set)[:n_candidates]
        if not candidates:
            return Counter()
        
        if weights is not None:
            dists = self._weighted_dist(state_int, state_tensor, candidates, weights)
        else:
            dists = [(self.states[i] ^ state_int).bit_count() for i in candidates]
        
        sorted_pairs = sorted(zip(candidates, dists), key=lambda x: x[1])
        top_pairs = sorted_pairs[:n_neighbors]
        votes = Counter()
        for idx, d in top_pairs:
            w = 1.0 / (1.0 + d / (STATE_DIM * 2))
            votes[self.tokens[idx]] += w
        return votes
    
    def query_state(self, state_tensor, n_candidates=200, n_neighbors=4, weights=None):
        """Query GAM with S_t, return predicted S_{t+1} via weighted average of next_states.
        
        This is the State → State retrieval path (BSM-X Phase 2).
        Returns: predicted_next_state (tensor, 256-dim), and token votes.
        """
        state_int = self.state_to_int(state_tensor)
        bucket = self.state_to_bucket(state_tensor)
        
        candidate_set = set()
        for idx in self.buckets[bucket]:
            candidate_set.add(idx)
        if len(candidate_set) < n_candidates:
            for b in range(N_BUCKET_BITS):
                neighbor = bucket ^ (1 << b)
                for idx in self.buckets[neighbor]:
                    candidate_set.add(idx)
                    if len(candidate_set) >= n_candidates: break
                if len(candidate_set) >= n_candidates: break
        
        candidates = list(candidate_set)[:n_candidates]
        if not candidates:
            return None, Counter()
        
        if weights is not None:
            dists = self._weighted_dist(state_int, state_tensor, candidates, weights)
        else:
            dists = [(self.states[i] ^ state_int).bit_count() for i in candidates]
        
        sorted_pairs = sorted(zip(candidates, dists), key=lambda x: x[1])
        top_pairs = sorted_pairs[:n_neighbors]
        
        # Average next_states of nearest neighbors
        top_indices = [idx for idx, _ in top_pairs]
        next_tensor = self.next_state_tensor[top_indices]
        next_bits_avg = (next_tensor.float().mean(0) + 1) / 2  # [-1,1] → [0,1]
        next_state_pred = torch.where(next_bits_avg > 0.5, 1, -1)
        
        # Token votes (from nearest neighbors' tokens)
        votes = Counter()
        for idx, d in top_pairs:
            w = 1.0 / (1.0 + d / (STATE_DIM * 2))
            votes[self.tokens[idx]] += w
        
        return next_state_pred, votes


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
print("BSM-X: Training Phase", flush=True)
print("="*60, flush=True)

model = BSMv4()
opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
it = [make_batch_iter()]

t0 = time.time()
for step in range(STEPS):
    bits, tgt = next_batch(it)
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

base_acc = evaluate_decoder(model, [make_batch_iter()])
print(f"Decoder baseline: {base_acc:.2f}%", flush=True)

# ============ BUILD GAM ============
print(f"\n{'='*60}", flush=True)
print("Building GAM...", flush=True)
print("="*60, flush=True)
gam = GAM(model.simhash)
gam.build(model, max_examples=100000)

# ============ BASELINE EVALUATION ============
print(f"\n{'='*60}", flush=True)
print("Baseline: Standard GAM (unweighted Hamming)", flush=True)
print("="*60, flush=True)

def evaluate_gam_standard(model, gam, n_batches=100, weights=None):
    model.eval()
    it = [make_batch_iter()]
    correct_dec, correct_gam = 0, 0
    total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            bits, tgt = next_batch(it)
            logits, state_bin, _, _ = model(bits)
            for i in range(bits.shape[0]):
                t = tgt[i].item()
                pred_dec = model.predict_token(logits[i:i+1]).item()
                correct_dec += (pred_dec == t)
                votes = gam.query(state_bin[i], weights=weights)
                if votes:
                    pred_gam = votes.most_common(1)[0][0]
                else:
                    pred_gam = pred_dec
                correct_gam += (pred_gam == t)
                total += 1
    dec_acc = correct_dec / total * 100
    gam_acc = correct_gam / total * 100
    return dec_acc, gam_acc

dec_acc, gam_acc = evaluate_gam_standard(model, gam, weights=None)
print(f"  (unweighted) Decoder={dec_acc:.2f}%  GAM={gam_acc:.2f}%", flush=True)


# ====================================================================
# PHASE 1: WEIGHTED HAMMING — apprendi 256 pesi per bit
# ====================================================================
print(f"\n{'='*60}", flush=True)
print("PHASE 1: Weighted Hamming Distance", flush=True)
print("="*60, flush=True)

print("\nMethod 1a: Entropy-based weights...", flush=True)

# Compute per-bit entropy: how much does each bit predict the token?
# For each bit position, compute H(token | bit=1) and H(token | bit=0)
# Bits with lower conditional entropy are more informative → higher weight

from collections import defaultdict
import math as pymath

# Collect statistics
bit_token_counts = [defaultdict(lambda: [0, 0]) for _ in range(STATE_DIM)]
# bit_token_counts[j][token] = [count_bit0, count_bit1]

it = [make_batch_iter()]
n_samples = 3000
with torch.no_grad():
    for _ in range(n_samples // (BATCH_SIZE * (SEQ_LEN - C)) + 1):
        bits, tgt = next_batch(it)
        logits, state_bin, _, _ = model(bits)
        for i in range(bits.shape[0]):
            t = tgt[i].item()
            for j in range(STATE_DIM):
                b = int((state_bin[i, j].item() + 1) // 2)  # 0 or 1
                bit_token_counts[j][t][b] += 1
            n_samples -= 1
            if n_samples <= 0: break
        if n_samples <= 0: break

# Compute entropy reduction for each bit
entropy_weights = torch.zeros(STATE_DIM)
for j in range(STATE_DIM):
    total = sum(sum(counts) for counts in bit_token_counts[j].values())
    if total < 10: continue
    # H(token) overall
    token_total = sum(sum(counts) for counts in bit_token_counts[j].values())
    h_total = 0
    for t, counts in bit_token_counts[j].items():
        p = sum(counts) / token_total
        if p > 0: h_total -= p * pymath.log2(p)
    # Conditional H(token | bit=0) and H(token | bit=1)
    h_cond = 0
    for bit_val in [0, 1]:
        count_bit = sum(counts[bit_val] for counts in bit_token_counts[j].values())
        if count_bit < 2: continue
        h_bit = 0
        for t, counts in bit_token_counts[j].items():
            p = counts[bit_val] / count_bit
            if p > 0: h_bit -= p * pymath.log2(p)
        h_cond += (count_bit / token_total) * h_bit
    mi = h_total - h_cond  # mutual information
    entropy_weights[j] = max(mi, 0)

# Normalize weights to [0.1, 2.0]
if entropy_weights.max() > 0:
    entropy_weights = entropy_weights / entropy_weights.max() * 1.9 + 0.1

n_zero_weight = (entropy_weights < 0.15).sum().item()
n_high_weight = (entropy_weights > 1.5).sum().item()
print(f"  BIT IMPORTANCE: {n_zero_weight} low-info bits, {n_high_weight} high-info bits", flush=True)

# Top-10 most important bits
top_bits = entropy_weights.argsort(descending=True)[:10]
print(f"  Top-10 bit indices: {top_bits.tolist()}", flush=True)
print(f"  Top-10 weights: {[f'{entropy_weights[j].item():.3f}' for j in top_bits.tolist()]}", flush=True)

# Evaluate with entropy-based weights
dec_acc_w1, gam_acc_w1 = evaluate_gam_standard(model, gam, weights=entropy_weights)
print(f"\n  (entropy-weighted) Decoder={dec_acc_w1:.2f}%  GAM={gam_acc_w1:.2f}%", flush=True)
print(f"  Delta vs unweighted: {gam_acc_w1 - gam_acc:+.2f}%", flush=True)

# Use entropy weights for all subsequent experiments
best_weights_final = entropy_weights
print(f"\n  → Using entropy weights for all subsequent phases", flush=True)


# ====================================================================
# PHASE 2: STATE → STATE RETRIEVAL
# ====================================================================
print(f"\n{'='*60}", flush=True)
print("PHASE 2: State → State Retrieval", flush=True)
print("="*60, flush=True)

print("\nMethod: GAM query returns weighted average of neighbors' S_{t+1}", flush=True)
print("  S_t → nearest neighbors → average of their next_states → predicted S_{t+1}", flush=True)
print("  Then decode token from predicted S_{t+1} OR use token votes directly", flush=True)

def evaluate_state_retrieval(model, gam, weights, n_batches=100):
    """Evaluate state→state retrieval.
    
    For each query:
    1. GAM retrieves nearest (S_t, S_{t+1}, token) neighbors
    2. Predict S_{t+1} by averaging neighbors' S_{t+1}
    3. Decode token from predicted S_{t+1} via model.decoder
    4. Also get token votes from neighbors' tokens (standard GAM)
    """
    model.eval()
    it = [make_batch_iter()]
    correct_dec, correct_gam = 0, 0
    correct_state_dec = 0  # token from predicted S_{t+1}
    total = 0
    
    # For measuring state prediction quality:
    # compare predicted S_{t+1} with actual S_{t+1}
    bit_agreement = 0
    
    with torch.no_grad():
        for _ in range(n_batches):
            bits, tgt = next_batch(it)
            logits, state_bin, _, _ = model(bits)
            # Get actual next state
            logits_b, state_bin_b, _, _ = model(bits)  # same states, no next context needed
            
            for i in range(bits.shape[0]):
                t = tgt[i].item()
                
                # 1. Standard token decode
                pred_dec = model.predict_token(logits[i:i+1]).item()
                correct_dec += (pred_dec == t)
                
                # 2. State → State retrieval
                pred_next_state, votes = gam.query_state(state_bin[i], weights=weights)
                
                if pred_next_state is not None:
                    # Decode token from predicted next state
                    # Use decoder on the predicted next state
                    pred_next_logits = model.decoder(pred_next_state.unsqueeze(0).float())
                    pred_state_token = model.predict_token(pred_next_logits).item()
                    correct_state_dec += (pred_state_token == t)
                    
                    # Also: bit agreement between predicted S_{t+1} and actual S_{t+1}
                    # We need actual S_{t+1} — compute it from next context
                    # (Skip for now, requires next context)
                
                # 3. Standard GAM token votes
                if votes:
                    pred_gam = votes.most_common(1)[0][0]
                else:
                    pred_gam = pred_dec
                correct_gam += (pred_gam == t)
                
                total += 1
    
    dec_acc = correct_dec / total * 100
    gam_acc = correct_gam / total * 100
    state_dec_acc = correct_state_dec / total * 100
    
    return dec_acc, gam_acc, state_dec_acc

dec_acc2, gam_acc2, state_dec_acc2 = evaluate_state_retrieval(model, gam, best_weights_final, n_batches=50)
print(f"\n  Standard GAM (token votes):     {gam_acc2:.2f}%", flush=True)
print(f"  State→State → decoder:          {state_dec_acc2:.2f}%", flush=True)
print(f"  Decoder baseline:                {dec_acc2:.2f}%", flush=True)

# Compare: state-decoded vs standard GAM
delta_s2s = state_dec_acc2 - gam_acc2
delta_s2s_vs_dec = state_dec_acc2 - dec_acc2
print(f"\n  State→State vs GAM tokens:      {delta_s2s:+.2f}%", flush=True)
print(f"  State→State vs decoder:          {delta_s2s_vs_dec:+.2f}%", flush=True)

# Evaluate agreement between standard GAM and state→state predictions
print(f"\n  Agreement analysis:", flush=True)
print(f"    If state→state > GAM tokens: retrieval improves decoder", flush=True)
print(f"    If state→state < GAM tokens: GAM already optimal", flush=True)


# ====================================================================
# PHASE 3: STRUCTURED STATE BANKS
# ====================================================================
print(f"\n{'='*60}", flush=True)
print("PHASE 3: Structured State Banks", flush=True)
print("="*60, flush=True)

# Split 256 bits into banks with different roles
BANK_CONFIGS = {
    "working":  (0, 64),     # 0-63: immediate context
    "semantic": (64, 128),   # 64-127: meaning / content
    "episodic": (128, 192),  # 128-191: specific examples
    "routing":  (192, 256),  # 192-255: syntactic role
}

print(f"\nBank splits:", flush=True)
for name, (start, end) in BANK_CONFIGS.items():
    print(f"  {name:10s}: bits {start:3d}-{end-1:3d} ({end-start} bits)", flush=True)

def evaluate_bank_importance(model, gam, n_batches=100):
    """Evaluate how much each bank contributes to GAM retrieval accuracy.
    
    For each bank: disable that bank (set all bits to 0) and measure accuracy drop.
    A larger drop = more important bank.
    """
    model.eval()
    
    results = {}
    for name, (start, end) in BANK_CONFIGS.items():
        # Create mask: zero out this bank, keep others
        bank_acc = evaluate_gam_with_mask(model, gam, start, end, n_batches)
        results[name] = bank_acc
        drop = gam_acc - bank_acc
        print(f"  Without {name:10s} (bits {start:3d}-{end-1:3d}): GAM={bank_acc:.2f}% (drop={drop:+.2f}%)", flush=True)
    
    return results

def evaluate_gam_with_mask(model, gam, mask_start, mask_end, n_batches=100):
    """Evaluate GAM with a mask applied to query states:
    bits [mask_start, mask_end) are zeroed.
    """
    model.eval()
    it = [make_batch_iter()]
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            bits, tgt = next_batch(it)
            logits, state_bin, _, _ = model(bits)
            for i in range(bits.shape[0]):
                t = tgt[i].item()
                # Mask query state
                masked_state = state_bin[i].clone()
                masked_state[mask_start:mask_end] = 0
                
                votes = gam.query(masked_state)
                if votes:
                    pred = votes.most_common(1)[0][0]
                else:
                    pred = model.predict_token(logits[i:i+1]).item()
                correct += (pred == t)
                total += 1
    return correct / total * 100

print(f"\nBank ablation study (remove one bank → accuracy change):", flush=True)
bank_results = evaluate_bank_importance(model, gam)

# Per-bank weighted Hamming
print(f"\nPer-bank weighted distance:", flush=True)
print(f"  Compute separate weights for each bank, combine distances.", flush=True)

def evaluate_per_bank_weighted(model, gam, entropy_weights, n_batches=100):
    """Use per-bank weights: apply entropy weights only within each bank,
    using uniform weights for other banks."""
    model.eval()
    it = [make_batch_iter()]
    correct = 0
    total = 0
    
    # Per-bank weights: mix of entropy-based and uniform
    bank_weights = torch.ones(STATE_DIM)
    for name, (start, end) in BANK_CONFIGS.items():
        bank_weights[start:end] = entropy_weights[start:end].mean()  # uniform per-bank
    
    with torch.no_grad():
        for _ in range(n_batches):
            bits, tgt = next_batch(it)
            logits, state_bin, _, _ = model(bits)
            for i in range(bits.shape[0]):
                t = tgt[i].item()
                votes = gam.query(state_bin[i], weights=bank_weights)
                if votes:
                    pred = votes.most_common(1)[0][0]
                else:
                    pred = model.predict_token(logits[i:i+1]).item()
                correct += (pred == t)
                total += 1
    return correct / total * 100

per_bank_acc = evaluate_per_bank_weighted(model, gam, entropy_weights, n_batches=50)
print(f"\n  Per-bank uniform: {per_bank_acc:.2f}% (baseline: {gam_acc:.2f}%)", flush=True)

# Skip confusion analysis (slow, low value)
print(f"\n  (skipping bank confusion analysis — too slow)", flush=True)


# ====================================================================
# SUMMARY
# ====================================================================
print(f"\n{'='*60}", flush=True)
print("BSM-X SUMMARY", flush=True)
print("="*60, flush=True)
print(f"\nPhase 0 — Baseline:", flush=True)
print(f"  Decoder:           {base_acc:.2f}%", flush=True)
print(f"  GAM (unweighted):  {gam_acc:.2f}%", flush=True)
print(f"\nPhase 1 — Weighted Hamming:", flush=True)
print(f"  Entropy-weighted:  {gam_acc_w1:.2f}%  (Δ={gam_acc_w1-gam_acc:+.2f}%)", flush=True)
print(f"  Per-bank uniform:  {per_bank_acc:.2f}%  (Δ={per_bank_acc-gam_acc:+.2f}%)", flush=True)
print(f"\nPhase 2 — State→State Retrieval:", flush=True)
print(f"  GAM (token votes): {gam_acc2:.2f}%", flush=True)
print(f"  State→S→decoder:   {state_dec_acc2:.2f}%  (Δ={state_dec_acc2-gam_acc2:+.2f}% vs GAM)", flush=True)
print(f"\nPhase 3 — State Banks:", flush=True)
for name, acc in bank_results.items():
    drop = gam_acc - acc
    print(f"  Without {name}: GAM={acc:.2f}% (drop={drop:+.2f}%)", flush=True)
print(f"  Per-bank uniform:  {per_bank_acc:.2f}%", flush=True)
print(f"\nTotal time: {(time.time()-t0)/60:.0f}m", flush=True)
