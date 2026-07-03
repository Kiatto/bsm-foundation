"""
BSM v4.0 — State Dynamics Model
=================================
 
Paradigma: non predire token, predire traiettorie di stato.
Il token è un sottoprodotto del retrieval sulla traiettoria.

Architettura:
  Input → Encoder → S_t
                     ├── Dynamics Predictor → S_{t+1}_pred
                     └── GAM: nearest S_{t+1} → token

Training:
  Loss = α · token_loss(S_t → token) + (1-α) · dynamics_loss(S_t → S_{t+1})
  
Estremo (v4.1): dynamics_loss ONLY, nessuna supervisione token.
Il modello impara la geometria del linguaggio, non le parole.

Ipotesi: lo spazio degli stati addestrato con dinamica è più strutturato
         (traiettorie lisce, più predicibili) → GAM più efficace.
"""
import sys; sys.path.insert(0, 'training')
import torch, time, math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
from bsm.model_ff import int_to_bits, bits_to_int
from blm.tokenizer import load_tokenizer
from bsm.data import TextDataset
from torch.utils.data import DataLoader

V, C, IN_DIM, STATE_DIM = 4096, 4, 4*12, 256
STEPS = 6000
N_BUCKET_BITS = 14
N_BUCKETS = 2**N_BUCKET_BITS
ALPHA = 0.5  # weight for token loss vs dynamics loss

# ============ DATA ============

tok = load_tokenizer("checkpoints/tinystories_vocab4096.json")
with open("data/tinystories_train.txt") as f: text = f.read(1000000)
stories = [s.strip() for s in text.split("<|endoftext|>") if s.strip()][:5000]
ds = TextDataset(tok, stories, seq_len=64)
loader = DataLoader(ds, batch_size=16, shuffle=True, drop_last=True, num_workers=0)

def make_batch_iter():
    return iter(loader)

def next_batch(it):
    while True:
        try:
            x, y = next(it[0])
            break
        except StopIteration:
            it[0] = iter(loader)
            continue
    B, T = x.shape
    ctx_list, tgt_list = [], []
    for t in range(C, T):
        ctx_list.append(x[:, t-C:t]); tgt_list.append(x[:, t])
    ctx = torch.stack(ctx_list, dim=1).reshape(-1, C)
    tgt = torch.stack(tgt_list, dim=1).reshape(-1)
    bits = int_to_bits(ctx, 12).reshape(-1, IN_DIM)
    return bits, tgt

def next_batch_dynamics(it):
    """Returns (bits, tgt, next_bits) where next_bits = context shifted by 1 token."""
    while True:
        try:
            x, y = next(it[0])
            break
        except StopIteration:
            it[0] = iter(loader)
            continue
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


# ============ GAM ============

class GAM:
    def __init__(self, simhash):
        self.simhash = simhash.cpu()
        self.states = []
        self.next_states = []  # NEW: store S_{t+1} alongside each state
        self.tokens = []
        self.buckets = [[] for _ in range(N_BUCKETS)]
        self.total = 0

    def state_to_int(self, state_tensor):
        bits = ((state_tensor + 1) // 2).byte().cpu().numpy()
        packed = np.packbits(bits)
        return int.from_bytes(packed.tobytes(), 'big')

    def state_to_bucket(self, state_tensor):
        dots = self.simhash @ state_tensor
        bits = (dots > 0).int()
        bucket = 0
        for b in bits:
            bucket = (bucket << 1) | b.item()
        return bucket

    def add(self, state_tensor, next_state_tensor, token):
        state_int = self.state_to_int(state_tensor)
        next_state_int = self.state_to_int(next_state_tensor)
        bucket = self.state_to_bucket(state_tensor)
        idx = self.total
        self.states.append(state_int)
        self.next_states.append(next_state_int)
        self.tokens.append(token)
        self.buckets[bucket].append(idx)
        self.total += 1

    def build(self, model, max_examples=100000):
        model.eval()
        it = [make_batch_iter()]
        n = 0
        with torch.no_grad():
            while n < max_examples:
                bits, tgt, next_bits = next_batch_dynamics(it)
                logits, state_bin, next_state_bin, _ = model(bits)
                _, next_state_bin2, _, _ = model(next_bits)  # target future state
                for i in range(bits.shape[0]):
                    self.add(state_bin[i], next_state_bin2[i], tgt[i].item())
                    n += 1
                    if n >= max_examples: break
        model.train()
        occ = sum(1 for b in self.buckets if b)
        print(f"  GAM: {n} states, {occ} buckets, {n//max(occ,1)} avg/bucket", flush=True)

    def query_by_bucket(self, state_tensor, next_state_tensor=None, n_candidates=500, n_neighbors=4):
        """Query GAM with S_t, optionally filter by S_{t+1} similarity.
        
        NEW: if next_state_tensor is provided, score neighbors by
        SIMILARITY TO BOTH S_t AND S_{t+1} (trajectory similarity).
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
        
        if next_state_tensor is not None:
            # Trajectory similarity: combine S_t and S_{t+1} distances
            next_state_int = self.state_to_int(next_state_tensor)
            dists = []
            for i in candidates:
                d_state = (self.states[i] ^ state_int).bit_count()
                d_next = (self.next_states[i] ^ next_state_int).bit_count()
                dists.append(d_state + d_next)  # combined distance
        else:
            dists = [(self.states[i] ^ state_int).bit_count() for i in candidates]
        
        sorted_pairs = sorted(zip(candidates, dists), key=lambda x: x[1])
        top_pairs = sorted_pairs[:n_neighbors]
        votes = Counter()
        for idx, d in top_pairs:
            w = 1.0 / (1.0 + d / (STATE_DIM * 2))
            votes[self.tokens[idx]] += w
        return votes


# ============ MODEL v4 ============

class BSMv4(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = nn.Linear(IN_DIM, 384)
        self.e2 = nn.Linear(384, STATE_DIM)
        # Token decoder (optional for v4.1)
        self.decoder = nn.Linear(STATE_DIM, 12)
        # Dynamics predictor: S_t → S_{t+1}
        self.dynamics = nn.Linear(STATE_DIM, STATE_DIM)
        self.register_buffer('simhash', torch.randn(N_BUCKET_BITS, STATE_DIM))

    def forward(self, bits):
        h = torch.tanh(self.e1(bits))
        state_pre = self.e2(h)
        state_bin = torch.sign(state_pre)
        logits = self.decoder(state_pre)
        # Predict next state from current state
        next_state_pre = self.dynamics(state_pre)
        next_state_bin = torch.sign(next_state_pre)
        return logits, state_bin, next_state_bin, next_state_pre

    def dynamics_loss(self, next_state_bin, next_state_target):
        """Hamming-style loss: encourage predicted next state to match target.
        Uses binary cross-entropy on binarized states.
        """
        # Convert {-1, 1} to {0, 1} probabilities
        target_bits = ((next_state_target + 1) // 2).float()
        # next_state_bin is already binarized {-1, 1}, convert to logits
        # Actually use the pre-binarization value (next_state_pre)
        return F.binary_cross_entropy_with_logits(
            next_state_bin,  # Wait, this is post-sign. Need pre-sign.
            target_bits
        )

    def compute_loss(self, logits, targets, next_state_bin, next_state_target, ls=0.05):
        """Combined token + dynamics loss."""
        # Token loss (same as before)
        target_bits = int_to_bits(targets, 12)
        t01 = (target_bits.reshape(-1, 12) + 1) / 2
        if ls > 0: t01 = t01 * (1 - ls) + ls / 2
        token_loss = F.binary_cross_entropy_with_logits(logits.reshape(-1, 12), t01)
        
        # Dynamics loss
        target_state = ((next_state_target.detach() + 1) * 0.5).float()
        # Use the pre-binarization value from the last forward
        # But we already computed next_state_bin... need next_state_pre
        # Store it from the forward pass
        dyn_loss = F.binary_cross_entropy_with_logits(self._next_state_pre, target_state)
        
        return ALPHA * token_loss + (1 - ALPHA) * dyn_loss, token_loss.item(), dyn_loss.item()

    def predict_token(self, logits):
        return bits_to_int(torch.sign(logits), 12)

    def accuracy(self, logits, targets):
        return (self.predict_token(logits) == targets).float().mean()

    def state_to_int(self, state_tensor):
        bits = ((state_tensor + 1) // 2).byte().cpu().numpy()
        packed = np.packbits(bits)
        return int.from_bytes(packed.tobytes(), 'big')


# ============ TRAINING ============

t0 = time.time()
torch.manual_seed(1000)

model = BSMv4()
# Standard optimizer
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0, betas=(0.9, 0.95))
it = [make_batch_iter()]

for step in range(STEPS):
    bits, tgt = next_batch(it)
    # Forward (no dynamics loss during first phase)
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

bits, tgt = next_batch([make_batch_iter()])
logits, _, _, _ = model(bits)
base_acc = model.accuracy(logits, tgt).item() * 100
print(f"Decoder baseline: {base_acc:.2f}%", flush=True)

# Build GAM (standard: only S_t, no dynamics retrieval yet)
gam = GAM(model.simhash)
gam.build(model, max_examples=100000)

# Evaluate standard GAM + decoder
from collections import Counter as EvalCounter

def evaluate_gam(model, gam, use_dynamics=False):
    model.eval()
    it = [make_batch_iter()]
    correct_dec, correct_gam = 0, 0
    total = 0
    with torch.no_grad():
        for _ in range(500):
            bits, tgt = next_batch(it)
            logits, state_bin, next_state_bin, _ = model(bits)
            for i in range(bits.shape[0]):
                t = tgt[i].item()
                pred_dec = model.predict_token(logits[i:i+1]).item()
                correct_dec += (pred_dec == t)
                if use_dynamics:
                    votes = gam.query_by_bucket(state_bin[i], next_state_bin[i])
                else:
                    votes = gam.query_by_bucket(state_bin[i])
                if votes:
                    pred_gam = votes.most_common(1)[0][0]
                else:
                    pred_gam = pred_dec
                correct_gam += (pred_gam == t)
                total += 1
    model.train()
    return correct_dec / total * 100, correct_gam / total * 100

dec_acc, gam_acc = evaluate_gam(model, gam, use_dynamics=False)
print(f"Standard:  Decoder={dec_acc:.2f}%  GAM={gam_acc:.2f}%", flush=True)

# ============ PHASE 2: Train with dynamics loss ============
print(f"\n{'='*60}", flush=True)
print("Phase 2: Training with State Dynamics Loss", flush=True)
print("="*60, flush=True)

opt2 = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0, betas=(0.9, 0.95))
it = [make_batch_iter()]

dy_losses, tok_losses = [], []
for step in range(STEPS // 2):
    bits, tgt, next_bits = next_batch_dynamics(it)
    
    # Forward with explicit dynamics tracking
    h = torch.tanh(model.e1(bits))
    state_pre = model.e2(h)
    state_bin = torch.sign(state_pre)
    logits = model.decoder(state_pre)
    
    # Encode next context for target state
    h_next = torch.tanh(model.e1(next_bits))
    next_state_pre = model.e2(h_next)
    next_state_target = torch.sign(next_state_pre)
    
    # Dynamics prediction: S_t → S_{t+1}
    dyn_pred_pre = model.dynamics(state_pre)
    dyn_pred_bin = torch.sign(dyn_pred_pre)
    model._next_state_pre = dyn_pred_pre  # store for compute_loss
    
    # Combined loss
    target_bits_tok = ((int_to_bits(tgt, 12) + 1) // 2).float()
    token_loss_val = F.binary_cross_entropy_with_logits(
        logits.reshape(-1, 12), target_bits_tok.reshape(-1, 12)
    )
    target_state = ((next_state_target.detach() + 1) * 0.5).float()
    dyn_loss_val = F.binary_cross_entropy_with_logits(dyn_pred_pre, target_state)
    loss = ALPHA * token_loss_val + (1 - ALPHA) * dyn_loss_val
    
    opt2.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt2.step()
    
    if step % 1000 == 0:
        tok_losses.append((step, token_loss_val.item()))
        dy_losses.append((step, dyn_loss_val.item()))
        print(f"  step {step}: token_loss={token_loss_val.item():.4f} dyn_loss={dyn_loss_val.item():.4f}", flush=True)

# ============ EVALUATION AFTER DYNAMICS TRAINING ============
print(f"\n{'='*60}", flush=True)
print("Evaluation After Dynamics Training", flush=True)
print("="*60, flush=True)

# Rebuild GAM with dynamics-trained states
print("\nRebuilding GAM with dynamics-trained states...", flush=True)
gam2 = GAM(model.simhash)
gam2.build(model, max_examples=100000)

# Standard evaluation (no dynamics retrieval)
dec_acc2, gam_acc2 = evaluate_gam(model, gam2, use_dynamics=False)
print(f"\nStandard retrieval:", flush=True)
print(f"  Decoder={dec_acc2:.2f}%  GAM={gam_acc2:.2f}%  ", flush=True)
if gam_acc2 > gam_acc:
    print(f"  (+{gam_acc2 - gam_acc:.2f}% vs pre-dynamics)", flush=True)

# Dynamics-aware evaluation (query with both S_t and predicted S_{t+1})
dec_acc3, gam_acc3 = evaluate_gam(model, gam2, use_dynamics=True)
print(f"\nDynamics-aware retrieval (S_t + predicted S_{{t+1}}):", flush=True)
print(f"  Decoder={dec_acc3:.2f}%  GAM={gam_acc3:.2f}%", flush=True)

# ============ SUMMARY ============
print(f"\n{'='*60}", flush=True)
print("V4.0 — State Dynamics Summary", flush=True)
print("="*60, flush=True)
print(f"  Baseline decoder:          {base_acc:.2f}%", flush=True)
print(f"  Pre-dynamics GAM:          {gam_acc:.2f}%", flush=True)
print(f"  Post-dynamics decoder:     {dec_acc2:.2f}%  (Δ={dec_acc2 - base_acc:+.2f}%)", flush=True)
print(f"  Post-dynamics GAM:         {gam_acc2:.2f}%  (Δ={gam_acc2 - gam_acc:+.2f}%)", flush=True)
print(f"  GAM + dynamics retrieval:  {gam_acc3:.2f}%", flush=True)

dyn_improvement = gam_acc2 - gam_acc
traj_improvement = gam_acc3 - gam_acc
if dyn_improvement > 1.0:
    print(f"\n  ⟹ LA DINAMICA AIUTA: GAM migliora di {dyn_improvement:+.2f}%", flush=True)
    print(f"     Lo spazio degli stati è più strutturato dopo training con dinamica.", flush=True)
elif dyn_improvement > -1.0:
    print(f"\n  ⟹ NEUTRALE: GAM invariato ({dyn_improvement:+.2f}%)", flush=True)
else:
    print(f"\n  ⟹ LA DINAMICA DANNEGGIA: GAM peggiora di {dyn_improvement:+.2f}%", flush=True)
    print(f"     Il training con dinamica distrugge la struttura dello spazio.", flush=True)

if traj_improvement > dyn_improvement + 1.0:
    print(f"\n  ⟹ RETRIEVAL SU TRAIETTORIA AIUTA: {traj_improvement - dyn_improvement:+.2f}% in più", flush=True)
    print(f"     La similarità di traiettoria (S_t + S_{{t+1}}) batte S_t da solo.", flush=True)

print(f"\n  Total time: {(time.time()-t0)/60:.0f}m", flush=True)
