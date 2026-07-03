"""
Measure the intrinsic dimension of the binary state manifold.
No sklearn dependency — manual nearest neighbor search.
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import time, math

torch.manual_seed(42)
np.random.seed(42)

IN_DIM = 48; C = 4
STEPS = 2000; BATCH_SIZE = 16; SEQ_LEN = 64; LR = 1e-3

DIMS = [128, 256, 512, 1024]
HID_DIMS = {128: 192, 256: 384, 512: 768, 1024: 1536}

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
        self.tok=tok; self.seq_len=seq_len; self.tokens=[]
        for s in texts:
            ids = tok.encode(s).ids
            if len(ids) >= seq_len+1:
                for start in range(0, len(ids)-seq_len, seq_len//2):
                    self.tokens.append(torch.tensor(ids[start:start+seq_len+1]))
    def __len__(self): return len(self.tokens)
    def __getitem__(self, i):
        t=self.tokens[i]; return t[:self.seq_len], t[1:self.seq_len+1]

ds = TextDataset(tok, stories_text, seq_len=SEQ_LEN)
loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
def make_iter(): return iter(loader)
def next_batch(it):
    while True:
        try: x,y=next(it[0])
        except StopIteration: it[0]=iter(loader); continue
        break
    B,T=x.shape
    ctx_list,tgt_list=[],[]
    for t in range(C,T):
        ctx_list.append(x[:,t-C:t]); tgt_list.append(x[:,t])
    ctx=torch.stack(ctx_list,dim=1).reshape(-1,C)
    tgt=torch.stack(tgt_list,dim=1).reshape(-1)
    bits=int_to_bits(ctx,12).reshape(-1,IN_DIM)
    return bits,tgt
def int_to_bits(x,bits=12):
    return ((x.unsqueeze(-1)>>torch.arange(bits,device=x.device))&1).float()

class DSGModel(nn.Module):
    def __init__(self, state_dim, hid_dim):
        super().__init__()
        self.state_dim = state_dim
        self.e1 = nn.Linear(IN_DIM, hid_dim)
        self.e2 = nn.Linear(hid_dim, state_dim)
        self.decoder = nn.Linear(state_dim, 12)
    def forward(self, bits):
        h = torch.tanh(self.e1(bits))
        sp = self.e2(h); sb = torch.sign(sp)
        return self.decoder(sp), sb, sp

# =============================================
# Intrinsic Dimension Estimators (no sklearn)
# =============================================

def _hamming_pairwise(X):
    """Compute pairwise Hamming distances for ±1 data.
    X: [N, D] int8 with values ±1
    Returns: [N*(N-1)//2] distances
    """
    N = X.shape[0]
    # For each pair (i,j): sum(x_i != x_j) = D/2 - 0.5 * sum(x_i * x_j)
    # Since x_i in {-1, +1}: x_i != x_j when x_i * x_j = -1
    # So x_i != x_j = (1 - x_i * x_j) / 2
    # sum(x_i != x_j) = D/2 - 0.5 * dot(x_i, x_j)
    # But this is for all pairs. Let's use batch computation.
    batch = 500
    dists = []
    for i in range(0, N, batch):
        end = min(i + batch, N)
        # X[i:end] @ X.T = [batch, N] dot products
        dot = X[i:end] @ X.T  # [batch, N]
        hd = (D - dot) // 2  # convert dot product to Hamming
        # Extract upper triangle for this batch
        for bi in range(end - i):
            idx = i + bi
            # Remove self and lower triangle
            # All pairs where j > idx
            if idx + 1 < N:
                dists.append(hd[bi, idx + 1:])
    return np.concatenate(dists) if dists else np.array([])

def _hamming_pairwise_fast(X):
    """Faster pairwise Hamming using chunked approach."""
    N, D = X.shape
    X_int8 = np.packbits((X + 1) // 2, axis=1)  # [N, D//8]
    n_bytes = X_int8.shape[1]
    dists = []
    chunk = 200
    for i in range(0, N, chunk):
        end = min(i + chunk, N)
        for j in range(i + 1, N, chunk):
            j_end = min(j + chunk, N)
            # Block of pairs
            for bi in range(end - i):
                row = X_int8[i + bi]
                for bj in range(max(j, i + bi + 1) - j, j_end - j):
                    if j + bj <= i + bi: continue
                    xor = np.bitwise_xor(row.astype(np.uint16), X_int8[j + bj].astype(np.uint16))
                    hd = int(np.unpackbits(xor.view(np.uint8)).sum())
                    dists.append(hd)
    return np.array(dists)

def _hamming_pairwise_blocked(X):
    """Blocked pairwise Hamming. X: [N, D] int8 with ±1.
    Uses numpy dot product: dot in {-D, -D+2, ..., D}
    Hamming = (D - dot) / 2
    """
    N, D = X.shape
    Xf = X.astype(np.float32)
    dists = []
    block = 200
    for i in range(0, N, block):
        end = min(i + block, N)
        block_X = Xf[i:end]  # [block, D]
        # Dot with all points after i
        dot = block_X @ Xf.T  # [block, N]
        # Only keep j > i + bi (within each row)
        for bi in range(end - i):
            idx = i + bi
            # j > idx
            if idx + 1 < N:
                d = (D - dot[bi, idx + 1:]) // 2
                dists.append(d)
    return np.concatenate(dists)

def _knn_hamming(X, k=3):
    """Compute k nearest neighbors for each point using Hamming distance.
    X: [N, D] int8 with values ±1
    Returns: [N, k] distances, [N, k] indices
    """
    N, D = X.shape
    Xf = X.astype(np.float32)
    all_dists = np.zeros((N, k))
    all_indices = np.zeros((N, k), dtype=np.int32)
    batch = 500
    for i in range(0, N, batch):
        end = min(i + batch, N)
        dot = Xf[i:end] @ Xf.T  # [batch, N]
        hd = (D - dot.astype(np.int32)) // 2  # [batch, N]
        for bi in range(end - i):
            row = hd[bi]
            row[i + bi] = D + 1  # mask self
            # Find k smallest
            idxs = np.argpartition(row, k)[:k]
            dists = row[idxs]
            # Sort within k
            order = np.argsort(dists)
            all_dists[i + bi] = dists[order]
            all_indices[i + bi] = idxs[order]
    return all_dists, all_indices

def participation_ratio(X):
    """Effective number of dimensions from covariance eigenspectrum."""
    C = np.cov(X.T)
    eigvals = np.linalg.eigvalsh(C)
    eigvals = eigvals[::-1]
    eigvals = eigvals[eigvals > 1e-12]
    total_var = np.sum(eigvals)
    if np.sum(eigvals**2) == 0: return 0, eigvals
    pr = total_var**2 / np.sum(eigvals**2)
    return pr, eigvals

def twonn_dimension_manual(X):
    """TwoNN estimator without sklearn. Uses manual KNN."""
    print(f"      TwoNN: computing KNN...", end=" ", flush=True)
    N, D = X.shape
    nbr_dists, _ = _knn_hamming(X, k=3)
    r1 = nbr_dists[:, 1].astype(np.float64)
    r2 = nbr_dists[:, 2].astype(np.float64)
    print(f"r1_mean={r1.mean():.1f} r2_mean={r2.mean():.1f}", end=" ", flush=True)
    mu = r2 / r1
    valid = (r1 > 0) & np.isfinite(mu) & (mu > 1.0)
    mu = mu[valid]
    print(f"valid={len(mu)}/{N}", end=" ", flush=True)
    if len(mu) < 20:
        return np.nan
    mu_sorted = np.sort(mu)
    F = np.arange(1, len(mu_sorted) + 1) / len(mu_sorted)
    valid_mu = mu_sorted > 1.0
    if valid_mu.sum() < 10:
        return np.nan
    x = np.log(mu_sorted[valid_mu])
    y = np.log(1 - F[valid_mu] + 1e-15)
    d = -np.sum(x * y) / np.sum(x ** 2)
    return d

def correlation_dimension_manual(X):
    """Correlation dimension from sampled pair distances."""
    N, D = X.shape
    rng = np.random.RandomState(42)
    n_pairs = min(30000, N * (N - 1) // 2)
    idx1 = rng.randint(0, N, n_pairs * 2)
    idx2 = rng.randint(0, N, n_pairs * 2)
    mask = idx1 != idx2
    idx1 = idx1[mask][:n_pairs]
    idx2 = idx2[mask][:n_pairs]
    dot = np.sum(X[idx1].astype(np.float32) * X[idx2].astype(np.float32), axis=1)
    dists = ((D - dot) / 2).astype(np.float64)
    n_bins = min(40, D // 2)
    r_vals = np.geomspace(max(2, D // 200), D // 3, n_bins)
    Cr = np.array([(dists < r).mean() for r in r_vals])
    valid = (Cr > 1e-8) & (Cr < 0.3)
    if valid.sum() < 3:
        return np.nan
    x = np.log(r_vals[valid]); y = np.log(Cr[valid])
    slope, _ = np.polyfit(x, y, 1)
    return slope

# =============================================
# Main
# =============================================
all_ids = {}
t_start = time.time()

for D in DIMS:
    H = HID_DIMS[D]
    print(f"\n{'='*60}")
    print(f"  D={D}, H={H}")
    print(f"{'='*60}")
    t_d = time.time()

    # Train
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
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 1000 == 0:
            print(f"    step {step}: loss={loss.item():.4f}", flush=True)

    # Collect states
    model.eval()
    states_bin = []
    it_data = [make_iter()]
    n_col = 0
    with torch.no_grad():
        while n_col < 2000:
            bits, _ = next_batch(it_data)
            _, sb, _ = model(bits)
            for i in range(bits.shape[0]):
                states_bin.append(sb[i].cpu().to(torch.int8))
                n_col += 1
                if n_col >= 2000: break

    X = torch.stack(states_bin).numpy()  # [2000, D], ±1 as int8
    print(f"  States: {X.shape}, dtype={X.dtype}")

    # ---- Flags for speed ----
    # Use float32 for dot products
    Xf = X.astype(np.float32)
    N_eff = min(X.shape[0], 1500)

    # ---- Participation Ratio ----
    print(f"  PR:", end=" ", flush=True)
    pr_bin, ev_bin = participation_ratio(X)
    pr_rand, _ = participation_ratio(np.random.choice([-1, 1], size=X.shape).astype(np.float64))
    print(f"model={pr_bin:.1f}  random={pr_rand:.1f}", flush=True)

    # ---- TwoNN ----
    print(f"  TwoNN:", end=" ", flush=True)
    # Use subset for speed
    idx_sub = np.random.choice(X.shape[0], min(1000, N_eff), replace=False)
    d2 = twonn_dimension_manual(X[idx_sub])
    print(f"d={d2:.1f}" if not np.isnan(d2) else "d=N/A", flush=True)

    # ---- Correlation dimension ----
    print(f"  CorrDim:", end=" ", flush=True)
    dc = correlation_dimension_manual(X)
    print(f"d={dc:.1f}" if not np.isnan(dc) else "d=N/A", flush=True)

    # ---- Frozen bits ----
    var_bin = np.var(X.astype(np.float32), axis=0)
    frozen = int(np.sum(var_bin < 0.01))
    lowvar = int(np.sum(var_bin < 0.1))
    active = D - frozen
    print(f"  Active bits: {active}/{D}  (frozen={frozen}, lowvar={lowvar})", flush=True)

    # ---- Spectrum decay (raw would need the pre-activation values) ----
    # Use binary PR as estimate of effective rank
    n90 = int(np.searchsorted(np.cumsum(ev_bin) / np.sum(ev_bin), 0.9) + 1)

    all_ids[D] = {
        "pr_binary": round(pr_bin, 1),
        "pr_random": round(pr_rand, 1),
        "twonn": round(d2, 1) if not np.isnan(d2) else None,
        "corrdim": round(dc, 1) if not np.isnan(dc) else None,
        "n90": n90,
        "active_bits": active,
        "frozen_bits": frozen,
        "lowvar_bits": lowvar,
        "time": round((time.time() - t_d) / 60, 1),
    }

    print(f"  Time: {all_ids[D]['time']}m", flush=True)

# =============================================
# FINAL TABLE
# =============================================
print(f"\n{'='*60}")
print("INTRINSIC DIMENSION — FINAL TABLE")
print(f"{'='*60}")

header = f"{'Method':25s}"
for D in DIMS: header += f" {D:>8d}"
print(header)
print("-" * 60)

rows = [
    ("PR (binary)", "pr_binary"),
    ("PR (random)", "pr_random"),
    ("TwoNN", "twonn"),
    ("CorrDim", "corrdim"),
    ("Active bits", "active_bits"),
    ("Frozen bits", "frozen_bits"),
    ("Low-var bits (<0.1)", "lowvar_bits"),
]

for name, key in rows:
    line = f"{name:25s}"
    for D in DIMS:
        v = all_ids[D].get(key)
        if v is None:
            line += f" {'---':>8s}"
        elif isinstance(v, (int, np.integer)):
            line += f" {int(v):>8d}"
        else:
            line += f" {v:>8.2f}"
    print(line)

print(f"\nInvariant analysis:")
for name, key in rows:
    vals = [all_ids[D].get(key) for D in DIMS]
    vals = [float(v) for v in vals if v is not None]
    if len(vals) >= 3:
        mean = np.mean(vals); cv = np.std(vals) / max(mean, 1e-8)
        status = "STABLE" if cv < 0.1 else ("TRENDING" if cv < 0.25 else "SCALING")
        print(f"  [{status:>8s}] {name:20s} μ={mean:.1f} CV={cv:.3f}")

total = time.time() - t_start
print(f"\nTotal: {total/60:.0f}m")
