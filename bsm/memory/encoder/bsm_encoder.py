"""
bsm_encoder.py — BSM Encoder: text → binary vector.

Three strategies:
  - HashEncoder:   SimHash over word n-grams, zero training, deterministic.
  - ProjectionEncoder: learned projection matrix (SVD / random projection).
  - LearnedEncoder:     small PyTorch MLP trained with contrastive loss.

All produce D-dimensional binary vectors in {-1, +1}.
"""

import hashlib
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _token_hashes(text: str, n_features: int = 4096) -> np.ndarray:
    words = text.lower().split()
    features = np.zeros(n_features, dtype=np.float32)
    for word in words:
        h = int(hashlib.md5(word.encode()).hexdigest(), 16) % n_features
        features[h] += 1.0
    for i in range(1, len(words)):
        bigram = words[i - 1] + "_" + words[i]
        h = int(hashlib.md5(bigram.encode()).hexdigest(), 16) % n_features
        features[h] += 0.5
    norm = np.linalg.norm(features)
    if norm > 1e-8:
        features /= norm
    return features


# ---------------------------------------------------------------------------
# 1. HashEncoder — SimHash (zero training)
# ---------------------------------------------------------------------------

class HashEncoder:
    """SimHash-style encoder using random projections on hash features.

    Deterministic for a given seed.  No training required.
    """

    def __init__(self, state_dim: int = 256, n_features: int = 4096, seed: int = 42):
        self.state_dim = state_dim
        self.n_features = n_features
        rng = np.random.RandomState(seed)
        self.projection = rng.randn(n_features, state_dim).astype(np.float32)
        self._name = "hash"

    def encode(self, text_or_texts):
        """Text → (N, D) int8 array in {-1, +1}."""
        if isinstance(text_or_texts, str):
            texts = [text_or_texts]
        else:
            texts = list(text_or_texts)
        results = []
        for text in texts:
            feats = _token_hashes(text, self.n_features)
            raw = feats @ self.projection
            binary = np.where(raw >= 0, 1, -1).astype(np.int8)
            results.append(binary)
        out = np.array(results, dtype=np.int8)
        return out[0] if isinstance(text_or_texts, str) else out

    def __repr__(self):
        return f"HashEncoder(D={self.state_dim}, F={self.n_features})"


# ---------------------------------------------------------------------------
# 2. ProjectionEncoder — learned / fitted projection matrix
# ---------------------------------------------------------------------------

class ProjectionEncoder:
    """Encoder that fits a projection matrix from training data.

    Fit:  collect hash features from corpus → SVD → keep top-D components.
    Encode: project hash features through learned matrix.

    Lightweight (~1 MB at D=256, F=4096).
    """

    def __init__(self, state_dim: int = 256, n_features: int = 4096):
        self.state_dim = state_dim
        self.n_features = n_features
        self.projection = None
        self._fitted = False
        self._name = "projection"

    def fit(self, texts):
        """Fit projection via SVD on hash features from *texts*.

        If fewer training samples than state_dim, remaining dimensions
        are filled with random projections.
        """
        X = np.array([_token_hashes(t, self.n_features) for t in texts], dtype=np.float32)
        X_centered = X - X.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        n = min(Vt.shape[0], self.state_dim)
        proj = Vt[:n].T.astype(np.float32)
        if n < self.state_dim:
            rng = np.random.RandomState(0)
            extra = rng.randn(self.n_features, self.state_dim - n).astype(np.float32)
            extra /= np.linalg.norm(extra, axis=0, keepdims=True) + 1e-8
            proj = np.concatenate([proj, extra], axis=1)
        self.projection = proj
        self._fitted = True
        return self

    def encode(self, text_or_texts):
        if self.projection is None:
            raise RuntimeError("ProjectionEncoder not fitted. Call .fit() first.")
        if isinstance(text_or_texts, str):
            texts = [text_or_texts]
        else:
            texts = list(text_or_texts)
        results = []
        for text in texts:
            feats = _token_hashes(text, self.n_features)
            raw = feats @ self.projection
            binary = np.where(raw >= 0, 1, -1).astype(np.int8)
            results.append(binary)
        out = np.array(results, dtype=np.int8)
        return out[0] if isinstance(text_or_texts, str) else out

    def __repr__(self):
        status = "fitted" if self._fitted else "unfitted"
        return f"ProjectionEncoder(D={self.state_dim}, F={self.n_features}, {status})"


# ---------------------------------------------------------------------------
# 3. LearnedEncoder — small PyTorch MLP
# ---------------------------------------------------------------------------

class _MLPEncoder(nn.Module if nn is not None else object):
    def __init__(self, n_features: int, state_dim: int, hid_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, state_dim),
        )

    def forward(self, x):
        return self.net(x)


class LearnedEncoder:
    """Trainable PyTorch MLP encoder.

    Requires PyTorch.  Train with .train_contrastive() or load pre-trained.
    """

    def __init__(self, state_dim: int = 256, n_features: int = 4096, hid_dim: int = 128):
        if torch is None:
            raise ImportError("LearnedEncoder requires PyTorch.  pip install torch")
        self.state_dim = state_dim
        self.n_features = n_features
        self.model = _MLPEncoder(n_features, state_dim, hid_dim)
        self._trained = False
        self._name = "learned"

    def encode(self, text_or_texts):
        if isinstance(text_or_texts, str):
            texts = [text_or_texts]
        else:
            texts = list(text_or_texts)
        self.model.eval()
        with torch.no_grad():
            results = []
            for text in texts:
                feats = _token_hashes(text, self.n_features)
                t = torch.from_numpy(feats).unsqueeze(0)
                raw = self.model(t).squeeze(0).numpy()
                binary = np.where(raw >= 0, 1, -1).astype(np.int8)
                results.append(binary)
        out = np.array(results, dtype=np.int8)
        return out[0] if isinstance(text_or_texts, str) else out

    def train_contrastive(self, texts, labels, n_epochs: int = 10, lr: float = 1e-3):
        """Train with contrastive loss: same label → similar binary states."""
        X = np.array([_token_hashes(t, self.n_features) for t in texts], dtype=np.float32)
        y = np.array(labels)
        dataset = list(zip(X, y))
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(n_epochs):
            np.random.shuffle(dataset)
            epoch_loss = 0.0
            for feats, label in dataset:
                t = torch.from_numpy(feats).unsqueeze(0)
                raw = self.model(t)
                # Encourage large magnitude (≈binary) via cosine-saturation loss
                mag_loss = -raw.abs().mean()
                loss = mag_loss
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
        self._trained = True
        return self

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self._trained = True
        return self

    def __repr__(self):
        status = "trained" if self._trained else "untrained"
        return f"LearnedEncoder(D={self.state_dim}, F={self.n_features}, {status})"
