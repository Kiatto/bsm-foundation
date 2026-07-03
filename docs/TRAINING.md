# BLM — Training Methodology

## 1. Straight-Through Estimator (STE)

```
Filosofia:
  Forward:  w_bin = sign(w_hidden)    # {-1, +1}
            y = w_bin * x             # operazione binaria

  Backward: dL/dw_hidden ≈ dL/dy * x  # STE: sign() ≈ identity
            dL/dx ≈ dL/dy * w_bin

  Update:   w_hidden = w_hidden - lr * dL/dw_hidden
            w_bin = sign(w_hidden)    # next forward
```

Problema: gradiente rumoroso. Soluzione: gradient clipping, LR piccolo, warmup.

## 2. Hidden FP32 Weights

```
Ogni peso binario ha ombra FP32:

+----------------------+
|  w_hidden (FP32)     |  <- aggiornato dal gradiente
|       |              |
|       v sign()       |  <- STE
|  w_bin (binary)      |  <- usato nel forward
+----------------------+
```

In PyTorch:
```python
self.w_hidden = nn.Parameter(torch.randn(D_out, D_in))

def forward(self, x):
    w = torch.sign(self.w_hidden)
    return binary_matmul(x, w)
```

## 3. Weight Initialization

```
w_hidden ~ Uniform(-1, +1) * gain
gain = sqrt(2.0 / D_in)     # He initialization

Motivazione:
  - distribuiti attorno a 0
  - troppo piccoli -> segnale debole
  - troppo grandi -> saturazione precoce

Embedding:
  embed_hidden = Uniform(-0.5, +0.5)
  embed_binary = sign(embed_hidden)
```

## 4. Optimizer

Adam su hidden FP32 weights. Accorgimenti:

| Parametro | Valore | Motivazione |
|---|---|---|
| Learning rate | 1e-4 a 5e-4 | Gradienti STE rumorosi |
| Gradient clipping | max_norm = 1.0 | Evita esplosione gradiente |
| Weight decay | 1e-5 | Incoraggia pesi piccoli -> sign() stabile |
| Warmup | 1000 step lineare | Stabilizzazione iniziale |
| Schedule | Cosine decay a 0 | Convergenza graduale |

## 5. Binary Optimizer (Bop) — Alternativa

```python
# Invece di Adam, conta i sign-change:
# Se gradiente punta in direzione opposta al peso
# per K step consecutivi, allora flip il bit.
#
# threshold: numero di step prima del flip
# lr: scaling del gradiente prima del confronto

for step:
    momentum = beta * momentum + (1-beta) * gradient
    if momentum * w_hidden < -threshold:
        w_hidden = -w_hidden  # flip
```

Bop è più lento ma stabile per reti profonde.

## 6. Loss Function

```
Loss: Cross-Entropy standard

L = -log(softmax(logits)[target])

logits[v] = D - hamming_distance(state, embed_binary[v])
  dove hamming_distance = popcount(state XOR embed_binary[v])
  range: [0, D], invertito a similarita: [D, 0]

Temperature = 1.0 (logits in range appropriato)
```

Non serve Label Smoothing: similarita di Hamming produce distribuzioni intrinsecamente morbide.

## 7. Training Loop — PyTorch

```python
class BinaryCellularMixer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.D = config.dimension
        self.V = config.vocab_size
        self.N = config.num_layers
        self.embed_hidden = nn.Parameter(torch.randn(self.V, self.D) * 0.1)

        self.layers = nn.ModuleList([
            CellularBlock(config) for _ in range(self.N)
        ])

    def forward(self, input_ids):
        embed = torch.sign(self.embed_hidden)
        states = embed[input_ids]

        for layer in self.layers:
            states = layer(states)

        final = states[:, -1, :]
        logits = torch.matmul(final, embed.T)
        # matmul con {-1, +1} equivalente a D - 2*hamming_distance
        return logits


class CellularBlock(nn.Module):
    def __init__(self, config):
        self.D = config.dimension
        self.D_hidden = config.d_hidden

        self.W1_hidden = nn.Parameter(torch.randn(self.D_hidden, self.D))
        self.t1 = nn.Parameter(torch.zeros(self.D_hidden))
        self.W2_hidden = nn.Parameter(torch.randn(self.D, self.D_hidden))
        self.t2 = nn.Parameter(torch.zeros(self.D))
        self.Wg_hidden = nn.Parameter(torch.randn(self.D, self.D))
        self.tg = nn.Parameter(torch.zeros(self.D))

    def forward(self, states):
        B, L, D = states.shape

        left = torch.roll(states, 1, dims=1)
        right = torch.roll(states, -1, dims=1)
        neighbor = left ^ states ^ right

        W1 = torch.sign(self.W1_hidden)
        W2 = torch.sign(self.W2_hidden)
        Wg = torch.sign(self.Wg_hidden)

        h = torch.matmul(neighbor, W1.T)
        h = torch.where(h > self.t1, 1.0, -1.0)

        out = torch.matmul(h, W2.T)
        out = torch.where(out > self.t2, 1.0, -1.0)

        gate_val = torch.matmul(neighbor, Wg.T)
        gate = torch.where(gate_val > self.tg, 1.0, -1.0)

        gate_mask = (gate > 0).float()
        new_states = states * gate_mask + out * (1 - gate_mask)

        return new_states
```

## 8. Memory Requirements (Training)

```
Per D=2048, D_hidden=1024, N=6:

+------------------+------------+
| Componente       | Dimensione |
+------------------+------------+
| Pesi binari      | ~3 MB      |
| Hidden FP32      | ~12 MB     |
| Adam state       | ~24 MB     |
| Totale training  | ~40 MB     |
+------------------+------------+
```

Per la POC, PyTorch con sign() e matmul e' sufficiente. Versione di produzione userebbe custom CUDA kernel per XNOR+POPCOUNT.

## 9. Key Challenges

1. **Vanishing gradient**: STE puo' far morire il gradiente in reti profonde. D_hidden >= D aiuta.
2. **Collapse degli stati**: Stati di layer diversi collassano allo stesso pattern. 
3. **Threshold learning**: Le soglie binarie possono non convergere.
4. **Output layer bottleneck**: Similarita con tutto il vocabolario e' costosa.
