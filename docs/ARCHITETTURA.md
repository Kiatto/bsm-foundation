# BLM — Binary Language Model

## Architettura Completa

---

## 1. Analisi Critica delle Architetture Esistenti

### 1.1 Transformer

```
         +------------------------------------------+
         |               Output Token                |
         +------------------------------------------+
                          ▲
                     +--------+
                     | Linear  |  ← FP32 weights
                     +--------+
                          ▲
                     +---------+
                     | LayerNorm|  ← mean/var (FP32)
                     +---------+
                          ▲
                     +---------+
                     |  Add    |  ← FP32 addition
                     +---------+
                          ▲
                     +---------+
                     |  FFN    |  ← GELU (FP32), Linear (FP32)
                     +---------+
                          ▲
                     +---------+
                     |  Add    |  ← FP32 addition
                     +---------+
                          ▲
                     +---------+
                     | Attention| ← softmax (exp/div), QK^T (FP32)
                     +---------+
                          ▲
                     +---------+
                     | LayerNorm|  ← mean/var (FP32)
                     +---------+
                          ▲
                     +---------+
                     | Embed   |  ← FP32 lookup
                     +---------+
                          ▲
                     +---------+
                     | Input   |
                     +---------+
```

**Problemi con binario:**

| Componente | Perché NON funziona in binario |
|---|---|
| Softmax | Richiede esponenziale e divisione — FP obbligatorie |
| QK^T | Dot product binario perde informazione, range [-D, +D] |
| LayerNorm | Media e varianza — FP ineliminabili |
| GELU/SiLU | Attivazioni continue non binarizzabili |
| Residual Add | XOR è involuzione, non replicabile |
| Deep stacking | Errore di quantizzazione amplificato |

**Verdetto**: Il Transformer NON è adatto a un modello binary-first. Ogni componente richiede FP. La binarizzazione post-training degrada la perplexity irreversibilmente sotto 1B parametri.

### 1.2 Mamba / State Space Models

```
h_t = A h_{t-1} + B x_t     ← moltiplicazione matriciale FP
y_t = C h_t + D x_t         ← moltiplicazione matriciale FP
```

**Verdetto**: SSM binario è una contraddizione. La matrice A richiede decay differenziabile. Selective scan richiede moltiplicazioni. Non adatto.

### 1.3 RNN moderne (LSTM, GRU)

```
i_t = sigma(W_i * [h_{t-1}, x_t])   ← sigmoide (FP)
f_t = sigma(W_f * [h_{t-1}, x_t])   ← sigmoide (FP)
c_t = f_t * c_{t-1} + i_t * tanh(W_c * [h_{t-1}, x_t])  ← FP
h_t = o_t * tanh(c_t)               ← FP
```

**Verdetto**: RNN classiche sono intrinsecamente continue. Gate (sigmoide) non binarizzabili. Non adatte.

### 1.4 Memory Networks

**Potenziale**: Memoria binaria possibile (hash table). Retrieval con distanza di Hamming invece di coseno.

**Problemi**: Weighted sum richiede FP. Scrittura richiede gradiente.

**Verdetto**: Parzialmente adattabile ma weighted sum resta un problema.

### 1.5 Binary State Machines (BSM)

```
stato: S ∈ {0,1}^D
input: x ∈ {0,1}^D
transizione: S' = f(S, x)    ← funzione booleana
output: y = g(S)             ← funzione booleana
```

**Potenziale**: Stati discreti naturali, interpretabile, efficiente.

**Problemi**: Memoria limitata (D bit). Funzione di transizione da apprendere. Collasso stati possibile.

**Verdetto**: Promettente come building block.

### 1.6 Cellular Automata

```
s_i(t+1) = f(s_{i-1}(t), s_i(t), s_{i+1}(t))    ← regola locale
aggiornamento sincrono
```

**Potenziale**: Stati binari naturali, parallelo, scalabile, receptive field cresce.

**Verdetto**: BASE ARCHITETTURALE IDEALE.

---

## 2. Binary Cellular Mixer (BCM)

### 2.1 Architettura Proposta

```
                  +------------------------------------------+
                  |        TOKEN OUTPUT (next token)          |
                  +------------------------------------------+
                                  ^
                         +------------------+
                         |  Output Head     |
                         |  (Hamming Sim)   |
                         +------------------+
                                  ^
                    +-------------+-------------+
                    |    CellularBlock x N       |
                    |  +---------------------+  |
                    |  |  per ogni posizione  |  |
                    |  |  s' = BinMLP(       |  |
                    |  |    s_l XOR s XOR s_r,|  |
                    |  |    t_i               |  |
                    |  |  )                  |  |
                    |  +---------------------+  |
                    +---------------------------+
                                  ^
                    +----------------------------+
                    |    Binary Temporal Mixer    |
                    |    (Conv1D binaria)         |
                    +----------------------------+
                                  ^
                    +----------------------------+
                    |    Binary Channel Mixer     |
                    |    (feature permutation)    |
                    +----------------------------+
                                  ^
                         +------------------+
                         |  Embed Table     |
                         |  (binaria)       |
                         +------------------+
                                  ^
                         +------------------+
                         |  Tokenizer (BPE) |
                         +------------------+
                                  ^
                         +------------------+
                         |  Input Text      |
                         +------------------+
```

### 2.2 Hyperparameters

```
D = 2048          # Dimensione vettore binario (multiplo di 64)
V = 4096          # Vocabolario (BPE)
N = 6             # CellularBlock
L = 256           # Contesto massimo
K = 3             # Dimensione vicinato
D_hidden = 1024   # Hidden dimension del BinMLP
```

Formato dati: BitVector[D] = array di uint64[D/64] = 32 uint64 = 256 byte.

### 2.3 Forward Pass

```
function forward(tokens: List[int]) -> List[float]:
    ids = tokenizer.encode(tokens)

    states = [embed_table[id] for id in ids]
    # shape: [L, D/64]

    for layer in range(N):
        states = binary_channel_mixer(states)
        states = binary_temporal_mixer(states)

        new_states = copy(states)
        for i in range(1, L-1):
            left   = states[i-1]
            center = states[i]
            right  = states[i+1]

            neighbor = left XOR center XOR right
            input_vec = neighbor XOR token_embed[ids[i]]

            hidden = binary_linear(input_vec, layer.W1, layer.t1)
            output = binary_linear(hidden, layer.W2, layer.t2)

            gate = binary_gate(input_vec, layer.Wg, layer.tg)
            new_states[i] = (center AND gate) XOR (output AND NOT gate)

        states = new_states

    final = states[-1]

    logits = zeros(V)
    for v in range(V):
        dist = 0
        for chunk in range(D/64):
            dist += popcount(final[chunk] XOR embed_table[v][chunk])
        logits[v] = D - dist

    probs = softmax(logits / temperature)
    return probs
```

### 2.4 BinMLP

```
BinMLP(x: BitVector) -> BitVector:
    for i in range(D_hidden):
        dot = 0
        for c in range(D/64):
            match = ~(x[c] XOR W1[i][c])
            dot += popcount(match)
        h[i] = dot > threshold_1[i]

    for i in range(D):
        dot = 0
        for c in range(D_hidden/64):
            match = ~(h[c] XOR W2[i][c])
            dot += popcount(match)
        y[i] = dot > threshold_2[i]

    return y
```

### 2.5 Binary Gate

```
Funzione di gate binario:
    gate = popcount(XNOR(W_g, x)) > threshold_g
    out = (x AND gate) OR (mlp_out AND NOT gate)
```

Analogo binario della residual connection.

### 2.6 Binary Temporal Mixer (Conv1D)

```
Input:  States[T][D]
Kernel: W[K][D]  (K=3)

For each position i:
    for k in range(K):
        pos = i + k - 1
        if 0 <= pos < T:
            acc += popcount(~(states[pos] XOR W[k]))
    out[i] = acc > threshold
```

### 2.7 Binary Channel Mixer

```
Input:  States[T][D]
G = 64  # bit per gruppo
N_groups = D / G  # 32 per D=2048

for g in range(N_groups):
    group_data = extract_bits(states, g*G, (g+1)*G)
    group_out = binary_lut(group_data, LUT[g])
    write_bits(out_states, g*G, group_out)
```

### 2.8 Schema a Blocchi

```
+-----------------------------------------------------------+
|                    BCM-2048 Architecture                   |
+-----------------------------------------------------------+
|                                                           |
|  Token IDs ---> Embed Table ---> [States: L x 2048 bit]  |
|                                                           |
|  +---------------------------------------------------+   |
|  |  CellularBlock x 6                                 |   |
|  |  +----------+  +----------+  +----------------+   |   |
|  |  | Channel  |  | Temporal |  | Cellular       |   |   |
|  |  | Mixer    |->| Mixer    |->| Update (BinMLP) |   |   |
|  |  +----------+  +----------+  +----------------+   |   |
|  +---------------------------------------------------+   |
|                                                           |
|  Final State ---> Hamming Distance ---> Softmax ---> P   |
|                                                           |
+-----------------------------------------------------------+
```

### 2.9 Perché funziona

1. **Naturale per binario**: Solo XNOR, XOR, POPCOUNT, AND, OR, NOT. Zero moltiplicazioni FP.
2. **Parallela**: Ogni posizione indipendente nel layer. Solo dipendenza locale.
3. **Receptive field**: Con N=6, campo = 13 token. Dilatable con skip.
4. **Interpretabile**: Stati binari ispezionabili.
5. **Memory-bound**: Collo di bottiglia = memoria, favorisce efficienza.
6. **Training stabile**: STE + hidden weights FP32.
