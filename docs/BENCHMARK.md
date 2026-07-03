# BLM — Benchmark Reali (i7-8550U, 4C/8T, 1.80GHz)

## 1. Operazioni Bitwise (micro-benchmark)

| Operazione | Dimensione | ns/op | allocs/op |
|------------|-----------|-------|-----------|
| PopcountWords | 2048 bit (4 words) | 22.2 ns | 0 |
| BinaryDot | 2048 bit | 16.9 ns | 0 |

POPCNT hardware utilizzato via `math/bits.OnesCount64`. Zero allocazioni.

## 2. Throughput Inferenza

### baseline (forward sequenziale)

| Modello | D | Layers | Vocab | Parametri | tok/s | ms/tok | Memoria |
|---------|---|--------|-------|-----------|-------|--------|---------|
| bsm-mini | 64 | 2 | 64 | 28K bit | 56,305 | 0.018 | 7.4 MB |
| bsm-tiny | 128 | 4 | 256 | 228K bit | 8,123 | 0.123 | 8.3 MB |
| bsm-small | 256 | 6 | 1024 | 1.4M bit | 1,089 | 0.919 | 8.5 MB |
| bsm-medium | 256 | 6 | 2048 | 1.4M bit | 576 | 1.735 | 10 MB |

### con output head parallelizzato (4 goroutine)

| Modello | D | Layers | Vocab | tok/s | ms/tok | Speedup |
|---------|---|--------|-------|-------|--------|---------|
| bsm-mini | 64 | 2 | 64 | 37,229 | 0.027 | 0.7x |
| bsm-tiny | 128 | 4 | 256 | 10,446 | 0.096 | **1.3x** |
| bsm-small | 256 | 6 | 1024 | 2,205 | 0.454 | **2.0x** |
| bsm-medium | 256 | 6 | 2048 | 1,399 | 0.715 | **2.4x** |

Speedup cresce con vocab_size: il collo di bottiglia è l'output head (matrice FP32
vocab×hidden_dim). La parallelizzazione con goroutine dà fino a 2.4x su 4 core.

## 3. Breakdown per-operazione (bsm-medium)

Basato su profiling strumentale:

| Operazione | Tempo | % |
|------------|-------|---|
| Embedding lookup | ~0.3 us | <1% |
| Layer forward (×6) | ~500 us | 70% |
| Output head (FP32, 4 goroutine) | ~200 us | 28% |
| Softmax + sampling | ~5 us | 1% |
| Totale/token | ~715 us | 100% |

## 4. Dimensione Modelli su Disco

| Modello | Dimensione file |
|---------|----------------|
| bsm-mini | 21 KB |
| bsm-tiny | 158 KB |
| bsm-small | 1.2 MB |
| bsm-medium | 2.3 MB |
| bsm-4096 (D=512, L=6) | ~9 MB |

Il formato BLMF è efficiente: zero overhead oltre ai dati grezzi + header 512B.
Head weight FP32 domina (>80% del file per modelli con vocab grande).

## 5. Rotte di Ottimizzazione Future

| Ottimizzazione | Speedup stimato | Sforzo |
|----------------|----------------|--------|
| AVX2 per BinaryDot (8× word parallele) | 2-3x | Alto (assembly Plan9) |
| Cache blocking per layer forward | 1.2x | Medio |
| mmap model loading (zero-copy) | 1.1x (cold start) | Basso |
| Hierarchical softmax per output head | 10x su vocab grandi | Alto (richiede training) |
| SIMD output head (AVX2 FMA) | 3-4x | Alto (assembly) |
| Pool di oggetti (sync.Pool) per stati | 1.05x | Basso |

## 6. Confronto con Stime Originali

Le stime originali (ROADMAP.md) prevedevano ~5000 tok/s per BCM-2048 (D=256, V=2048).
I benchmark reali mostrano 1,399 tok/s (versione parallelizzata). Il gap è dovuto a:
- Output head FP32 più pesante del previsto (domina per V≥2048)
- Layer forward in Go puro senza SIMD (le stime assumevano AVX2)

Con AVX2 per BinaryDot + output head FMA, 5,000 tok/s è raggiungibile.
