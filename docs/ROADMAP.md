# BLM — Roadmap e Repository

## 1. Repository Structure

```
blm/
|-- README.md                    # Panoramica e quick start
|-- Makefile                     # Comandi principali
|-- go.mod / go.sum             # Go module
|-- requirements.txt             # Python dependencies
|
|-- docs/
|   |-- ARCHITETTURA.md         # Architettura completa
|   |-- TRAINING.md             # Training methodology
|   |-- OPERAZIONI.md           # Binary operations
|   |-- BLMF.md                 # File format spec
|   |-- RUNTIME.md              # Go runtime arch
|   |-- CLI.md                  # CLI reference
|   |-- ROADMAP.md              # Questo file
|
|-- python/
|   |-- blm/
|   |   |-- __init__.py
|   |   |-- config.py           # Config dataclass
|   |   |-- model.py            # BCM model definition
|   |   |-- binary_ops.py       # Custom binary ops
|   |   |-- trainer.py          # Training loop
|   |   |-- data.py             # Data loading
|   |   |-- tokenizer.py        # BPE tokenizer
|   |   |-- export.py           # Export to BLMF
|   |   |-- validate.py         # Perplexity eval
|   |-- scripts/
|   |   |-- train.py            # CLI entry: training
|   |   |-- export.py           # CLI entry: export
|   |   |-- download_data.py    # Dataset download
|   |-- tests/
|       |-- test_model.py
|       |-- test_tokenizer.py
|       |-- test_training.py
|
|-- go/
|   |-- cmd/
|   |   |-- blm/
|   |       |-- main.go         # CLI entry point
|   |-- internal/
|   |   |-- model/
|   |   |   |-- model.go        # Model struct
|   |   |   |-- layers.go       # Layer implementations
|   |   |   |-- bits.go         # BitVector operations
|   |   |   |-- arch.go         # Architecture registry
|   |   |   |-- model_test.go
|   |   |-- tokenizer/
|   |   |   |-- tokenizer.go    # BPE tokenizer
|   |   |   |-- bpe.go          # Merge rules
|   |   |   |-- tokenizer_test.go
|   |   |-- format/
|   |   |   |-- blmf.go         # BLMF reader
|   |   |   |-- blmf_test.go
|   |   |-- cli/
|   |   |   |-- root.go         # Cobra root
|   |   |   |-- infer.go        # Infer command
|   |   |   |-- benchmark.go    # Benchmark command
|   |   |   |-- inspect.go      # Inspect command
|   |   |   |-- stats.go        # Stats command
|   |   |   |-- export.go       # Export command
|   |   |   |-- profile.go      # Profile command
|   |   |   |-- explain.go      # Explain command
|   |   |   |-- validate.go     # Validate command
|   |   |   |-- train.go        # Train (delegate to Python)
|   |   |   |-- version.go      # Version command
|   |   |-- bench/
|   |       |-- bench.go        # Benchmarking
|   |       |-- profiler.go     # Per-layer profiler
|   |-- Makefile                # Go-specific commands
|
|-- benchmarks/
|   |-- results/                # Benchmark results dir
|   |-- datasets/               # Small test datasets
|
|-- configs/
|   |-- bcm-2048.json           # Default config
|   |-- bcm-4096.json           # Larger variant
```

## 2. Milestone Roadmap

### Milestone 0 — Repository (Week 1)

```
Obiettivo: Struttura del progetto funzionante.

Deliverable:
  - Struttura directory completa
  - Makefile con target: setup, test, lint
  - go.mod con dipendenze minime
  - requirements.txt
  - README.md con quick start
  - Config JSON di default
  - CI base (github actions)

Criteri di successo:
  - go build funziona
  - python -m pytest passa (nessun test)
  - make setup installa dipendenze
```

### Milestone 1 — Tokenizer (Week 2)

```
Obiettivo: Tokenizer BPE funzionante in Python e Go.

Deliverable:
  Python:
    - BPE trainer (apprende merge rules)
    - BPE tokenizer (encode/decode)
    - Test: roundtrip token/detokenize
    - Saver/caricamento da JSON

  Go:
    - BPE tokenizer identico
    - Caricamento merge rules da JSON
    - Test: roundtrip identico a Python

  BLMF:
    - Sezione token table nel formato
    - Salvataggio/caricamento tokenizer

Criteri di successo:
  - Tokenize("Hello world") == [12, 456, 89] (identico in Python e Go)
  - Decode([12, 456, 89]) == "Hello world"
  - ~10000 token/sec in Go
```

### Milestone 2 — Binary Embedding (Week 3)

```
Obiettivo: Embedding table binaria.

Deliverable:
  Python:
    - Binary embedding table (FP32 hidden + sign())
    - STE per embedding
    - Forward/backward con embedding binario
    - Export embedding a BLMF

  Go:
    - Caricamento embedding da BLMF
    - Embed lookup (BitMatrix.Get)
    - Benchmark lookup speed

Criteri di successo:
  - Embedding .forward() produce vettori {-1, +1}
  - Gradiente fluisce attraverso STE
  - Go: lookup < 100ns per token
```

### Milestone 3 — Binary Core (Week 4-5)

```
Obiettivo: CellulalBlock completo e funzionante.

Deliverable:
  Python:
    - BinMLP (D -> D_hidden -> D) con STE
    - Binary Gate (D -> D) con STE
    - Binary Channel Mixer
    - Binary Temporal Mixer (Conv1D)
    - CellularBlock completo
    - Stack di N CellularBlock
    - Forward pass end-to-end

  Go:
    - Implementazione BinMLP
    - Implementazione Binary Gate
    - Implementazione Channel/Temporal Mixer
    - Forward pass end-to-end
    - Roundtrip test: stessi input -> stessi output

  BLMF:
    - Sezione layers nel formato
    - Salvataggio/caricamento di tutti i layer

Criteri di successo:
  - Python forward pass funziona con dati reali
  - Go forward produce output identico (a meno di rounding)
  - ~5000 token/sec in Go
```

### Milestone 4 — Training (Week 6-8)

```
Obiettivo: Modello addestrato su TinyShakespeare.

Deliverable:
  - Training loop completo
  - Data loader per testo
  - Adam optimizer su hidden weights
  - STE backward pass
  - Gradient clipping
  - Learning rate schedule
  - Validation loop
  - Perplexity tracking
  - Model checkpointing
  - Export a BLMF

Criteri di successo:
  - Modello converge (perplexity decrescente)
  - Genera testo riconoscibile dopo training
  - Training su TinyShakespeare: < 1 ora su GPU
  - Export produce file BLMF valido
```

### Milestone 5 — Inference (Week 9-10)

```
Obiettivo: Generazione testo via Go CLI.

Deliverable:
  - blm infer funzionante
  - Caricamento modello da BLMF
  - Greedy decoding
  - Sampling con temperatura
  - Top-k sampling
  - Stato interattivo (prompt -> genera fino a EOS)
  - Opzione --show-states per debug

Criteri di successo:
  - blm infer --model model.blmf --prompt "Hello" produce output
  - Generazione ~5000 token/sec
  - Output ragionevole (meglio di random)
```

### Milestone 6 — Benchmark (Week 11)

```
Obiettivo: Profilazione completa del runtime.

Deliverable:
  - blm benchmark: throughput (tok/s)
  - blm benchmark: latenza (ms/token)
  - blm benchmark: memoria (RSS, heap)
  - blm profile: per-layer breakdown
  - blm stats: modello stats complete

Criteri di successo:
  - Benchmark riproducibili
  - Documentazione dei risultati
  - Identificazione dei colli di bottiglia
```

### Milestone 7 — Scaling (Week 12+)

```
Obiettivo: Modelli piu' grandi e migliori.

Deliverable:
  - Config per BCM-4096 (D=4096, V=8192)
  - Training su dataset piu' grande (WikiText-2)
  - Confronto perplexity vs baseline
  - SIMD ottimizzazioni (AVX2, NEON)
  - Parallelizzazione output head
  - bfloat16 per hidden weights (risparmio memoria)

Criteri di successo:
  - BCM-4096 batte BCM-2048 in perplexity
  - Documentazione dei tradeoff dimensione/perplexity
```

## 3. Dipendenze Software

### Python (training)
```
torch >= 2.0
numpy
tqdm
datasets (opzionale, per dataset)
sentencepiece (per tokenizer alternativo)
```

### Go (runtime)
```
github.com/spf13/cobra        # CLI framework
golang.org/x/exp/mmap          # Memory-mapped files
github.com/klauspost/cpuid/v2  # CPU feature detection (AVX2, POPCNT)
```

## 4. Makefile Comandi

```
make setup          # Installa dipendenze
make test-python    # pytest
make test-go        # go test ./...
make train          # blm train --config configs/bcm-2048.json --data data.txt
make infer          # blm infer --model model.blmf --prompt "Test"
make benchmark      # blm benchmark --model model.blmf
make build          # go build -o bin/blm ./go/cmd/blm
make clean          # rimuovi file generati
make lint           # golangci-lint + ruff
```
