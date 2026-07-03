# BLM — CLI Design

## 1. Comandi

```
blm train --data <path> --config <path> --output <path> [flags]
blm infer --model <path> --prompt <string> [flags]
blm benchmark --model <path> [flags]
blm inspect --model <path> [flags]
blm export --model <path> --format <string> [flags]
blm stats --model <path>
blm profile --model <path> --prompt <string>
blm explain --model <path> --prompt <string> --token <string>
blm quantize --input <path> --output <path> [flags]
blm validate --model <path> --data <path>
blm version
```

## 2. Dettaglio

### blm train

```
Avvia il training (python).
Il Go invoca lo script Python come subprocess.

Usage:
  blm train --data data.txt --config config.json --output model.blmf

Flags:
  --data PATH        Training data (text file, one doc per line)
  --config PATH      JSON config file (model architecture)
  --output PATH      Output BLMF file
  --epochs INT       Training epochs (default: 10)
  --lr FLOAT         Learning rate (default: 1e-4)
  --batch-size INT   Batch size (default: 64)
  --device STRING    Device (cpu/cuda) (default: "cuda")
  --seed INT         RNG seed (default: 42)
  --resume PATH      Resume from checkpoint
  --eval-every INT   Eval perplexity every N steps (default: 100)
```

### blm infer

```
Inferenza interattiva o one-shot.

Usage:
  blm infer --model model.blmf --prompt "Hello" --steps 100

Flags:
  --model PATH       BLMF model file
  --prompt STRING    Input text
  --steps INT        Max generation steps (default: 256)
  --temperature FLOAT Sampling temp (default: 0.8, 0=greedy)
  --top-k INT        Top-k sampling (default: 0=disabled)
  --show-states      Dump binary states per token (debug)
  --output PATH      Write output to file (default: stdout)
```

### blm benchmark

```
Benchmark throughput e latenza.

Usage:
  blm benchmark --model model.blmf

Flags:
  --model PATH       BLMF model file
  --runs INT         Number of runs (default: 100)
  --warmup INT       Warmup runs before measuring (default: 10)
  --context INT      Context length for benchmark (default: 256)
  --batch INT        Simulated batch size (default: 1)
  --output FORMAT    Output format: text/json/csv (default: text)

Output:
  Average tokens/sec
  Average ms/token
  Memory usage (model + runtime)
  CPU cache misses (if perf events available)
```

### blm inspect

```
Ispezione interna del modello.

Usage:
  blm inspect --model model.blmf

Flags:
  --model PATH       BLMF model file
  --layer INT        Show specific layer (default: all)
  --weights          Show weight statistics
  --states           Show layer dimensions
  --tokens           Show vocabulary sample
  --output FORMAT    text/json (default: text)

Output example:
  Model: BCM-2048
  Vocab: 4096 tokens
  Dim:   2048 bits
  Layers: 6
  Params: ~5.2M bits (~0.65M params)
  Size:   4.2 MB
```

### blm export

```
Esporta il modello in altri formati.

Usage:
  blm export --model model.blmf --format onnx

Flags:
  --model PATH       BLMF model file
  --format STRING    Export format: onnx, c-header, raw
  --output PATH      Output path (default: model.<format>)
```

### blm stats

```
Statistiche del modello.

Usage:
  blm stats --model model.blmf

Output:
  +----------------------+-----------+
  | Metric               | Value     |
  +----------------------+-----------+
  | Architecture         | BCM-2048  |
  | Vocabulary           | 4096      |
  | Dimension            | 2048 bit  |
  | Layers               | 6         |
  | Hidden dim           | 1024      |
  | Total bits           | 5,242,880 |
  | Total bytes          | 655,360   |
  | Embedding            | 1,048,576 byte |
  | Model file size      | 4.2 MB    |
  | Embeddings           | 4096 x 256 byte |
  | Layer 0: BinMLP      | 1,572,864 bit |
  | Layer 1: BinMLP      | 1,572,864 bit |
  | ...                  | ...       |
  | Layer 5: BinMLP      | 1,572,864 bit |
  | Bit density (W1)     | 51.2%     |
  | Bit density (W2)     | 49.8%     |
  | Bit density (Wg)     | 50.1%     |
  +----------------------+-----------+
```

### blm explain

```
Spiega perche' il modello ha generato un token.

Usage:
  blm explain --model model.blmf --prompt "The cat" --token "sat"

Output:
  Token "sat" (id: 1294) generated:
  - Layer 0: 1072 bits matched, 976 bits changed
  - Layer 1: 892 bits matched, 1156 bits changed
  - Layer 2: site: 1340 bits matched
  - ...
  - Top-5 candidates at output:
    1. "sat"  (sim: 0.87)
    2. "sits" (sim: 0.72)
    3. "was"  (sim: 0.68)
    4. "lay"  (sim: 0.65)
    5. "the"  (sim: 0.61)
```

### blm quantize

```
Non serve per il modello binario (gia' binario).
Serve per convertire un modello FP32 -> BLM (futuro).

Usage:
  blm quantize --input model.pt --output model.blmf --mode binary
```

### blm validate

```
Valuta perplexity su un test set.

Usage:
  blm validate --model model.blmf --data test.txt

Output:
  Validation perplexity: 45.2
  Tokens processed: 10240
  Time: 1.23s (8325 tok/s)
```

### blm version

```
Stampa versione.

Usage:
  blm version

Output:
  BLM v0.1.0
  Binary Language Model Runtime
  Go runtime, BCM-2048 architecture
```

## 3. Global Flags

```
  --verbose           Enable debug output
  --quiet             Suppress non-essential output
  --log-file PATH     Write logs to file
  --no-color          Disable ANSI colors
```

## 4. Exit Codes

```
0  - Success
1  - General error
2  - Invalid input/model
3  - Configuration error
4  - Runtime error (OOM, etc.)
```

## 5. Integrazione con Python

```
blm train  → Go invoca: python python/scripts/train.py [args]
blm export → Go invoca: python python/scripts/export.py [args]

Tutti gli altri comandi sono eseguiti direttamente dal Go runtime.
```
