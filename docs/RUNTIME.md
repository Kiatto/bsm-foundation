# BLM — Go Runtime Architecture

## 1. Package Structure

```
go/
  cmd/blm/main.go            # entry point
  internal/
    model/
      model.go               # Model struct, load, forward
      layers.go              # Layer interface, BinMLP, Conv1D
      bits.go                # BitVector, POPCOUNT, XOR helpers
      arch.go                # Architecture registry
    tokenizer/
      tokenizer.go           # BPE tokenizer
      bpe.go                 # BPE merge logic
    format/
      blmf.go                # BLMF reader, header, validation
      blmf_test.go           # round-trip test
    cli/
      root.go                # root command
      train.go               # delegates to python
      infer.go               # local inference
      benchmark.go           # perf benchmarks
      inspect.go             # model introspection
      export.go              # export formats
      stats.go               # model statistics
      explain.go             # token-level explanations
    bench/
      bench.go               # timing, throughput, memory
      profiler.go            # per-layer profiling
```

## 2. Model Struct

```go
package model

type Model struct {
    Config     Config
    EmbedTable BitMatrix
    Layers     []Layer
    Tokenizer  *tokenizer.Tokenizer
}

type Config struct {
    Dimension   int
    NumLayers   int
    ContextLen  int
    DHidden     int
    VocabSize   int
}

// BitMatrix: D bits per riga, packed in uint64
type BitMatrix struct {
    Rows int
    Cols int // in uint64 (Dimension/64)
    Data []uint64
}

func (bm *BitMatrix) Get(row int) BitVector {
    start := row * bm.Cols
    return BitVector(bm.Data[start : start+bm.Cols])
}

// BitVector: slice view over a BitMatrix row
type BitVector []uint64
```

## 3. Layer Interface

```go
type Layer interface {
    Type() LayerType
    Forward(input []BitVector, tokenIDs []int) ([]BitVector, error)
    Profile() ProfileInfo
}

type LayerType byte

const (
    LayerBinMLP LayerType = 0x01
    LayerConv1D LayerType = 0x02
)

type BinMLP struct {
    W1 BitMatrix  // [D_hidden, D/64]
    t1 []uint64   // [D_hidden/64]
    W2 BitMatrix  // [D, D_hidden/64]
    t2 []uint64   // [D/64]
    Wg BitMatrix  // [D, D/64]
    tg []uint64   // [D/64]
    D, DHidden int
}

type Conv1D struct {
    W  []BitVector  // [K] each D bits
    t  []uint64     // [D/64]
    K  int
    D  int
}
```

## 4. Forward Inference

```go
func (m *Model) Forward(ids []int) ([]float32, error) {
    L := len(ids)
    if L > m.Config.ContextLen {
        ids = ids[L-m.Config.ContextLen:]
        L = m.Config.ContextLen
    }

    // Embed lookup
    states := make([]BitVector, L)
    for i, id := range ids {
        states[i] = m.EmbedTable.Get(id)
    }

    // Stack of layers
    for _, layer := range m.Layers {
        var err error
        states, err = layer.Forward(states, ids)
        if err != nil {
            return nil, err
        }
    }

    // Output head: Hamming distance to all vocab
    final := states[L-1]
    logits := make([]float32, m.Config.VocabSize)

    // Parallelize this with goroutines
    for v := 0; v < m.Config.VocabSize; v++ {
        dist := 0
        vocab := m.EmbedTable.Get(v)
        for c := 0; c < m.Config.Dimension/64; c++ {
            dist += bits.OnesCount64(final[c] ^ vocab[c])
        }
        logits[v] = float32(m.Config.Dimension - dist)
    }

    // Softmax (only FP operation)
    return softmax(logits), nil
}
```

## 5. Text Generation

```go
func (m *Model) Generate(prompt string, maxTokens int, temp float32) (string, error) {
    ids := m.Tokenizer.Encode(prompt)

    for len(ids) < maxTokens {
        logits, err := m.Forward(ids)
        if err != nil {
            return "", err
        }

        // Sample next token
        var next int
        if temp == 0 {
            next = argmax(logits)
        } else {
            next = sample(logits, temp)
        }

        ids = append(ids, next)

        if next == m.Tokenizer.EOS() {
            break
        }
    }

    return m.Tokenizer.Decode(ids), nil
}
```

## 6. Concurrency

Ogni forward e' puramente funzionale (nessuno stato mutabile). Si puo' parallelizzare:

1. **Per-batch**: Più richieste di inferenza in parallelo (goroutines separate)
2. **Output head**: Calcolo similarita' diviso tra goroutine (V/numCPU per worker)
3. **Layer pipeline**: Una goroutine per layer (ma overhead > beneficio per N=6)

```
Esempio parallelizzazione output head:

func parallelHamming(final BitVector, embed *BitMatrix, vocabSize int) []float32 {
    nCPU := runtime.NumCPU()
    chunk := (vocabSize + nCPU - 1) / nCPU
    results := make([]float32, vocabSize)
    var wg sync.WaitGroup

    for cpu := 0; cpu < nCPU; cpu++ {
        start := cpu * chunk
        end := min(start+chunk, vocabSize)
        wg.Add(1)
        go func(s, e int) {
            defer wg.Done()
            for v := s; v < e; v++ {
                dist := 0
                vocab := embed.Get(v)
                for c := range final {
                    dist += bits.OnesCount64(final[c] ^ vocab[c])
                }
                results[v] = float32(len(final)*64 - dist)
            }
        }(start, end)
    }
    wg.Wait()
    return results
}
```

## 7. Tokenizer (BPE in Go)

```go
type Tokenizer struct {
    vocab    map[string]int
    idToStr  map[int]string
    merges   []MergeRule
}

type MergeRule struct {
    Left, Right string
}

func (t *Tokenizer) Encode(text string) []int {
    // 1. Normalize
    // 2. Split to bytes/unicode chars
    // 3. Apply BPE merges greedily
    // 4. Map to IDs
}

func (t *Tokenizer) Decode(ids []int) string {
    // 1. Map IDs to tokens
    // 2. Concatenate
    // 3. Handle special tokens
}
```

## 8. Model Loading

```go
package format

func LoadBLMF(path string) (*model.Model, error) {
    f, err := os.Open(path)
    if err != nil { return nil, err }
    defer f.Close()

    // mmap for zero-copy
    data, err := mmap.Mmap(f, 0, stat.Size())

    // Parse header
    h := (*Header)(unsafe.Pointer(&data[0]))
    if h.Magic != 0x464D4C42 { return nil, ErrInvalidMagic }
    if h.Version != 1 { return nil, ErrUnsupportedVersion }

    // Parse config (JSON)
    configJSON := data[64:64+h.ConfigSize]

    // Parse embeddings
    embedOff := 64 + h.ConfigSize
    embedData := data[embedOff : embedOff + uint32(h.VocabSize)*h.Dimension/8]

    // Parse layers
    layerOff := embedOff + uint32(h.VocabSize)*h.Dimension/8
    layers := []model.Layer{}
    for i := uint32(0); i < h.NumLayers; i++ {
        lh := (*LayerHeader)(unsafe.Pointer(&data[layerOff]))
        // read layer data
        // construct appropriate Layer type
        layerOff += 32 + lh.WeightBytes + lh.ThresholdBytes
    }

    // Parse token table
    // ...

    return &model.Model{...}, nil
}
```
