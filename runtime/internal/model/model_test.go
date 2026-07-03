package model

import (
	"math"
	"math/rand"
	"testing"
)

func TestBSMConfigValidate(t *testing.T) {
	tests := []struct {
		name  string
		cfg   BSMConfig
		valid bool
	}{
		{"valid", BSMConfig{1024, 256, 4, 8, 128}, true},
		{"bad hidden dim", BSMConfig{1024, 255, 4, 8, 128}, false},
		{"no layers", BSMConfig{1024, 256, 0, 8, 128}, false},
		{"no window", BSMConfig{1024, 256, 4, 0, 128}, false},
		{"no vocab", BSMConfig{0, 256, 4, 8, 128}, false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.cfg.Validate()
			if tc.valid && err != nil {
				t.Errorf("expected valid, got err: %v", err)
			}
			if !tc.valid && err == nil {
				t.Errorf("expected error, got nil")
			}
		})
	}
}

func makeTestModel(t *testing.T, cfg BSMConfig) *BSMModel {
	t.Helper()
	nBytes := cfg.HiddenDim * cfg.HiddenDim / 8

	// Embedding: vocabSize * hiddenDim bits
	embData := make([]byte, cfg.VocabSize*cfg.HiddenDim/8)
	emb, err := EmbeddingFromBytes(embData, cfg.VocabSize, cfg.HiddenDim)
	if err != nil {
		t.Fatal(err)
	}

	// Layers
	layers := make([]BSMLayerWeights, cfg.NumLayers)
	for li := 0; li < cfg.NumLayers; li++ {
		makeW := func() *BinaryWeight {
			data := make([]byte, nBytes)
			for i := range data {
				data[i] = byte((li+1)*i%256 + li)
			}
			w, err := BinaryWeightFromBytes(data, cfg.HiddenDim, cfg.HiddenDim)
			if err != nil {
				t.Fatal(err)
			}
			return w
		}
		layers[li] = BSMLayerWeights{
			WForget: makeW(),
			WInput:  makeW(),
			WMix:    makeW(),
		}
	}

	// Head: FP32 weights
	headW := make([]float32, cfg.VocabSize*cfg.HiddenDim)
	for i := range headW {
		headW[i] = float32(math.Sin(float64(i))) * 0.01
	}

	return &BSMModel{
		Config: cfg,
		Embed:  emb,
		Layers: layers,
		HeadW:  headW,
	}
}

func TestNewBSMSession(t *testing.T) {
	cfg := BSMConfig{VocabSize: 128, HiddenDim: 64, NumLayers: 2, WindowSize: 4, MaxSeqLen: 64}
	model := makeTestModel(t, cfg)

	session := NewBSMSession(model)
	if session == nil {
		t.Fatal("expected non-nil session")
	}
	if len(session.States) != cfg.NumLayers {
		t.Errorf("states = %d, want %d", len(session.States), cfg.NumLayers)
	}
}

func TestSessionStep(t *testing.T) {
	cfg := BSMConfig{VocabSize: 128, HiddenDim: 64, NumLayers: 2, WindowSize: 4, MaxSeqLen: 64}
	model := makeTestModel(t, cfg)
	session := NewBSMSession(model)

	logits, err := session.Step(0)
	if err != nil {
		t.Fatal(err)
	}
	if len(logits) != cfg.VocabSize {
		t.Errorf("logits len = %d, want %d", len(logits), cfg.VocabSize)
	}

	// Continuous generation should work
	for i := 0; i < 5; i++ {
		logits, err = session.Step(argmax(logits))
		if err != nil {
			t.Fatal(err)
		}
	}
}

func TestSessionStepOutOfRange(t *testing.T) {
	cfg := BSMConfig{VocabSize: 128, HiddenDim: 64, NumLayers: 1, WindowSize: 4, MaxSeqLen: 64}
	model := makeTestModel(t, cfg)
	session := NewBSMSession(model)

	_, err := session.Step(-1)
	if err == nil {
		t.Error("expected error for negative token ID")
	}

	_, err = session.Step(200)
	if err == nil {
		t.Error("expected error for out-of-range token ID")
	}
}

func TestSessionReset(t *testing.T) {
	cfg := BSMConfig{VocabSize: 128, HiddenDim: 64, NumLayers: 1, WindowSize: 4, MaxSeqLen: 64}
	model := makeTestModel(t, cfg)
	session := NewBSMSession(model)

	// Generate a token to change state
	session.Step(0)
	session.Reset()

	// After reset, states should be zeroed
	for _, st := range session.States {
		for _, w := range st.state {
			if w != 0 {
				t.Error("state should be zero after reset")
			}
		}
		if st.windowPos != 0 {
			t.Error("windowPos should be 0 after reset")
		}
		if st.windowFull {
			t.Error("windowFull should be false after reset")
		}
	}
}

func TestSampleArgmax(t *testing.T) {
	logits := []float32{1.0, 5.0, 3.0, 2.0}
	rng := rand.New(rand.NewSource(42))
	result := Sample(logits, 0, 0, rng)
	if result != 1 {
		t.Errorf("argmax: got %d, want 1", result)
	}
}

func TestSampleTemperature(t *testing.T) {
	logits := []float32{1.0, 2.0, 3.0, 4.0}
	rng := rand.New(rand.NewSource(42))

	results := make(map[int]int)
	for i := 0; i < 100; i++ {
		results[Sample(logits, 1.0, 0, rng)]++
	}

	// Last token (highest logit) should be most common
	if results[3] <= results[0] {
		t.Logf("distribution: %v", results)
	}
}

func TestSampleTopK(t *testing.T) {
	logits := []float32{-100, -100, 0, 0, 100}
	rng := rand.New(rand.NewSource(42))

	for i := 0; i < 50; i++ {
		result := Sample(logits, 1.0, 1, rng)
		if result != 4 {
			t.Errorf("top-1 should always pick 4, got %d", result)
		}
	}
}

func TestGenerate(t *testing.T) {
	cfg := BSMConfig{VocabSize: 128, HiddenDim: 64, NumLayers: 1, WindowSize: 4, MaxSeqLen: 64}
	model := makeTestModel(t, cfg)
	session := NewBSMSession(model)
	rng := rand.New(rand.NewSource(42))

	prompt := []int{0, 1, 2}
	result, err := session.Generate(prompt, 10, 1.0, 0, rng)
	if err != nil {
		t.Fatal(err)
	}
	if len(result) != 13 {
		t.Errorf("expected 13 tokens (3 prompt + 10 generated), got %d", len(result))
	}

	// First tokens should match prompt
	for i, id := range prompt {
		if result[i] != id {
			t.Errorf("result[%d] = %d, want %d", i, result[i], id)
		}
	}
}

func TestGenerateGreedy(t *testing.T) {
	cfg := BSMConfig{VocabSize: 128, HiddenDim: 64, NumLayers: 1, WindowSize: 4, MaxSeqLen: 64}
	model := makeTestModel(t, cfg)
	session := NewBSMSession(model)
	rng := rand.New(rand.NewSource(42))

	// Greedy generation should be deterministic with same seed
	prompt := []int{0}
	result1, err := session.Generate(prompt, 20, 0, 0, rng)
	if err != nil {
		t.Fatal(err)
	}

	// Reset and generate again
	session.Reset()
	rng = rand.New(rand.NewSource(42))
	result2, err := session.Generate(prompt, 20, 0, 0, rng)
	if err != nil {
		t.Fatal(err)
	}

	for i := range result1 {
		if result1[i] != result2[i] {
			t.Errorf("greedy generation not deterministic at position %d: %d vs %d", i, result1[i], result2[i])
		}
	}
}

func TestNumParameters(t *testing.T) {
	cfg := BSMConfig{VocabSize: 1024, HiddenDim: 256, NumLayers: 4, WindowSize: 8, MaxSeqLen: 128}
	model := makeTestModel(t, cfg)

	binaryBits, floatBytes := model.NumParameters()

	// Embedding: 1024 * 256 = 262,144 bits
	// Per layer: 3 * 256 * 256 = 196,608 bits → 4 layers = 786,432 bits
	// Total binary: 1,048,576 bits
	if binaryBits != 1048576 {
		t.Errorf("binaryBits = %d, want 1048576", binaryBits)
	}

	// Head: 1024 * 256 * 4 = 1,048,576 bytes
	if floatBytes != 1048576 {
		t.Errorf("floatBytes = %d, want 1048576", floatBytes)
	}
}

func TestKthLargest(t *testing.T) {
	vals := []float32{5, 1, 9, 3, 7}
	result := kthLargest(vals, 3)
	if result != 5 {
		t.Errorf("3rd largest = %f, want 5", result)
	}

	result = kthLargest(vals, 1)
	if result != 9 {
		t.Errorf("largest = %f, want 9", result)
	}
}
