package model

import (
	"fmt"
	"math"
	"math/rand"
	"sync"

	"github.com/blm/runtime/internal/bits"
)

// BSMConfig is the runtime configuration for a BSM model.
type BSMConfig struct {
	VocabSize  int
	HiddenDim  int
	NumLayers  int
	WindowSize int
	MaxSeqLen  int
}

// Validate checks that config parameters are valid.
func (c *BSMConfig) Validate() error {
	if c.HiddenDim%64 != 0 {
		return fmt.Errorf("hiddenDim %d must be multiple of 64", c.HiddenDim)
	}
	if c.NumLayers < 1 {
		return fmt.Errorf("numLayers must be >= 1")
	}
	if c.WindowSize < 1 {
		return fmt.Errorf("windowSize must be >= 1")
	}
	if c.VocabSize < 1 {
		return fmt.Errorf("vocabSize must be >= 1")
	}
	return nil
}

// BSMModel is the complete BSM language model loaded into memory.
type BSMModel struct {
	Config   BSMConfig
	Embed    *BinaryEmbedding      // [vocabSize, hiddenDim]
	Layers   []BSMLayerWeights     // [numLayers]
	HeadW    []float32            // [vocabSize * hiddenDim] FP32 weight matrix
	TreeHead *BinaryTreeHead      // optional binary tree head (replaces HeadW)
}

// NumParameters returns the number of binary and float parameters.
func (m *BSMModel) NumParameters() (binaryBits, floatBytes int) {
	// Embedding: vocabSize * hiddenDim bits
	binaryBits = m.Config.VocabSize * m.Config.HiddenDim

	// Per layer: 3 BinaryLinear each: hiddenDim * hiddenDim bits
	binaryBits += m.Config.NumLayers * 3 * m.Config.HiddenDim * m.Config.HiddenDim

	// Head: vocabSize * hiddenDim floats (4 bytes each)
	floatBytes = m.Config.VocabSize * m.Config.HiddenDim * 4

	return
}

// BSMSession holds the runtime state for inference.
type BSMSession struct {
	Model  *BSMModel
	States []*BSMLayerState // [numLayers]
	nw     int              // cached NWords(HiddenDim)
	bufA   []uint64         // [nw] pre-allocated layer I/O buffer (swapped with embedding buffer)
}

// NewBSMSession creates a new inference session for the given model.
func NewBSMSession(model *BSMModel) *BSMSession {
	nw := bits.NWords(model.Config.HiddenDim)
	states := make([]*BSMLayerState, model.Config.NumLayers)
	for i := 0; i < model.Config.NumLayers; i++ {
		states[i] = NewBSMLayerState(model.Config.HiddenDim, model.Config.WindowSize)
	}

	return &BSMSession{
		Model:  model,
		States: states,
		nw:     nw,
		bufA:   make([]uint64, nw),
	}
}

// Reset clears all session state for a new generation.
func (s *BSMSession) Reset() {
	for _, st := range s.States {
		st.ResetState()
	}
}

// Step executes one autoregressive step.
//
// tokenID: the current token ID
// Returns: logits (vocab scores) and optional error
func (s *BSMSession) Step(tokenID int) ([]float32, error) {
	m := s.Model

	if tokenID < 0 || tokenID >= m.Config.VocabSize {
		return nil, fmt.Errorf("token ID %d out of range [0, %d)", tokenID, m.Config.VocabSize)
	}

	// --- Embedding lookup ---
	// Get packed bits for this token
	x := m.Embed.Lookup(uint16(tokenID)) // [nw] uint64

	// --- Process through layers ---
	layerOut := x
	dst := s.bufA

	for i := 0; i < m.Config.NumLayers; i++ {
		StepBSMLayer(layerOut, &m.Layers[i], s.States[i], dst)
		layerOut, dst = dst, layerOut
	}

	// --- Output head (FP32 matmul) ---
	// headW: [vocabSize, hiddenDim] stored row-major as []float32
	// layerOut: [nw] packed bits in {-1,+1}
	// Unpack once, then compute dot product for each vocab entry.
	vocab := m.Config.VocabSize
	hd := m.Config.HiddenDim

	logits := make([]float32, vocab)

	// Unpack layerOut to float32 {-1,+1} slice (one allocation, O(hd))
	unpacked := make([]float32, hd)
	for i := 0; i < hd; i++ {
		if (layerOut[i/64]>>(i%64))&1 == 1 {
			unpacked[i] = 1.0
		} else {
			unpacked[i] = -1.0
		}
	}

	// Parallel dot product using goroutines (4 workers).
	// Each vocab entry is independent.
	headW := m.HeadW
	numWorkers := 4
	chunkSize := (vocab + numWorkers - 1) / numWorkers
	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > vocab {
			end = vocab
		}
		if start >= end {
			break
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for v := start; v < end; v++ {
				var dot float32
				base := v * hd
				j := 0
				for ; j+4 <= hd; j += 4 {
					dot += headW[base+j]*unpacked[j] + headW[base+j+1]*unpacked[j+1] +
						headW[base+j+2]*unpacked[j+2] + headW[base+j+3]*unpacked[j+3]
				}
				for ; j < hd; j++ {
					dot += headW[base+j] * unpacked[j]
				}
				logits[v] = dot
			}
		}(start, end)
	}
	wg.Wait()

	return logits, nil
}

// StepTree executes one autoregressive step using the binary tree head.
//
// Unlike Step(), which returns logits over the full vocabulary,
// StepTree returns pseudo-logits: -1.0 for most tokens and 1.0 for the
// predicted token (or top-K candidates). This avoids the O(V) softmax
// path while maintaining the same interface.
//
// When TreeHead is nil, falls back to regular Step().
func (s *BSMSession) StepTree(tokenID int) ([]float32, error) {
	m := s.Model
	if m.TreeHead == nil {
		return s.Step(tokenID)
	}

	if tokenID < 0 || tokenID >= m.Config.VocabSize {
		return nil, fmt.Errorf("token ID %d out of range [0, %d)", tokenID, m.Config.VocabSize)
	}

	x := m.Embed.Lookup(uint16(tokenID))

	layerOut := x
	dst := s.bufA
	for i := 0; i < m.Config.NumLayers; i++ {
		StepBSMLayer(layerOut, &m.Layers[i], s.States[i], dst)
		layerOut, dst = dst, layerOut
	}

	tokID := m.TreeHead.PredictToken(layerOut)

	logits := make([]float32, m.Config.VocabSize)
	for i := range logits {
		logits[i] = -1.0
	}
	if tokID < len(logits) {
		logits[tokID] = 1.0
	}
	return logits, nil
}

// StepTreeLogits executes one step using the tree head and returns
// binary dot scores for the top-K candidate tokens instead of full logits.
//
// Returns: (topTokenIDs, topScores) where scores are BinaryDot values
// in range [-HiddenDim, +HiddenDim]. Useful for sampling.
func (s *BSMSession) StepTreeLogits(tokenID int) ([]int, []int, error) {
	m := s.Model
	if m.TreeHead == nil {
		return nil, nil, fmt.Errorf("tree head not available")
	}

	if tokenID < 0 || tokenID >= m.Config.VocabSize {
		return nil, nil, fmt.Errorf("token ID %d out of range [0, %d)", tokenID, m.Config.VocabSize)
	}

	x := m.Embed.Lookup(uint16(tokenID))

	layerOut := x
	dst := s.bufA
	for i := 0; i < m.Config.NumLayers; i++ {
		StepBSMLayer(layerOut, &m.Layers[i], s.States[i], dst)
		layerOut, dst = dst, layerOut
	}

	topK := 10
	candidates := m.TreeHead.PredictTopK(layerOut, topK)

	scores := make([]int, len(candidates))
	for i, cid := range candidates {
		candidateState := m.Embed.Lookup(uint16(cid))
		scores[i] = bits.BinaryDot(layerOut, candidateState)
	}

	return candidates, scores, nil
}

// Sample picks a token ID from logits using temperature and top-k.
func Sample(logits []float32, temperature float32, topK int, rng *rand.Rand) int {
	if temperature == 0 {
		return argmax(logits)
	}

	// Apply temperature
	scaled := make([]float32, len(logits))
	for i, v := range logits {
		scaled[i] = v / temperature
	}

	// Apply top-k filtering
	if topK > 0 && topK < len(logits) {
		// Find the k-th largest value
		kth := kthLargest(scaled, topK)
		for i, v := range scaled {
			if v < kth {
				scaled[i] = -1e38
			}
		}
	}

	// Softmax
	maxVal := float32(-1e38)
	for _, v := range scaled {
		if v > maxVal {
			maxVal = v
		}
	}
	var sum float32
	probs := make([]float32, len(scaled))
	for i, v := range scaled {
		probs[i] = expf(v - maxVal)
		sum += probs[i]
	}
	if sum > 0 {
		for i := range probs {
			probs[i] /= sum
		}
	}

	// Sample from distribution
	r := rng.Float32()
	var cumulative float32
	for i, p := range probs {
		cumulative += p
		if r < cumulative {
			return i
		}
	}

	return len(probs) - 1
}

// Generate generates tokens autoregressively.
//
// promptIDs: initial token IDs
// maxNewTokens: number of tokens to generate
// temperature: sampling temperature (0 = greedy)
// topK: top-k filtering (0 = disabled)
// rng: random number generator
//
// Returns: full sequence of token IDs (prompt + generated)
func (s *BSMSession) Generate(promptIDs []int, maxNewTokens int, temperature float32, topK int, rng *rand.Rand) ([]int, error) {
	s.Reset()

	// Prefill: process the prompt
	generated := make([]int, len(promptIDs))
	copy(generated, promptIDs)

	for _, tokenID := range promptIDs {
		_, err := s.Step(tokenID)
		if err != nil {
			return nil, fmt.Errorf("prefill step: %w", err)
		}
	}

	// Autoregressive generation
	for i := 0; i < maxNewTokens; i++ {
		lastID := generated[len(generated)-1]
		logits, err := s.Step(lastID)
		if err != nil {
			return nil, fmt.Errorf("generate step %d: %w", i, err)
		}

		nextID := Sample(logits, temperature, topK, rng)
		generated = append(generated, nextID)
	}

	return generated, nil
}

// --- Internal helpers ---

func argmax(vals []float32) int {
	idx := 0
	for i := 1; i < len(vals); i++ {
		if vals[i] > vals[idx] {
			idx = i
		}
	}
	return idx
}

func kthLargest(vals []float32, k int) float32 {
	// Simple O(n*k) approach for small k
	// A more efficient approach could use a heap
	largest := make([]float32, k)
	for i := range largest {
		largest[i] = -1e38
	}

	for _, v := range vals {
		for j := 0; j < k; j++ {
			if v > largest[j] {
				// Shift and insert
				copy(largest[j+1:], largest[j:k-1])
				largest[j] = v
				break
			}
		}
	}

	return largest[k-1]
}

func expf(x float32) float32 {
	return float32(math.Exp(float64(x)))
}
