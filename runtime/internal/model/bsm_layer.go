package model

import (
	"github.com/blm/runtime/internal/bits"
)

// BSMLayerWeights holds all binary weight matrices for one BSM layer.
// Immutable after loading.
type BSMLayerWeights struct {
	WForget *BinaryWeight // [D, D] state → forget gate
	WInput  *BinaryWeight // [D, D] input → input projection
	WMix    *BinaryWeight // [D, D] feature mixing
}

// BSMLayerState holds the runtime state of one BSM layer during inference.
type BSMLayerState struct {
	state         []uint64   // [D/64] current hidden state (packed bits)
	windowRing    [][]uint64 // [windowSize][D/64] ring buffer of past activations
	windowPos     int        // current write position in ring buffer
	windowFull    bool       // true once ring buffer has been filled once
	forget        []uint64   // [D/64] pre-allocated temp for forget gate output
	inp           []uint64   // [D/64] pre-allocated temp for input gate output
	newState      []uint64   // [D/64] pre-allocated temp for majority3 result
	windowProduct []uint64   // [D/64] pre-allocated temp for temporal mixing XOR
}

// NewBSMLayerState allocates and initializes a layer state.
func NewBSMLayerState(hiddenDim, windowSize int) *BSMLayerState {
	nw := bits.NWords(hiddenDim)

	s := &BSMLayerState{
		state:         bits.AllocWords(hiddenDim),
		windowRing:    make([][]uint64, windowSize),
		windowPos:     0,
		windowFull:    false,
		forget:        make([]uint64, nw),
		inp:           make([]uint64, nw),
		newState:      make([]uint64, nw),
		windowProduct: make([]uint64, nw),
	}

	for i := 0; i < windowSize; i++ {
		s.windowRing[i] = bits.AllocWords(hiddenDim)
	}

	return s
}

// StepBSMLayer executes one autoregressive step for one layer.
//
// x: packed bits [nw] for current token activation
// w: layer weights
// s: layer state (mutated in place)
// dst: [nw] output packed bits
func StepBSMLayer(x []uint64, w *BSMLayerWeights, s *BSMLayerState, dst []uint64) {
	nw := len(x)

	// --- State update ---
	w.WForget.ForwardThreshold(s.state, s.forget)
	w.WInput.ForwardThreshold(x, s.inp)

	bits.Majority3(s.newState, s.state, s.forget, s.inp)
	copy(s.state, s.newState)

	// --- Mixer (single step) ---
	copy(s.windowRing[s.windowPos], x)

	nValid := 1
	if s.windowFull {
		nValid = len(s.windowRing)
	} else {
		nValid = s.windowPos + 1
	}

	copy(s.windowProduct, s.windowRing[0])
	for i := 1; i < nValid; i++ {
		bits.XORWords(s.windowProduct, s.windowProduct, s.windowRing[i])
	}

	s.windowPos++
	if s.windowPos >= len(s.windowRing) {
		s.windowPos = 0
		s.windowFull = true
	}

	w.WMix.ForwardThreshold(s.windowProduct, dst)

	for i := 0; i < nw; i++ {
		dst[i] = x[i] | dst[i]
	}
}

// CopyState copies layer state from src to dst.
func (s *BSMLayerState) CopyState(src *BSMLayerState) {
	copy(s.state, src.state)
	s.windowPos = src.windowPos
	s.windowFull = src.windowFull
	for i := range s.windowRing {
		copy(s.windowRing[i], src.windowRing[i])
	}
}

// ResetState reinitializes state to all -1 and clears window.
func (s *BSMLayerState) ResetState() {
	for i := range s.state {
		s.state[i] = 0
	}
	s.windowPos = 0
	s.windowFull = false
	for i := range s.windowRing {
		for j := range s.windowRing[i] {
			s.windowRing[i][j] = 0
		}
	}
}
