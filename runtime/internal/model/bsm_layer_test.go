package model

import (
	"testing"

	"github.com/blm/runtime/internal/bits"
)

func TestNewBSMLayerState(t *testing.T) {
	hiddenDim := 256
	windowSize := 8

	s := NewBSMLayerState(hiddenDim, windowSize)
	if s == nil {
		t.Fatal("expected non-nil state")
	}

	nw := bits.NWords(hiddenDim)
	if len(s.state) != nw {
		t.Errorf("state len = %d, want %d", len(s.state), nw)
	}
	if len(s.windowRing) != windowSize {
		t.Errorf("windowRing len = %d, want %d", len(s.windowRing), windowSize)
	}
	for i, buf := range s.windowRing {
		if len(buf) != nw {
			t.Errorf("windowRing[%d] len = %d, want %d", i, len(buf), nw)
		}
	}
	if s.windowPos != 0 {
		t.Errorf("windowPos = %d, want 0", s.windowPos)
	}
	if s.windowFull {
		t.Error("windowFull should be false")
	}

	// State should be all -1 (bit = 0)
	for _, w := range s.state {
		if w != 0 {
			t.Errorf("initial state word = 0x%X, want 0", w)
		}
	}
}

func TestBSMLayerStateReset(t *testing.T) {
	s := NewBSMLayerState(128, 4)

	// Fill with some non-zero values
	for i := range s.state {
		s.state[i] = ^uint64(0)
	}
	s.windowPos = 2
	s.windowFull = true

	s.ResetState()

	for _, w := range s.state {
		if w != 0 {
			t.Errorf("after reset state word = 0x%X, want 0", w)
		}
	}
	if s.windowPos != 0 {
		t.Errorf("windowPos = %d, want 0", s.windowPos)
	}
	if s.windowFull {
		t.Error("windowFull should be false")
	}
}

func TestBSMLayerStateCopy(t *testing.T) {
	src := NewBSMLayerState(128, 4)
	dst := NewBSMLayerState(128, 4)

	// Modify src
	for i := range src.state {
		src.state[i] = ^uint64(0)
	}
	src.windowPos = 3
	src.windowFull = true
	for i := range src.windowRing {
		for j := range src.windowRing[i] {
			src.windowRing[i][j] = ^uint64(0)
		}
	}

	dst.CopyState(src)

	for i := range dst.state {
		if dst.state[i] != ^uint64(0) {
			t.Errorf("dst.state[%d] = 0x%X, want MaxUint64", i, dst.state[i])
		}
	}
	if dst.windowPos != 3 {
		t.Errorf("windowPos = %d, want 3", dst.windowPos)
	}
	if !dst.windowFull {
		t.Error("windowFull should be true")
	}
	for i := range dst.windowRing {
		for j := range dst.windowRing[i] {
			if dst.windowRing[i][j] != ^uint64(0) {
				t.Errorf("dst.windowRing[%d][%d] = 0x%X, want MaxUint64", i, j, dst.windowRing[i][j])
			}
		}
	}
}

func makeTestLayerWeights(t *testing.T, hiddenDim int) *BSMLayerWeights {
	t.Helper()
	nBytes := hiddenDim * hiddenDim / 8 // per weight matrix

	makeWeight := func() *BinaryWeight {
		data := make([]byte, nBytes)
		// Fill with alternating pattern for deterministic test
		for i := range data {
			data[i] = byte(i % 256)
		}
		w, err := BinaryWeightFromBytes(data, hiddenDim, hiddenDim)
		if err != nil {
			t.Fatal(err)
		}
		return w
	}

	return &BSMLayerWeights{
		WForget: makeWeight(),
		WInput:  makeWeight(),
		WMix:    makeWeight(),
	}
}

func TestStepBSMLayerOutputDimension(t *testing.T) {
	hiddenDim := 64
	windowSize := 4
	nw := bits.NWords(hiddenDim)

	w := makeTestLayerWeights(t, hiddenDim)
	s := NewBSMLayerState(hiddenDim, windowSize)

	x := bits.AllocWords(hiddenDim)
	x[0] = ^uint64(0) // all +1

	dst := bits.AllocWords(hiddenDim)
	StepBSMLayer(x, w, s, dst)

	if len(dst) != nw {
		t.Errorf("dst len = %d, want %d", len(dst), nw)
	}
}

func TestStepBSMLayerStateChanges(t *testing.T) {
	hiddenDim := 64 + 64
	windowSize := 4

	w := makeTestLayerWeights(t, hiddenDim)
	s := NewBSMLayerState(hiddenDim, windowSize)

	// Use non-zero state: state = x = all +1
	nw := bits.NWords(hiddenDim)
	x := bits.AllocWords(hiddenDim)
	for i := range x {
		x[i] = ^uint64(0) // all +1
		s.state[i] = ^uint64(0) // initial state = all +1
	}

	dst := bits.AllocWords(hiddenDim)
	initialState := make([]uint64, nw)
	copy(initialState, s.state)

	StepBSMLayer(x, w, s, dst)

	// State should have changed (at least one word differs)
	changed := false
	for i := range s.state {
		if s.state[i] != initialState[i] {
			changed = true
			break
		}
	}
	if !changed {
		t.Error("state should have changed after step")
	}
}

func TestStepBSMLayerMultipleSteps(t *testing.T) {
	hiddenDim := 64
	windowSize := 4

	w := makeTestLayerWeights(t, hiddenDim)
	s := NewBSMLayerState(hiddenDim, windowSize)

	x := bits.AllocWords(hiddenDim)
	x[0] = ^uint64(0)

	dst := bits.AllocWords(hiddenDim)

	// Run several steps — should not panic
	for i := 0; i < 10; i++ {
		StepBSMLayer(x, w, s, dst)
	}

	// After 10 steps with windowSize=4, window should have wrapped
	if !s.windowFull {
		t.Error("window should be full after 10 steps with windowSize=4")
	}
}

func TestStepBSMLayerDifferentInputs(t *testing.T) {
	hiddenDim := 64
	windowSize := 4

	w := makeTestLayerWeights(t, hiddenDim)
	s := NewBSMLayerState(hiddenDim, windowSize)

	// All ones
	x1 := bits.AllocWords(hiddenDim)
	x1[0] = ^uint64(0)

	// All zeros
	x2 := bits.AllocWords(hiddenDim)
	x2[0] = 0

	dst1 := bits.AllocWords(hiddenDim)
	dst2 := bits.AllocWords(hiddenDim)

	StepBSMLayer(x1, w, s, dst1)

	// Reuse same state for different input
	s2 := NewBSMLayerState(hiddenDim, windowSize)
	StepBSMLayer(x2, w, s2, dst2)

	// Different inputs should produce different outputs
	same := true
	for i := range dst1 {
		if dst1[i] != dst2[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("different inputs should produce different outputs")
	}
}
