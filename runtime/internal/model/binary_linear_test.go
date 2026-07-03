package model

import (
	"testing"

	"github.com/blm/runtime/internal/bits"
)

func TestBinaryWeightForward(t *testing.T) {
	outDim := 64
	inDim := 64

	// Balanced weights per row: first 4 bytes +1, last 4 bytes -1
	// This gives dot=0 for all-ones input (64-bit dot = 32 - 32 = 0)
	data := make([]byte, outDim*inDim/8)
	for row := 0; row < outDim; row++ {
		base := row * (inDim / 8)
		for j := 0; j < 4; j++ { // first 32 bits = +1
			data[base+j] = 0xFF
		}
		for j := 4; j < 8; j++ { // last 32 bits = -1
			data[base+j] = 0x00
		}
	}

	w, err := BinaryWeightFromBytes(data, outDim, inDim)
	if err != nil {
		t.Fatal(err)
	}

	x := bits.AllocWords(inDim)
	x[0] = ^uint64(0) // all ones (+1)

	y := make([]int32, outDim)
	w.Forward(x, y)

	for i, v := range y {
		if v != 0 {
			t.Errorf("y[%d] = %d, want 0", i, v)
		}
	}
}

func TestBinaryWeightForwardThreshold(t *testing.T) {
	outDim := 64
	inDim := 64

	// Weight: all ones
	data := make([]byte, outDim*inDim/8)
	for i := range data {
		data[i] = 0xFF
	}

	w, err := BinaryWeightFromBytes(data, outDim, inDim)
	if err != nil {
		t.Fatal(err)
	}

	x := bits.AllocWords(inDim)
	x[0] = ^uint64(0) // input all ones

	dst := make([]uint64, outDim/64)
	w.ForwardThreshold(x, dst)

	// All weights = +1, all inputs = +1 → dot = +64 > 0 → all bits = 1
	for i, v := range dst {
		if v != ^uint64(0) {
			t.Errorf("dst[%d] = 0x%X, want MaxUint64", i, v)
		}
	}
}

func TestBinaryWeightForwardThresholdOpposite(t *testing.T) {
	outDim := 64
	inDim := 64

	data := make([]byte, outDim*inDim/8)
	for i := range data {
		data[i] = 0x00 // weights all 0 → in {-1,+1} representation: all -1
	}

	w, err := BinaryWeightFromBytes(data, outDim, inDim)
	if err != nil {
		t.Fatal(err)
	}

	x := bits.AllocWords(inDim)
	x[0] = ^uint64(0) // input all +1

	dst := make([]uint64, outDim/64)
	w.ForwardThreshold(x, dst)

	// All weights = -1, all inputs = +1 → dot = -64 < 0 → all bits = 0
	for i, v := range dst {
		if v != 0 {
			t.Errorf("dst[%d] = 0x%X, want 0", i, v)
		}
	}
}

func TestBinaryWeightHalfMatch(t *testing.T) {
	outDim := 64
	inDim := 64

	// Weights: first 32 rows all +1 (byte 0xFF), last 32 rows all -1 (byte 0x00)
	data := make([]byte, outDim*inDim/8)
	for i := 0; i < outDim/2; i++ {
		for j := 0; j < inDim/8; j++ {
			data[i*inDim/8+j] = 0xFF
		}
	}
	for i := outDim / 2; i < outDim; i++ {
		for j := 0; j < inDim/8; j++ {
			data[i*inDim/8+j] = 0x00
		}
	}

	w, err := BinaryWeightFromBytes(data, outDim, inDim)
	if err != nil {
		t.Fatal(err)
	}

	x := bits.AllocWords(inDim)
	x[0] = ^uint64(0) // all +1

	y := make([]int32, outDim)
	w.Forward(x, y)

	for i := 0; i < outDim/2; i++ {
		if y[i] != 64 {
			t.Errorf("first half y[%d] = %d, want 64", i, y[i])
		}
	}
	for i := outDim / 2; i < outDim; i++ {
		if y[i] != -64 {
			t.Errorf("second half y[%d] = %d, want -64", i, y[i])
		}
	}
}

func TestBinaryWeightFromBytesErrors(t *testing.T) {
	_, err := BinaryWeightFromBytes(nil, 64, 63)
	if err == nil {
		t.Error("expected error for inDim=63")
	}

	_, err = BinaryWeightFromBytes(nil, 63, 64)
	if err == nil {
		t.Error("expected error for outDim=63")
	}
}

func TestBinaryWeightForwardPanicsOnShortY(t *testing.T) {
	w, _ := BinaryWeightFromBytes(make([]byte, 64*8), 64, 64)
	x := bits.AllocWords(64)

	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for short y")
		}
	}()
	w.Forward(x, make([]int32, 1))
}
