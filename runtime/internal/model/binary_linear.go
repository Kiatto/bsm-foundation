package model

import (
	"fmt"

	"github.com/blm/runtime/internal/bits"
)

// BinaryWeight is a binary weight matrix packed in uint64.
// Logical shape: [OutDim, InDim]
// Physical shape: [OutDim][InDim/64] uint64
// Immutable after loading.
type BinaryWeight struct {
	Data   [][]uint64
	OutDim int
	InDim  int
}

// Forward computes y = BinaryLinear(W, x).
//
// x: packed input [InDim/64] uint64 in {-1,+1} representation
// y: pre-allocated [OutDim] int32 — receives dot products in [-InDim, +InDim]
//
// Formula: y[i] = BinaryDot(W[i], x)
//
// Zero allocations in the hot path if y is pre-allocated.
func (w *BinaryWeight) Forward(x []uint64, y []int32) {
	if len(y) < w.OutDim {
		panic("model: BinaryWeight.Forward y too small")
	}
	for i := 0; i < w.OutDim; i++ {
		y[i] = int32(bits.BinaryDot(w.Data[i], x))
	}
}

// ForwardThreshold computes binary matmul + threshold in one pass.
//
// For each group of 64 output neurons, computes 64 dot products and
// packs results: bit=1 if dot > 0.
func (w *BinaryWeight) ForwardThreshold(x []uint64, dst []uint64) {
	if len(dst) < w.OutDim/64 {
		panic("model: BinaryWeight.ForwardThreshold dst too small")
	}

	for outGroup := 0; outGroup < w.OutDim/64; outGroup++ {
		var result uint64
		baseRow := outGroup * 64
		for bit := 0; bit < 64; bit++ {
			neuronIdx := baseRow + bit
			dot := bits.BinaryDot(w.Data[neuronIdx], x)
			if dot > 0 {
				result |= 1 << bit
			}
		}
		dst[outGroup] = result
	}
}

// BinaryWeightFromBytes loads weights from packed bytes (BLMF format).
//
// data: raw bytes, shape [outDim * inDim/8] uint8 packed
// Each row is inDim consecutive bits packed into inDim/8 bytes.
func BinaryWeightFromBytes(data []byte, outDim, inDim int) (*BinaryWeight, error) {
	if inDim%64 != 0 {
		return nil, fmt.Errorf("inDim %d must be multiple of 64", inDim)
	}
	if outDim%64 != 0 {
		return nil, fmt.Errorf("outDim %d must be multiple of 64", outDim)
	}

	wordsPerRow := inDim / 64
	bytesPerRow := inDim / 8
	expectedBytes := outDim * bytesPerRow

	if len(data) < expectedBytes {
		return nil, fmt.Errorf("data too short: %d bytes, need %d", len(data), expectedBytes)
	}

	w := &BinaryWeight{
		Data:   make([][]uint64, outDim),
		OutDim: outDim,
		InDim:  inDim,
	}

	for i := 0; i < outDim; i++ {
		rowBytes := data[i*bytesPerRow : (i+1)*bytesPerRow]
		rowWords := make([]uint64, wordsPerRow)

		for wIdx := 0; wIdx < wordsPerRow; wIdx++ {
			var word uint64
			for b := 0; b < 8; b++ {
				byteIdx := wIdx*8 + b
				if byteIdx < len(rowBytes) {
					word |= uint64(rowBytes[byteIdx]) << (b * 8)
				}
			}
			rowWords[wIdx] = word
		}
		w.Data[i] = rowWords
	}

	return w, nil
}

// BinaryDot is a convenience wrapper for the bits package.
func BinaryDot(a, b []uint64) int {
	return bits.BinaryDot(a, b)
}
