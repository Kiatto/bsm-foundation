// Package model implements BSM model weights and forward pass.
package model

import (
	"fmt"

	"github.com/blm/runtime/internal/bits"
)

// BinaryEmbedding is a lookup table mapping token IDs to binary vectors.
// Immutable after loading. Supports mmap-friendly zero-copy loading.
type BinaryEmbedding struct {
	Table     [][]uint64 // [VocabSize][HiddenDim/64]
	VocabSize int
	HiddenDim int
}

// EmbeddingStats contains statistics for debugging/inspection.
type EmbeddingStats struct {
	VocabSize   int
	HiddenDim   int
	SizeBytes   int
	AvgBitRatio float64 // average fraction of 1-bits per embedding
}

// Lookup returns the binary vector for a token ID.
// Returns nil if tokenID >= VocabSize.
func (e *BinaryEmbedding) Lookup(tokenID uint16) []uint64 {
	if int(tokenID) >= e.VocabSize {
		return nil
	}
	return e.Table[tokenID]
}

// LookupBatch returns the binary vectors for a sequence of tokens.
func (e *BinaryEmbedding) LookupBatch(tokenIDs []uint16) [][]uint64 {
	result := make([][]uint64, len(tokenIDs))
	for i, id := range tokenIDs {
		result[i] = e.Lookup(id)
	}
	return result
}

// EmbeddingFromBytes constructs a BinaryEmbedding from packed bytes.
//
// data: raw bytes from BLMF file, shape [vocabSize * hiddenDim/8]
// Each byte packs 8 bits (LSB first). Data is reorganized into []uint64 words.
func EmbeddingFromBytes(data []byte, vocabSize, hiddenDim int) (*BinaryEmbedding, error) {
	if hiddenDim%64 != 0 {
		return nil, fmt.Errorf("hiddenDim %d must be multiple of 64", hiddenDim)
	}
	if hiddenDim%8 != 0 {
		return nil, fmt.Errorf("hiddenDim %d must be multiple of 8", hiddenDim)
	}

	wordsPerRow := hiddenDim / 64
	bytesPerRow := hiddenDim / 8

	expectedBytes := vocabSize * bytesPerRow
	if len(data) < expectedBytes {
		return nil, fmt.Errorf("data too short: %d bytes, need %d", len(data), expectedBytes)
	}

	table := make([][]uint64, vocabSize)
	for i := 0; i < vocabSize; i++ {
		rowBytes := data[i*bytesPerRow : (i+1)*bytesPerRow]
		rowWords := make([]uint64, wordsPerRow)

		// Convert bytes to uint64 (LSB first within each byte,
		// little-endian across bytes within each word)
		for w := 0; w < wordsPerRow; w++ {
			var word uint64
			for b := 0; b < 8; b++ {
				byteIdx := w*8 + b
				if byteIdx < len(rowBytes) {
					word |= uint64(rowBytes[byteIdx]) << (b * 8)
				}
			}
			rowWords[w] = word
		}

		table[i] = rowWords
	}

	return &BinaryEmbedding{
		Table:     table,
		VocabSize: vocabSize,
		HiddenDim: hiddenDim,
	}, nil
}

// Stats returns statistics for inspection/debug.
func (e *BinaryEmbedding) Stats() EmbeddingStats {
	totalBits := e.VocabSize * e.HiddenDim
	totalOnes := 0

	for _, row := range e.Table {
		totalOnes += bits.PopcountWords(row)
	}

	return EmbeddingStats{
		VocabSize:   e.VocabSize,
		HiddenDim:   e.HiddenDim,
		SizeBytes:   e.VocabSize * e.HiddenDim / 8,
		AvgBitRatio: float64(totalOnes) / float64(totalBits),
	}
}
