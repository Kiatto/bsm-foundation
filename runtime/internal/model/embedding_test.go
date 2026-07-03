package model

import (
	"testing"
)

func TestLookupReturnsCorrectShape(t *testing.T) {
	vocabSize := 10
	hiddenDim := 64

	data := make([]byte, vocabSize*hiddenDim/8)
	emb, err := EmbeddingFromBytes(data, vocabSize, hiddenDim)
	if err != nil {
		t.Fatal(err)
	}

	vec := emb.Lookup(3)
	if vec == nil {
		t.Fatal("Lookup returned nil for valid ID")
	}
	if len(vec) != hiddenDim/64 {
		t.Errorf("Lookup len = %d, want %d", len(vec), hiddenDim/64)
	}
}

func TestLookupOOBReturnsNil(t *testing.T) {
	emb, err := EmbeddingFromBytes(make([]byte, 10*8), 10, 64)
	if err != nil {
		t.Fatal(err)
	}

	vec := emb.Lookup(99)
	if vec != nil {
		t.Error("Lookup OOB should return nil")
	}
}

func TestFromBytesRoundtrip(t *testing.T) {
	vocabSize := 4
	hiddenDim := 64
	bytesPerRow := hiddenDim / 8

	// Create test data with known patterns
	data := make([]byte, vocabSize*bytesPerRow)
	for i := range data {
		data[i] = byte(i * 0x33)
	}

	emb, err := EmbeddingFromBytes(data, vocabSize, hiddenDim)
	if err != nil {
		t.Fatal(err)
	}

	// Verify each row converts back correctly
	for v := 0; v < vocabSize; v++ {
		vec := emb.Lookup(uint16(v))
		if vec == nil {
			t.Fatalf("Lookup(%d) returned nil", v)
		}

		rowStart := v * bytesPerRow
		for w := 0; w < hiddenDim/64; w++ {
			var expected uint64
			for b := 0; b < 8; b++ {
				byteIdx := rowStart + w*8 + b
				if byteIdx < len(data) {
					expected |= uint64(data[byteIdx]) << (b * 8)
				}
			}
			if vec[w] != expected {
				t.Errorf("Row %d, word %d: got 0x%X, want 0x%X",
					v, w, vec[w], expected)
			}
		}
	}
}

func TestEmbeddingFromBytesErrors(t *testing.T) {
	_, err := EmbeddingFromBytes(nil, 10, 63) // not multiple of 64
	if err == nil {
		t.Error("expected error for hiddenDim=63")
	}

	_, err = EmbeddingFromBytes(nil, 10, 32) // multiple of 64 but not 64 itself
	if err == nil {
		t.Error("expected error for hiddenDim=32")
	}
}

func TestStats(t *testing.T) {
	vocabSize := 4
	hiddenDim := 64

	data := make([]byte, vocabSize*hiddenDim/8)
	// Set alternating bits for known popcount
	data[0] = 0xAA // 10101010 = 4 ones
	data[8] = 0xFF // 8 ones

	emb, err := EmbeddingFromBytes(data, vocabSize, hiddenDim)
	if err != nil {
		t.Fatal(err)
	}

	stats := emb.Stats()
	if stats.VocabSize != vocabSize {
		t.Errorf("VocabSize = %d, want %d", stats.VocabSize, vocabSize)
	}
	if stats.HiddenDim != hiddenDim {
		t.Errorf("HiddenDim = %d, want %d", stats.HiddenDim, hiddenDim)
	}
	if stats.SizeBytes != vocabSize*hiddenDim/8 {
		t.Errorf("SizeBytes = %d, want %d", stats.SizeBytes, vocabSize*hiddenDim/8)
	}
}
