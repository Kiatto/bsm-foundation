// Package bits implements bitwise operations for binary neural networks.
//
// All operations work on packed uint64 slices where each uint64 stores 64 bits.
// Bit 0 of word 0 corresponds to index 0, bit 63 of word 0 corresponds to index 63,
// bit 0 of word 1 corresponds to index 64, etc.
//
// This package is the computational core of the BLM runtime.
// Zero external dependencies.
package bits

import "math/bits"

// NWords returns the number of uint64 words needed to store n bits.
func NWords(nBits int) int {
	return (nBits + 63) / 64
}

// AllocWords allocates a []uint64 of size NWords(nBits), zero-initialized.
func AllocWords(nBits int) []uint64 {
	return make([]uint64, NWords(nBits))
}

// Pack converts a slice of bool to packed uint64 words.
// bools[i] corresponds to bit i in the packed representation (bit 0 = LSB).
// If len(bools) is not a multiple of 64, the remaining bits are zero-padded.
func Pack(bools []bool) []uint64 {
	n := len(bools)
	words := make([]uint64, NWords(n))
	for i, b := range bools {
		if b {
			words[i/64] |= 1 << (i % 64)
		}
	}
	return words
}

// Unpack converts packed uint64 words back to a slice of bool of length n.
func Unpack(words []uint64, n int) []bool {
	bools := make([]bool, n)
	for i := 0; i < n; i++ {
		wordIdx := i / 64
		bitIdx := i % 64
		if wordIdx < len(words) {
			bools[i] = (words[wordIdx]>>bitIdx)&1 == 1
		}
	}
	return bools
}

// XORWords computes dst[i] = a[i] ^ b[i] for all i.
// Panics if a, b, and dst have different lengths.
func XORWords(dst, a, b []uint64) {
	if len(a) != len(b) || len(dst) != len(a) {
		panic("bits: XORWords length mismatch")
	}
	for i := range a {
		dst[i] = a[i] ^ b[i]
	}
}

// XNORWords computes dst[i] = ^(a[i] ^ b[i]) for all i.
// Panics if a, b, and dst have different lengths.
func XNORWords(dst, a, b []uint64) {
	if len(a) != len(b) || len(dst) != len(a) {
		panic("bits: XNORWords length mismatch")
	}
	for i := range a {
		dst[i] = ^(a[i] ^ b[i])
	}
}

// ANDWords computes dst[i] = a[i] & b[i] for all i.
// Panics if a, b, and dst have different lengths.
func ANDWords(dst, a, b []uint64) {
	if len(a) != len(b) || len(dst) != len(a) {
		panic("bits: ANDWords length mismatch")
	}
	for i := range a {
		dst[i] = a[i] & b[i]
	}
}

// ORWords computes dst[i] = a[i] | b[i] for all i.
// Panics if a, b, and dst have different lengths.
func ORWords(dst, a, b []uint64) {
	if len(a) != len(b) || len(dst) != len(a) {
		panic("bits: ORWords length mismatch")
	}
	for i := range a {
		dst[i] = a[i] | b[i]
	}
}

// NOTWords computes dst[i] = ^a[i] for all i.
// Panics if dst and a have different lengths.
func NOTWords(dst, a []uint64) {
	if len(dst) != len(a) {
		panic("bits: NOTWords length mismatch")
	}
	for i := range a {
		dst[i] = ^a[i]
	}
}

// PopcountWords returns the total number of 1-bits across all words.
func PopcountWords(words []uint64) int {
	total := 0
	for _, w := range words {
		total += bits.OnesCount64(w)
	}
	return total
}

// Majority3 computes the bitwise majority of three vectors.
// dst[i] = (a[i] & b[i]) | (b[i] & c[i]) | (a[i] & c[i])
// Panics if all inputs have different lengths.
func Majority3(dst, a, b, c []uint64) {
	if len(a) != len(b) || len(a) != len(c) || len(dst) != len(a) {
		panic("bits: Majority3 length mismatch")
	}
	for i := range a {
		// majority(a,b,c) = (a&b) | (b&c) | (a&c)
		dst[i] = (a[i] & b[i]) | (b[i] & c[i]) | (a[i] & c[i])
	}
}

// BinaryDot computes the binary dot product between two packed vectors.
//
// For vectors in {-1, +1} representation:
//   dot(a, b) = popcount(xnor(a, b)) - popcount(xor(a, b))
//             = 2*popcount(xnor(a,b)) - n
//
// Returns an int in range [-n, +n] where n = len(a)*64.
func BinaryDot(a, b []uint64) int {
	if len(a) != len(b) {
		panic("bits: BinaryDot length mismatch")
	}
	total := 0
	n := len(a) * 64
	for i := range a {
		match := ^(a[i] ^ b[i])
		total += bits.OnesCount64(match)
	}
	// dot = 2*match - n
	return 2*total - n
}

// Threshold converts int32 values to packed binary based on a threshold.
// Bit i is 1 if vals[i] > threshold, otherwise 0.
// dst must have length NWords(len(vals)).
func Threshold(vals []int32, threshold int32) []uint64 {
	dst := AllocWords(len(vals))
	for i, v := range vals {
		if v > threshold {
			dst[i/64] |= 1 << (i % 64)
		}
	}
	return dst
}

// WordsEqual returns true if two uint64 slices have identical content.
func WordsEqual(a, b []uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
