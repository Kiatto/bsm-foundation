package bits

import (
	"math"
	"testing"
)

// --- NWords ---

func TestNWords(t *testing.T) {
	cases := []struct {
		nBits int
		want  int
	}{
		{0, 0},
		{1, 1},
		{63, 1},
		{64, 1},
		{65, 2},
		{128, 2},
		{129, 3},
	}
	for _, c := range cases {
		got := NWords(c.nBits)
		if got != c.want {
			t.Errorf("NWords(%d) = %d, want %d", c.nBits, got, c.want)
		}
	}
}

// --- AllocWords ---

func TestAllocWords(t *testing.T) {
	w := AllocWords(128)
	if len(w) != 2 {
		t.Fatalf("AllocWords(128) len = %d, want 2", len(w))
	}
	for i, word := range w {
		if word != 0 {
			t.Errorf("AllocWords word %d = 0x%X, want 0", i, word)
		}
	}
	w2 := AllocWords(0)
	if len(w2) != 0 {
		t.Errorf("AllocWords(0) len = %d, want 0", len(w2))
	}
}

// --- Pack / Unpack roundtrip ---

func TestPackUnpackRoundtrip(t *testing.T) {
	input := []bool{
		true, false, true, false, false, true, true, true,
		false, true, false, false, false, false, false, false,
		true, true, true, true, true, true, true, true,
	}
	packed := Pack(input)
	unpacked := Unpack(packed, len(input))

	if len(unpacked) != len(input) {
		t.Fatalf("length mismatch: %d vs %d", len(unpacked), len(input))
	}
	for i := range input {
		if input[i] != unpacked[i] {
			t.Errorf("bit %d: got %v, want %v", i, unpacked[i], input[i])
		}
	}
}

func TestPackAllFalse(t *testing.T) {
	bools := make([]bool, 128)
	packed := Pack(bools)
	for i, w := range packed {
		if w != 0 {
			t.Errorf("word %d = 0x%X, want 0", i, w)
		}
	}
}

func TestPackAllTrue(t *testing.T) {
	bools := make([]bool, 128)
	for i := range bools {
		bools[i] = true
	}
	packed := Pack(bools)
	for i, w := range packed {
		if w != math.MaxUint64 {
			t.Errorf("word %d = 0x%X, want MaxUint64", i, w)
		}
	}
}

func TestPackNonMultipleOf64(t *testing.T) {
	bools := []bool{true, false, true}
	packed := Pack(bools)
	if len(packed) != 1 {
		t.Fatalf("len(packed) = %d, want 1", len(packed))
	}
	if packed[0] != 0b101 {
		t.Errorf("packed[0] = 0b%b, want 0b101", packed[0])
	}
	unpacked := Unpack(packed, 3)
	for i := range bools {
		if bools[i] != unpacked[i] {
			t.Errorf("bit %d: got %v, want %v", i, unpacked[i], bools[i])
		}
	}
}

func TestUnpackPartialWord(t *testing.T) {
	words := []uint64{0b1010}
	bools := Unpack(words, 4)
	want := []bool{false, true, false, true}
	for i := range want {
		if bools[i] != want[i] {
			t.Errorf("bit %d: got %v, want %v", i, bools[i], want[i])
		}
	}
}

// --- XORWords ---

func TestXORWordsBasic(t *testing.T) {
	a := []uint64{0xFF00, 0x0F0F}
	b := []uint64{0x0FF0, 0x00FF}
	dst := make([]uint64, 2)
	XORWords(dst, a, b)

	if dst[0] != 0xFF00^0x0FF0 {
		t.Errorf("dst[0] = 0x%X, want 0x%X", dst[0], 0xFF00^0x0FF0)
	}
	if dst[1] != 0x0F0F^0x00FF {
		t.Errorf("dst[1] = 0x%X, want 0x%X", dst[1], 0x0F0F^0x00FF)
	}
}

func TestXORWordsZero(t *testing.T) {
	a := []uint64{0xABCD}
	b := []uint64{0x0000}
	dst := make([]uint64, 1)
	XORWords(dst, a, b)
	if dst[0] != a[0] {
		t.Errorf("XOR with zero: got 0x%X, want 0x%X", dst[0], a[0])
	}
}

func TestXORWordsSelf(t *testing.T) {
	a := []uint64{0xABCD, 0x1234}
	dst := make([]uint64, 2)
	XORWords(dst, a, a)
	for i, v := range dst {
		if v != 0 {
			t.Errorf("self-XOR word %d = 0x%X, want 0", i, v)
		}
	}
}

func TestXORWordsPanicsOnMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on length mismatch")
		}
	}()
	XORWords(make([]uint64, 1), make([]uint64, 1), make([]uint64, 2))
}

// --- XNORWords ---

func TestXNORWords(t *testing.T) {
	a := []uint64{0xFF00}
	b := []uint64{0x0FF0}
	dst := make([]uint64, 1)
	XNORWords(dst, a, b)

	want := ^(a[0] ^ b[0])
	if dst[0] != want {
		t.Errorf("XNOR = 0x%X, want 0x%X", dst[0], want)
	}
}

func TestXNORIdentical(t *testing.T) {
	a := []uint64{0xABCD1234}
	dst := make([]uint64, 1)
	XNORWords(dst, a, a)
	if dst[0] != ^uint64(0) {
		t.Errorf("identical XNOR = 0x%X, want MaxUint64", dst[0])
	}
}

func TestXNORPanicsOnMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on length mismatch")
		}
	}()
	XNORWords(make([]uint64, 1), make([]uint64, 2), make([]uint64, 1))
}

// --- ANDWords ---

func TestANDWords(t *testing.T) {
	a := []uint64{0xFF00}
	b := []uint64{0x0FF0}
	dst := make([]uint64, 1)
	ANDWords(dst, a, b)
	if dst[0] != 0x0F00 {
		t.Errorf("AND = 0x%X, want 0x0F00", dst[0])
	}
}

// --- ORWords ---

func TestORWords(t *testing.T) {
	a := []uint64{0xFF00}
	b := []uint64{0x0FF0}
	dst := make([]uint64, 1)
	ORWords(dst, a, b)
	if dst[0] != 0xFFF0 {
		t.Errorf("OR = 0x%X, want 0xFFF0", dst[0])
	}
}

// --- NOTWords ---

func TestNOTWords(t *testing.T) {
	a := []uint64{0xFF00}
	dst := make([]uint64, 1)
	NOTWords(dst, a)
	if dst[0] != ^uint64(0xFF00) {
		t.Errorf("NOT = 0x%X, want 0x%X", dst[0], ^uint64(0xFF00))
	}
}

func TestNOTPanicsOnMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on length mismatch")
		}
	}()
	NOTWords(make([]uint64, 1), make([]uint64, 2))
}

// --- PopcountWords ---

func TestPopcountWordsSingleWord(t *testing.T) {
	cases := []struct {
		word uint64
		want int
	}{
		{0, 0},
		{0xFFFFFFFFFFFFFFFF, 64},
		{0x1, 1},
		{0x8000000000000000, 1},
		{0xFFFF0000FFFF0000, 32},
	}
	for _, c := range cases {
		got := PopcountWords([]uint64{c.word})
		if got != c.want {
			t.Errorf("PopcountWords(0x%X) = %d, want %d", c.word, got, c.want)
		}
	}
}

func TestPopcountWordsMultiple(t *testing.T) {
	words := []uint64{0xFF, 0xFF00, 0xF0F0}
	// 8 + 8 + 8 = 24
	got := PopcountWords(words)
	if got != 24 {
		t.Errorf("PopcountWords = %d, want 24", got)
	}
}

func TestPopcountWordsEmpty(t *testing.T) {
	got := PopcountWords(nil)
	if got != 0 {
		t.Errorf("PopcountWords(nil) = %d, want 0", got)
	}
}

// --- Majority3 ---

func TestMajority3Basic(t *testing.T) {
	// Bit-by-bit truth table:
	// a b c => majority
	// 0 0 0 => 0
	// 0 0 1 => 0
	// 0 1 0 => 0
	// 0 1 1 => 1
	// 1 0 0 => 0
	// 1 0 1 => 1
	// 1 1 0 => 1
	// 1 1 1 => 1
	a := []uint64{0b10101010}
	b := []uint64{0b11001100}
	c := []uint64{0b11110000}
	dst := make([]uint64, 1)
	Majority3(dst, a, b, c)

	// For each bit position:
	want := uint64(0)
	for i := 0; i < 8; i++ {
		bitA := (a[0] >> i) & 1
		bitB := (b[0] >> i) & 1
		bitC := (c[0] >> i) & 1
		majority := (bitA & bitB) | (bitB & bitC) | (bitA & bitC)
		want |= majority << i
	}
	if dst[0] != want {
		t.Errorf("Majority3 = 0b%b, want 0b%b", dst[0], want)
	}
}

func TestMajority3TwoOutOfThree(t *testing.T) {
	a := []uint64{0xFFFFFFFFFFFFFFFF}
	b := []uint64{0xFFFFFFFFFFFFFFFF}
	c := []uint64{0x0000000000000000}
	dst := make([]uint64, 1)
	Majority3(dst, a, b, c)
	if dst[0] != ^uint64(0) {
		t.Errorf("2/3 majority: got 0x%X, want MaxUint64", dst[0])
	}
}

func TestMajority3OneOutOfThree(t *testing.T) {
	a := []uint64{0xFFFFFFFFFFFFFFFF}
	b := []uint64{0x0000000000000000}
	c := []uint64{0x0000000000000000}
	dst := make([]uint64, 1)
	Majority3(dst, a, b, c)
	if dst[0] != 0 {
		t.Errorf("1/3 majority: got 0x%X, want 0", dst[0])
	}
}

func TestMajority3PanicsOnMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on length mismatch")
		}
	}()
	Majority3(make([]uint64, 1), make([]uint64, 2), make([]uint64, 2), make([]uint64, 2))
}

// --- BinaryDot ---

func TestBinaryDotIdentical(t *testing.T) {
	a := []uint64{0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF}
	b := []uint64{0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF}
	got := BinaryDot(a, b)
	n := len(a) * 64 // 128
	if got != n {
		t.Errorf("identical vectors: got %d, want %d", got, n)
	}
}

func TestBinaryDotOpposite(t *testing.T) {
	a := []uint64{0xFFFFFFFFFFFFFFFF}
	b := []uint64{0x0000000000000000}
	got := BinaryDot(a, b)
	// All XNOR = 0 => total = 0 => dot = -64
	want := -64
	if got != want {
		t.Errorf("opposite vectors: got %d, want %d", got, want)
	}
}

func TestBinaryDotHalfMatch(t *testing.T) {
	a := []uint64{0xFFFF0000FFFF0000}
	b := []uint64{0xFFFFFFFFFFFFFFFF}
	// a has 32 bits set, 32 bits clear
	// xnor(a,b) where b=all-1:
	//   for bits where a=1: 1 XNOR 1 = 1
	//   for bits where a=0: 0 XNOR 1 = 0
	// So 32 matches => dot = 2*32 - 64 = 0
	got := BinaryDot(a, b)
	if got != 0 {
		t.Errorf("half match: got %d, want 0", got)
	}
}

func TestBinaryDotPanicsOnMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on length mismatch")
		}
	}()
	BinaryDot(make([]uint64, 1), make([]uint64, 2))
}

// --- Threshold ---

func TestThresholdBasic(t *testing.T) {
	vals := []int32{1, 5, -3, 0, 10, -10, 100, -100}
	dst := Threshold(vals, 0)
	// Expected: vals[i] > 0 -> 1
	// 1 > 0 => 1, 5 > 0 => 1, -3 > 0 => 0, 0 > 0 => 0
	// 10 > 0 => 1, -10 > 0 => 0, 100 > 0 => 1, -100 > 0 => 0
	// => 0b11001010 = 0xCA (bit 7=1, bit 6=1, bit 3=1, bit 1=1)
	want := uint64(0)
	for _, v := range []int32{1, 5, 10, 100} {
		for i, w := range vals {
			if w == v {
				want |= 1 << i
			}
		}
	}
	if dst[0] != want {
		t.Errorf("Threshold = 0b%b, want 0b%b", dst[0], want)
	}
}

func TestThresholdHighThreshold(t *testing.T) {
	vals := []int32{1, 2, 3, 4, 5}
	dst := Threshold(vals, 3)
	// vals[i] > 3 => vals[3]=4, vals[4]=5 => bits 3, 4
	want := uint64(0b11000)
	if dst[0] != want {
		t.Errorf("threshold=3: got 0b%b, want 0b%b", dst[0], want)
	}
}

// --- WordsEqual ---

func TestWordsEqual(t *testing.T) {
	a := []uint64{1, 2, 3}
	b := []uint64{1, 2, 3}
	if !WordsEqual(a, b) {
		t.Error("identical slices should be equal")
	}
	b[0] = 99
	if WordsEqual(a, b) {
		t.Error("different slices should not be equal")
	}
}

func TestWordsEqualDifferentLengths(t *testing.T) {
	if WordsEqual([]uint64{1}, []uint64{1, 2}) {
		t.Error("different length slices should not be equal")
	}
}

func TestWordsEqualNil(t *testing.T) {
	if !WordsEqual(nil, nil) {
		t.Error("nil slices should be equal")
	}
	if WordsEqual(nil, []uint64{}) {
		// Both have length 0, so they are equal
	}
}

// --- Benchmarks ---

func BenchmarkPopcountWords(b *testing.B) {
	words := make([]uint64, 32) // 2048 bits
	for i := range words {
		words[i] = uint64(i*0x123456789ABCDEF)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		PopcountWords(words)
	}
}

func BenchmarkBinaryDot(b *testing.B) {
	a := make([]uint64, 32)
	bb := make([]uint64, 32)
	for i := range a {
		a[i] = uint64(i * 0x12345678)
		bb[i] = uint64((i + 1) * 0x87654321)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		BinaryDot(a, bb)
	}
}
