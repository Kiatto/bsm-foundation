package main

import (
	"encoding/binary"
	"fmt"
	"math/rand"
	"os"
)

// BSM v2.6 binary inference engine
// No floating-point operations; only XOR, popcount, sign.
type BSM struct {
	T     [4096][128]int8 // token embeddings
	W1    [128][128]int8  // correction layer 1
	W2    [128][128]int8  // correction layer 2
	Ddec  [32][128]int8   // decoder projection: 128 -> 32
	Bdec  [12][32]int8    // bit decoder: 32 -> 12 bits
	State [128]int8       // current state
}

// NewBSM loads model from .npy files exported by export.py
func NewBSM(dir string) *BSM {
	m := &BSM{}
	mustLoadNpy(dir+"/T.npy", m.T[:][:])
	mustLoadNpy(dir+"/W1.npy", m.W1[:][:])
	mustLoadNpy(dir+"/W2.npy", m.W2[:][:])
	mustLoadNpy(dir+"/D_dec.npy", m.Ddec[:][:])
	mustLoadNpy(dir+"/B_dec.npy", m.Bdec[:][:])
	m.Reset()
	return m
}

// Reset state to all -1 (start-of-sequence)
func (m *BSM) Reset() {
	for i := range m.State {
		m.State[i] = -1
	}
}

// Step runs one inference step: state -> next token
func (m *BSM) Step(tokenID int) int {
	// Embedding: T[token]
	var emb [128]int8
	copy(emb[:], m.T[tokenID][:])

	// a = state XOR emb (element-wise multiply in {-1,+1})
	var a [128]int8
	for i := 0; i < 128; i++ {
		a[i] = m.State[i] * emb[i]
	}

	// h1 = sign(W1 @ a)
	var h1 [128]int8
	for i := 0; i < 128; i++ {
		dot := 0
		for j := 0; j < 128; j++ {
			dot += int(m.W1[i][j]) * int(a[j])
		}
		if dot >= 0 {
			h1[i] = 1
		} else {
			h1[i] = -1
		}
	}

	// h2 = sign(W2 @ h1) * a  (residual)
	var h2raw [128]int8
	for i := 0; i < 128; i++ {
		dot := 0
		for j := 0; j < 128; j++ {
			dot += int(m.W2[i][j]) * int(h1[j])
		}
		if dot >= 0 {
			h2raw[i] = 1
		} else {
			h2raw[i] = -1
		}
	}
	var h2 [128]int8
	for i := 0; i < 128; i++ {
		h2[i] = h2raw[i] * a[i]
	}

	// new_state = state * emb * h2
	var newState [128]int8
	for i := 0; i < 128; i++ {
		newState[i] = m.State[i] * emb[i] * h2[i]
	}
	m.State = newState

	// Decoder: h_dec = sign(Ddec @ state)  (128 -> 32)
	var hDec [32]int8
	for i := 0; i < 32; i++ {
		dot := 0
		for j := 0; j < 128; j++ {
			dot += int(m.Ddec[i][j]) * int(m.State[j])
		}
		if dot >= 0 {
			hDec[i] = 1
		} else {
			hDec[i] = -1
		}
	}

	// Bit decoder: 12 bits from hDec @ Bdec
	var bits [12]uint8
	for i := 0; i < 12; i++ {
		dot := 0
		for j := 0; j < 32; j++ {
			dot += int(m.Bdec[i][j]) * int(hDec[j])
		}
		if dot > 0 {
			bits[i] = 1
		} else {
			bits[i] = 0
		}
	}

	// bits -> token ID (big-endian)
	tokenID = 0
	for i := 0; i < 12; i++ {
		tokenID = tokenID<<1 | int(bits[i])
	}
	if tokenID >= 4096 {
		tokenID = 0
	}
	return tokenID
}

// Generate produces tokens from an initial context.
func (m *BSM) Generate(context []int, steps int) []int {
	m.Reset()
	// Feed context
	for _, t := range context {
		m.Step(t)
	}
	// Generate
	out := make([]int, 0, steps)
	for i := 0; i < steps; i++ {
		next := m.Step(out[len(out)-1])
		out = append(out, next)
	}
	return out
}

// ---- .npy loading (simple float32 format, raw binary after header) ----

func mustLoadNpy(path string, dst []int8) {
	f, err := os.Open(path)
	if err != nil {
		panic(fmt.Sprintf("cannot open %s: %v", path, err))
	}
	defer f.Close()

	// Skip .npy header (find the end marker)
	buf := make([]byte, 8)
	const headerMagic = "\x93NUMPY"
	magic := make([]byte, 6)
	if _, err := f.Read(magic); err != nil {
		panic(fmt.Sprintf("read magic from %s: %v", path, err))
	}
	if string(magic) != headerMagic {
		panic(fmt.Sprintf("bad magic in %s: got %x", path, magic))
	}
	if _, err := f.Read(buf[:2]); err != nil { // version
		panic(err)
	}
	if _, err := f.Read(buf[:4]); err != nil { // header len
		panic(err)
	}
	headerLen := binary.LittleEndian.Uint32(buf[:4])
	if _, err := f.Read(make([]byte, headerLen)); err != nil {
		panic(err)
	}
	// Read all float32 values, convert to int8
	stat, _ := f.Stat()
	dataLen := stat.Size() - 6 - 2 - 4 - int64(headerLen)
	numFloats := dataLen / 4
	data := make([]float32, numFloats)
	if err := binary.Read(f, binary.LittleEndian, data); err != nil {
		panic(fmt.Sprintf("read data from %s: %v", path, err))
	}
	for i, v := range data {
		if i >= len(dst) {
			break
		}
		if v >= 0 {
			dst[i] = 1
		} else {
			dst[i] = -1
		}
	}
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: bsm <model_dir>")
		os.Exit(1)
	}
	m := NewBSM(os.Args[1])
	_ = rand.Intn // unused

	fmt.Println("BSM v2.6 loaded. Enter seed text:")
	var seed string
	fmt.Scanln(&seed)
	fmt.Printf("Seed: %q", seed)
}
