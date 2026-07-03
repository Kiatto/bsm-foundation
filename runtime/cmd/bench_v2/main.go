package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/blm/runtime/internal/bits"
	"github.com/blm/runtime/internal/blmf"
	"github.com/blm/runtime/internal/model"
)

func main() {
	fmt.Println("=== V2 Benchmark: FP32 Head vs Trained Tree Head ===")
	fmt.Println()

	m, err := blmf.LoadModel("/tmp/bsm_with_tree.blmf")
	if err != nil {
		panic(err)
	}
	if m.TreeHead == nil {
		panic("no tree head in model")
	}

	vocab := m.Config.VocabSize
	hd := m.Config.HiddenDim
	nw := bits.NWords(hd)

	fmt.Printf("Model: V=%d D=%d L=%d\n", vocab, hd, m.Config.NumLayers)
	fmt.Printf("Tree:  nodes=%d levels=%d leafBase=%d\n", m.TreeHead.NumNodes, m.TreeHead.NumLevels, m.TreeHead.LeafBase)
	treeBytes := len(m.TreeHead.TreeBytes())
	fp32Bytes := vocab * hd * 4
	fmt.Printf("Tree size: %d bytes (FP32 head: %d bytes, %.1fx smaller)\n",
		treeBytes, fp32Bytes, float64(fp32Bytes)/float64(treeBytes))
	fmt.Println()

	// Accuracy: run two sessions in lockstep
	sessF := model.NewBSMSession(m)
	sessT := model.NewBSMSession(m)
	rng := rand.New(rand.NewSource(42))

	const Nacc = 500
	correct := 0
	for i := 0; i < Nacc; i++ {
		tok := rng.Intn(vocab)
		logits, _ := sessF.Step(tok)
		fp32Tok := argmax(logits)
		treeLogits, _ := sessT.StepTree(tok)
		treeTok := argmax(treeLogits)
		if treeTok == fp32Tok {
			correct++
		}
	}

	acc := float64(correct) / float64(Nacc) * 100
	fmt.Printf("Accuracy vs FP32 argmax (n=%d): %.1f%% (%d/%d)\n", Nacc, acc, correct, Nacc)
	fmt.Println()

	// Generate synthetic hidden states for head-only benchmarks
	rng = rand.New(rand.NewSource(42))
	synthStates := make([][]uint64, 100)
	for i := range synthStates {
		state := make([]uint64, nw)
		for j := range state {
			state[j] = rng.Uint64()
		}
		synthStates[i] = state
	}

	// Benchmark: tree PredictToken
	const N = 50000
	start := time.Now()
	result := 0
	for i := 0; i < N; i++ {
		result += m.TreeHead.PredictToken(synthStates[i%len(synthStates)])
	}
	treeNs := float64(time.Since(start).Nanoseconds()) / float64(N)
	_ = result

	fmt.Printf("PredictToken (pure tree, %d states): %.0f ns/op\n", len(synthStates), treeNs)

	// Benchmark: FP32 full logits
	start = time.Now()
	headW := m.HeadW
	result2 := 0
	for i := 0; i < N; i++ {
		state := synthStates[i%len(synthStates)]
		unpacked := make([]float32, hd)
		for j := 0; j < hd; j++ {
			if (state[j/64]>>(j%64))&1 == 1 {
				unpacked[j] = 1.0
			} else {
				unpacked[j] = -1.0
			}
		}
		bestIdx := 0
		bestVal := float32(-1e10)
		for v := 0; v < vocab; v++ {
			var dot float32
			base := v * hd
			for j := 0; j < hd; j++ {
				dot += headW[base+j] * unpacked[j]
			}
			if dot > bestVal {
				bestVal = dot
				bestIdx = v
			}
		}
		result2 += bestIdx
	}
	fp32HeadNs := float64(time.Since(start).Nanoseconds()) / float64(N)
	_ = result2

	fmt.Printf("FP32 full logits (V*D=%d):     %.0f ns/op\n", vocab*hd, fp32HeadNs)
	fmt.Printf("Head-only speedup:              %.0fx\n", fp32HeadNs/treeNs)
	fmt.Println()

	// E2E benchmark: StepTree vs Step (full pipeline)
	start = time.Now()
	sessB := model.NewBSMSession(m)
	rngB := rand.New(rand.NewSource(42))
	for i := 0; i < N; i++ {
		tok := rngB.Intn(vocab)
		_, _ = sessB.StepTree(tok)
	}
	e2eTreeNs := float64(time.Since(start).Nanoseconds()) / float64(N)

	start = time.Now()
	sessB2 := model.NewBSMSession(m)
	rngB2 := rand.New(rand.NewSource(42))
	for i := 0; i < N; i++ {
		tok := rngB2.Intn(vocab)
		_, _ = sessB2.Step(tok)
	}
	e2eFp32Ns := float64(time.Since(start).Nanoseconds()) / float64(N)

	fmt.Printf("End-to-end StepTree:    %.0f ns/op\n", e2eTreeNs)
	fmt.Printf("End-to-end Step (FP32): %.0f ns/op\n", e2eFp32Ns)
	fmt.Printf("End-to-end speedup:     %.1fx\n", e2eFp32Ns/e2eTreeNs)
	fmt.Println()

	fmt.Println("========================================")
	fmt.Println("V2 BENCHMARK SUMMARY")
	fmt.Println("========================================")
	fmt.Printf("  Accuracy (vs FP32):    %.1f%% top-1\n", acc)
	fmt.Printf("  PredictToken:          %.0f ns/op\n", treeNs)
	fmt.Printf("  FP32 full logits:     %.0f ns/op\n", fp32HeadNs)
	fmt.Printf("  Head-only speedup:     %.0fx\n", fp32HeadNs/treeNs)
	fmt.Printf("  E2E StepTree:          %.0f ns/op\n", e2eTreeNs)
	fmt.Printf("  E2E Step (FP32):       %.0f ns/op\n", e2eFp32Ns)
	fmt.Printf("  E2E speedup:           %.1fx\n", e2eFp32Ns/e2eTreeNs)
	fmt.Printf("  Tree size:             %d bytes\n", treeBytes)
	fmt.Printf("  FP32 head size:        %d bytes\n", fp32Bytes)
	fmt.Printf("  Memory ratio:          %.1fx\n", float64(fp32Bytes)/float64(treeBytes))
}

func argmax(vals []float32) int {
	idx := 0
	for i := 1; i < len(vals); i++ {
		if vals[i] > vals[idx] {
			idx = i
		}
	}
	return idx
}
