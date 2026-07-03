package main

import (
	"fmt"
	"math/rand"
	"github.com/blm/runtime/internal/blmf"
	"github.com/blm/runtime/internal/model"
)

func main() {
	m, _ := blmf.LoadModel("/tmp/bsm_bench_v4/checkpoint_final.blmf")

	preds := make(map[int]int)
	sess := model.NewBSMSession(m)
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 500; i++ {
		tok := rng.Intn(m.Config.VocabSize)
		logits, _ := sess.Step(tok)
		p := argmax(logits)
		preds[p]++
	}
	fmt.Printf("FP32 argmax over 500 steps:\n")
	for p, c := range preds {
		fmt.Printf("  token %d: %d\n", p, c)
	}
	fmt.Printf("Unique: %d\n", len(preds))
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
