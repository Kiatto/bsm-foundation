package main

import (
	"fmt"
	"math/rand"
	"github.com/blm/runtime/internal/bits"
	"github.com/blm/runtime/internal/blmf"
	"github.com/blm/runtime/internal/model"
)

func main() {
	m, err := blmf.LoadModel("/tmp/bsm_with_tree_v3.blmf")
	if err != nil {
		panic(err)
	}

	nw := bits.NWords(m.Config.HiddenDim)

	// Run model with one token
	sess := model.NewBSMSession(m)
	_, _ = sess.Step(5)

	// Tree head is on the model. Simulate traversal manually.
	tree := m.TreeHead
	fmt.Printf("Tree: NumNodes=%d NumLevels=%d LeafBase=%d Nw=%d\n",
		tree.NumNodes, tree.NumLevels, tree.LeafBase, tree.Nw)

	// Test with many different starting hidden states
	rng := rand.New(rand.NewSource(42))
	for trial := 0; trial < 5; trial++ {
		state := make([]uint64, nw)
		for j := range state {
			state[j] = rng.Uint64()
		}

		fmt.Printf("\nTrial %d: state = %016x %016x\n", trial, state[0], state[1])
		node := 0
		for level := 0; level < tree.NumLevels-1; level++ {
			if node >= tree.NumNodes || len(tree.Nodes[node]) == 0 {
				fmt.Printf("  Level %d: node %d EMPTY, breaking\n", level, node)
				break
			}
			dot := bits.BinaryDot(tree.Nodes[node], state)
			dir := "LEFT"
			if dot > 0 {
				dir = "RIGHT"
			}
			fmt.Printf("  Level %d: node=%d node_w0=%016x dot=%d → %s\n",
				level, node, tree.Nodes[node][0], dot, dir)
			if dot > 0 {
				node = 2*node + 2
			} else {
				node = 2*node + 1
			}
		}
		leafIdx := node - tree.LeafBase
		if leafIdx < 0 || leafIdx >= tree.VocabSize {
			leafIdx = 0
		}
		fmt.Printf("  → leaf node=%d leafIdx=%d token=%d\n", node, leafIdx, leafIdx)
	}

	// Count unique tree predictions across 1000 random inputs
	rng2 := rand.New(rand.NewSource(42))
	preds := make(map[int]int)
	for i := 0; i < 1000; i++ {
		state := make([]uint64, nw)
		for j := range state {
			state[j] = rng2.Uint64()
		}
		p := tree.PredictToken(state)
		preds[p]++
	}
	fmt.Printf("\nPredictToken distribution over 1000 random states:\n")
	for p, c := range preds {
		fmt.Printf("  token %d: %d\n", p, c)
	}

	// Also check with actual model hidden states
	sess2 := model.NewBSMSession(m)
	rng3 := rand.New(rand.NewSource(42))
	preds2 := make(map[int]int)
	for i := 0; i < 200; i++ {
		tok := rng3.Intn(m.Config.VocabSize)
		logits, _ := sess2.StepTree(tok)
		p := argmax(logits)
		preds2[p]++
	}
	fmt.Printf("\nStepTree prediction distribution over 200 steps:\n")
	for p, c := range preds2 {
		fmt.Printf("  token %d: %d\n", p, c)
	}
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
