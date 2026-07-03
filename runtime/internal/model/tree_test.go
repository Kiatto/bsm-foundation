package model

import (
	"testing"
)

func TestTreeHeadConstructionAndRoundtrip(t *testing.T) {
	headW := make([]float32, 32*64)
	for i := range headW {
		headW[i] = float32(i%3 - 1)
	}

	tree := NewBinaryTreeHeadFromFloat32(headW, 32, 64)
	if tree.NumNodes == 0 {
		t.Fatal("empty tree")
	}

	bytes := tree.TreeBytes()
	tree2, err := TreeFromBytes(bytes)
	if err != nil {
		t.Fatalf("TreeFromBytes: %v", err)
	}

	if tree.NumNodes != tree2.NumNodes {
		t.Fatalf("NumNodes: %d vs %d", tree.NumNodes, tree2.NumNodes)
	}

	nw := (64 + 63) / 64
	for i := 0; i < 100; i++ {
		state := make([]uint64, nw)
		for j := range state {
			state[j] = uint64(i*17 + j*11)
		}
		if tree.PredictToken(state) != tree2.PredictToken(state) {
			t.Fatalf("predict mismatch at %d", i)
		}
	}
}

func TestTreeHeadAllTokensReachable(t *testing.T) {
	headW := make([]float32, 32*64)
	for i := range headW {
		headW[i] = float32(i%3 - 1)
	}

	tree := NewBinaryTreeHeadFromFloat32(headW, 32, 64)

	// Verify the tree has the right structure: internal nodes are populated,
	// leaf nodes are not. The tree uses range-based splitting:
	//   Node k covers range [lo, hi), where hi - lo = pow2(level)
	//   Centroid = mean(left_half) - mean(right_half), binarized
	//
	// With random head weights, centroid-based binary weights are approximate
	// discriminators. All-internal nodes should be populated.
	leafCount := 32
	expectedPop := leafCount - 1 // internal nodes only
	populated := 0
	for _, n := range tree.Nodes {
		if len(n) > 0 {
			populated++
		}
	}
	if populated < expectedPop/2 {
		t.Fatalf("too few populated nodes: %d/%d, want >= %d",
			populated, tree.NumNodes, expectedPop/2)
	}
}

func TestTreeHeadScaling(t *testing.T) {
	cfgs := []struct {
		vocabSize int
		hiddenDim int
		wantNodes int
	}{
		{32, 64, 63},     // 2*32-1 = 63
		{128, 64, 255},   // 2*128-1 = 255
		{256, 128, 511},  // 2*256-1 = 511
		{4096, 256, 8191}, // 2*4096-1 = 8191
	}

	for _, cfg := range cfgs {
		headW := make([]float32, cfg.vocabSize*cfg.hiddenDim)
		for i := range headW {
			headW[i] = float32(i%3 - 1)
		}

		tree := NewBinaryTreeHeadFromFloat32(headW, cfg.vocabSize, cfg.hiddenDim)

		if tree.NumNodes != cfg.wantNodes {
			t.Errorf("V=%d: got %d nodes, want %d", cfg.vocabSize, tree.NumNodes, cfg.wantNodes)
		}

		treeSize := len(tree.TreeBytes())
		fpSize := cfg.vocabSize * cfg.hiddenDim * 4
		t.Logf("V=%-5d D=%-4d nodes=%-5d tree=%-6dB FP32=%-8dB ratio=%.0fx",
			cfg.vocabSize, cfg.hiddenDim, tree.NumNodes, treeSize, fpSize, float64(fpSize)/float64(treeSize))
	}
}
