package model

import (
	"fmt"

	"github.com/blm/runtime/internal/bits"
)

// BinaryTreeHead replaces the FP32 output head with binary tree traversal.
//
// Architecture: complete binary tree over the vocabulary.
// Each internal node has a binary weight vector (packed bits).
// Traversal: at each node, BinaryDot(hidden_state, node_weight) > 0 → right child.
//
// Tree layout (heap-indexed):
//   Node 0 = root
//   For node i: left child = 2*i + 1, right child = 2*i + 2
//   Leaves start at leafBase = nextPow2(VocabSize) - 1
//   Token ID = leafIndex - leafBase
type BinaryTreeHead struct {
	Nodes     [][]uint64 // [numNodes][nw] — binary weights per node
	NumNodes  int
	NumLevels int
	Nw        int
	VocabSize int
	LeafBase  int // first leaf index = next_pow2(VocabSize) - 1
}

// NewBinaryTreeHeadFromFloat32 constructs a binary tree from FP32 head weights.
//
// Construction (range-based):
//   Complete binary tree over leaf indices [0, leafCount).
//   For each internal node covering range [lo, hi):
//     mid = (lo + hi) / 2
//     left_set  = tokens with indices in [lo, min(mid, vocabSize))
//     right_set = tokens with indices in [mid, min(hi, vocabSize))
//     centroid = mean(head_w[right]) - mean(head_w[left])
//     node_weight = sign(centroid)
//   Traversal: dot > 0 → right child, else left child.
func NewBinaryTreeHeadFromFloat32(headW []float32, vocabSize, hiddenDim int) *BinaryTreeHead {
	nw := bits.NWords(hiddenDim)

	leafCount := 1
	for leafCount < vocabSize {
		leafCount <<= 1
	}

	numNodes := 2*leafCount - 1
	leafBase := leafCount - 1
	numLevels := 0
	for x := leafCount; x > 0; x >>= 1 {
		numLevels++
	}

	nodes := make([][]uint64, numNodes)

	tokenVecs := make([][]float32, vocabSize)
	for i := 0; i < vocabSize; i++ {
		tokenVecs[i] = headW[i*hiddenDim : (i+1)*hiddenDim]
	}

	var buildNode func(nodeIdx, lo, hi int)
	buildNode = func(nodeIdx, lo, hi int) {
		if nodeIdx >= numNodes || hi-lo <= 1 {
			return
		}
		mid := (lo + hi) / 2

		leftLo := lo
		leftHi := mid
		if leftHi > vocabSize {
			leftHi = vocabSize
		}
		rightLo := mid
		rightHi := hi
		if rightHi > vocabSize {
			rightHi = vocabSize
		}

		if leftLo >= leftHi || rightLo >= rightHi {
			buildNode(2*nodeIdx+1, lo, mid)
			buildNode(2*nodeIdx+2, mid, hi)
			return
		}

		nLeft := leftHi - leftLo
		nRight := rightHi - rightLo

		centroid := make([]float32, hiddenDim)
		for t := rightLo; t < rightHi; t++ {
			for j := 0; j < hiddenDim; j++ {
				centroid[j] += tokenVecs[t][j]
			}
		}
		for j := 0; j < hiddenDim; j++ {
			centroid[j] /= float32(nRight)
		}
		for t := leftLo; t < leftHi; t++ {
			for j := 0; j < hiddenDim; j++ {
				centroid[j] -= tokenVecs[t][j]
			}
		}
		for j := 0; j < hiddenDim; j++ {
			centroid[j] /= float32(nLeft)
		}

		packed := make([]uint64, nw)
		for j := 0; j < hiddenDim; j++ {
			if centroid[j] > 0 {
				packed[j/64] |= 1 << (j % 64)
			}
		}
		nodes[nodeIdx] = packed

		buildNode(2*nodeIdx+1, lo, mid)
		buildNode(2*nodeIdx+2, mid, hi)
	}

	buildNode(0, 0, leafCount)

	return &BinaryTreeHead{
		Nodes:     nodes,
		NumNodes:  numNodes,
		NumLevels: numLevels,
		Nw:        nw,
		VocabSize: vocabSize,
		LeafBase:  leafBase,
	}
}

// PredictToken returns the leaf token ID for a given hidden state.
// Greedy: follows the deterministic binary path.
func (h *BinaryTreeHead) PredictToken(state []uint64) int {
	node := 0
	for level := 0; level < h.NumLevels-1; level++ {
		if node >= h.NumNodes || len(h.Nodes[node]) == 0 {
			break
		}
		dot := bits.BinaryDot(h.Nodes[node], state)
		if dot > 0 {
			node = 2*node + 2
		} else {
			node = 2*node + 1
		}
	}
	leafIdx := node - h.LeafBase
	if leafIdx < 0 || leafIdx >= h.VocabSize {
		return 0
	}
	return leafIdx
}

// PredictTopK returns the top-K token IDs by traversing the tree
// and scoring candidate leaves.
//
// Strategy: traverse greedily, then score nearby leaves using BinaryDot.
// For K candidates, we keep a beam of K paths.
func (h *BinaryTreeHead) PredictTopK(state []uint64, k int) []int {
	if k >= h.VocabSize {
		result := make([]int, h.VocabSize)
		for i := 0; i < h.VocabSize; i++ {
			result[i] = i
		}
		return result
	}

	beam := make([]beamEntry, 0, k*2)
	beam = append(beam, beamEntry{node: 0, score: 0, level: 0, pathSet: false})

	for level := 0; level < h.NumLevels-1; level++ {
		if len(beam) == 0 {
			break
		}

		candidates := make([]beamEntry, 0, len(beam)*2)
		for _, b := range beam {
			if b.node >= h.NumNodes || len(h.Nodes[b.node]) == 0 {
				continue
			}
			dot := bits.BinaryDot(h.Nodes[b.node], state)

			leftNode := 2*b.node + 1
			rightNode := 2*b.node + 2

			leftScore := b.score
			if dot <= 0 {
				leftScore -= dot
			}
			rightScore := b.score
			if dot > 0 {
				rightScore += dot
			}

			candidates = append(candidates,
				beamEntry{node: leftNode, score: leftScore, level: level + 1, pathSet: false},
				beamEntry{node: rightNode, score: rightScore, level: level + 1, pathSet: false},
			)
		}

		if len(candidates) > k {
			sortByScoreDesc(candidates)
			beam = candidates[:k]
		} else {
			beam = candidates
		}
	}

	result := make([]int, 0, len(beam))
	seen := make(map[int]bool)
	for _, b := range beam {
		leafIdx := b.node - h.LeafBase
		if leafIdx >= 0 && leafIdx < h.VocabSize && !seen[leafIdx] {
			result = append(result, leafIdx)
			seen[leafIdx] = true
		}
	}

	sortByScoreDescEntry(result, beam[:len(result)])
	return result
}

func sortByScoreDesc(entries []beamEntry) {
	for i := 0; i < len(entries); i++ {
		for j := i + 1; j < len(entries); j++ {
			if entries[j].score > entries[i].score {
				entries[i], entries[j] = entries[j], entries[i]
			}
		}
	}
}

func sortByScoreDescEntry(ids []int, entries []beamEntry) {
	for i := 0; i < len(ids); i++ {
		for j := i + 1; j < len(ids); j++ {
			if entries[j].score > entries[i].score {
				ids[i], ids[j] = ids[j], ids[i]
				entries[i], entries[j] = entries[j], entries[i]
			}
		}
	}
}

type beamEntry struct {
	node    int
	score   int
	level   int
	pathSet bool
}

// TreeBytes returns the packed tree for serialization.
// Format: [numNodes:uint32] [nw:uint32] [vocabSize:uint32] [leafBase:uint32]
//         [node_0_bits...] [node_1_bits...] ...
func (h *BinaryTreeHead) TreeBytes() []byte {
	headerSize := 16
	wordSize := h.Nw * 8
	dataSize := h.NumNodes * wordSize
	buf := make([]byte, headerSize+dataSize)

	binaryLittleEndianPutUint32(buf[0:4], uint32(h.NumNodes))
	binaryLittleEndianPutUint32(buf[4:8], uint32(h.Nw))
	binaryLittleEndianPutUint32(buf[8:12], uint32(h.VocabSize))
	binaryLittleEndianPutUint32(buf[12:16], uint32(h.LeafBase))

	for i := 0; i < h.NumNodes; i++ {
		base := headerSize + i*wordSize
		if i < len(h.Nodes) && len(h.Nodes[i]) > 0 {
			for w := 0; w < h.Nw && w < len(h.Nodes[i]); w++ {
				binaryLittleEndianPutUint64(buf[base+w*8:base+w*8+8], h.Nodes[i][w])
			}
		}
	}
	return buf
}

// TreeFromBytes deserializes a tree from packed bytes.
func TreeFromBytes(data []byte) (*BinaryTreeHead, error) {
	if len(data) < 16 {
		return nil, fmt.Errorf("tree data too short: %d bytes", len(data))
	}

	numNodes := int(binaryLittleEndianUint32(data[0:4]))
	nw := int(binaryLittleEndianUint32(data[4:8]))
	vocabSize := int(binaryLittleEndianUint32(data[8:12]))
	leafBase := int(binaryLittleEndianUint32(data[12:16]))

	wordSize := nw * 8
	expectedSize := 16 + numNodes*wordSize
	if len(data) < expectedSize {
		return nil, fmt.Errorf("tree data too short: %d < %d", len(data), expectedSize)
	}

	nodes := make([][]uint64, numNodes)
	for i := 0; i < numNodes; i++ {
		base := 16 + i*wordSize
		row := make([]uint64, nw)
		for w := 0; w < nw; w++ {
			row[w] = binaryLittleEndianUint64(data[base+w*8 : base+w*8+8])
		}
		nodes[i] = row
	}

	leafCount := 1
	for leafCount < vocabSize {
		leafCount <<= 1
	}
	numLevels := 0
	for x := leafCount; x > 0; x >>= 1 {
		numLevels++
	}

	return &BinaryTreeHead{
		Nodes:     nodes,
		NumNodes:  numNodes,
		NumLevels: numLevels,
		Nw:        nw,
		VocabSize: vocabSize,
		LeafBase:  leafBase,
	}, nil
}

func binaryLittleEndianPutUint32(b []byte, v uint32) {
	b[0] = byte(v)
	b[1] = byte(v >> 8)
	b[2] = byte(v >> 16)
	b[3] = byte(v >> 24)
}

func binaryLittleEndianPutUint64(b []byte, v uint64) {
	b[0] = byte(v)
	b[1] = byte(v >> 8)
	b[2] = byte(v >> 16)
	b[3] = byte(v >> 24)
	b[4] = byte(v >> 32)
	b[5] = byte(v >> 40)
	b[6] = byte(v >> 48)
	b[7] = byte(v >> 56)
}

func binaryLittleEndianUint32(b []byte) uint32 {
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
}

func binaryLittleEndianUint64(b []byte) uint64 {
	return uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 |
		uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56
}
