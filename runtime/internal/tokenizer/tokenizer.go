// Package tokenizer implements BPE tokenization for the BLM runtime.
//
// Must produce IDENTICAL token IDs to the Python BPETokenizer
// given the same vocabulary JSON file.
package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
)

// Special token IDs (must match Python SPECIAL_TOKENS).
const (
	PadID uint16 = 0
	UnkID uint16 = 1
	BosID uint16 = 2
	EosID uint16 = 3
)

// MergePair is a BPE merge rule: (left, right) -> left+right.
type MergePair [2]string

// VocabData matches the Python save format.
type VocabData struct {
	VocabSize int                `json:"vocab_size"`
	Vocab     map[string]uint16  `json:"vocab"`
	Merges    [][2]string        `json:"merges"`
}

// BPETokenizer is a BPE tokenizer loaded from a JSON vocabulary file.
type BPETokenizer struct {
	vocab       map[string]uint16
	idToToken   map[uint16]string
	merges      []MergePair
	mergeLookup map[MergePair]bool // O(1) membership
	vocabSize   int
}

// Load loads a tokenizer from a JSON file saved by the Python trainer.
func Load(path string) (*BPETokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read tokenizer file: %w", err)
	}

	var vd VocabData
	if err := json.Unmarshal(data, &vd); err != nil {
		return nil, fmt.Errorf("parse tokenizer JSON: %w", err)
	}

	t := &BPETokenizer{
		vocab:       vd.Vocab,
		idToToken:   make(map[uint16]string, len(vd.Vocab)),
		merges:      make([]MergePair, len(vd.Merges)),
		mergeLookup: make(map[MergePair]bool),
		vocabSize:   vd.VocabSize,
	}

	// Build reverse map
	for token, id := range t.vocab {
		t.idToToken[id] = token
	}

	// Load merges in order
	for i, m := range vd.Merges {
		pair := MergePair{m[0], m[1]}
		t.merges[i] = pair
		t.mergeLookup[pair] = true
	}

	return t, nil
}

// Encode converts text to token IDs.
// Must produce identical results to Python BPETokenizer.encode().
func (t *BPETokenizer) Encode(text string) ([]uint16, error) {
	words := pretokenize(text)
	var tokenStrings []string

	for _, word := range words {
		if _, ok := t.vocab[word]; ok {
			tokenStrings = append(tokenStrings, word)
			continue
		}

		// Split word into characters (as string tokens)
		chars := make([]string, 0, len(word))
		for _, r := range word {
			chars = append(chars, string(r))
		}

		// Apply merges greedily in training order
		changed := true
		for changed {
			changed = false
			for _, pair := range t.merges {
				merged := pair[0] + pair[1]
				i := 0
				for i < len(chars)-1 {
					if chars[i] == pair[0] && chars[i+1] == pair[1] {
						// Apply merge
						newChars := make([]string, 0, len(chars)-1)
						newChars = append(newChars, chars[:i]...)
						newChars = append(newChars, merged)
						newChars = append(newChars, chars[i+2:]...)
						chars = newChars
						changed = true
					} else {
						i++
					}
				}
			}
		}
		tokenStrings = append(tokenStrings, chars...)
	}

	// Build ID sequence with BOS/EOS
	ids := make([]uint16, 0, len(tokenStrings)+2)
	ids = append(ids, BosID)
	for _, token := range tokenStrings {
		if id, ok := t.vocab[token]; ok {
			ids = append(ids, id)
		} else {
			ids = append(ids, UnkID)
		}
	}
	ids = append(ids, EosID)

	return ids, nil
}

// Decode converts token IDs back to text.
// Ignores PAD, BOS, EOS. Unktokens are skipped.
func (t *BPETokenizer) Decode(ids []uint16) string {
	var parts []string
	for _, id := range ids {
		if id == PadID || id == BosID || id == EosID {
			continue
		}
		if token, ok := t.idToToken[id]; ok {
			parts = append(parts, token)
		}
	}

	// Join all parts
	result := ""
	for _, p := range parts {
		result += p
	}
	return result
}

// VocabSize returns the vocabulary size.
func (t *BPETokenizer) VocabSize() int { return t.vocabSize }

// PadID returns the padding token ID.
func (t *BPETokenizer) PadID() uint16 { return PadID }

// BosID returns the beginning-of-sequence token ID.
func (t *BPETokenizer) BosID() uint16 { return BosID }

// EosID returns the end-of-sequence token ID.
func (t *BPETokenizer) EosID() uint16 { return EosID }

// UnkID returns the unknown token ID.
func (t *BPETokenizer) UnkID() uint16 { return UnkID }

// pretokenize splits text on whitespace and punctuation.
// Must match Python BPETokenizer._pretokenize().
func pretokenize(text string) []string {
	var words []string
	var current []rune

	flush := func() {
		if len(current) > 0 {
			words = append(words, string(current))
			current = nil
		}
	}

	for _, r := range text {
		if isSeparator(r) {
			flush()
			words = append(words, string(r))
		} else {
			current = append(current, r)
		}
	}
	flush()

	return words
}

func isSeparator(r rune) bool {
	return r == ' ' || r == '\t' || r == '\n' || r == '\r' ||
		r == '.' || r == ',' || r == '!' || r == '?' ||
		r == ';' || r == ':' || r == '"' || r == '\'' ||
		r == '(' || r == ')' || r == '-'
}
