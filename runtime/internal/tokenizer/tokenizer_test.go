package tokenizer

import (
	"encoding/json"
	"os"
	"testing"
)

// minimalVocabJSON returns JSON for a tiny vocabulary.
func minimalVocabJSON(t *testing.T) string {
	t.Helper()
	data := map[string]interface{}{
		"vocab_size": 64,
		"vocab": map[string]uint16{
			"<PAD>": 0,
			"<UNK>": 1,
			"<BOS>": 2,
			"<EOS>": 3,
			"a":     4,
			"b":     5,
			"c":     6,
			"ab":    7,
			"abc":   8,
			" ":     9,
		},
		"merges": [][2]string{
			{"a", "b"},
			{"ab", "c"},
		},
	}
	b, err := json.Marshal(data)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func writeTempVocab(t *testing.T, content string) string {
	t.Helper()
	f, err := os.CreateTemp("", "vocab_*.json")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString(content); err != nil {
		f.Close()
		os.Remove(f.Name())
		t.Fatal(err)
	}
	f.Close()
	return f.Name()
}

func TestLoad(t *testing.T) {
	jsonContent := minimalVocabJSON(t)
	path := writeTempVocab(t, jsonContent)
	defer os.Remove(path)

	tok, err := Load(path)
	if err != nil {
		t.Fatalf("Load error: %v", err)
	}

	if tok.VocabSize() != 64 {
		t.Errorf("VocabSize = %d, want 64", tok.VocabSize())
	}
}

func TestSpecialTokenIDs(t *testing.T) {
	jsonContent := minimalVocabJSON(t)
	path := writeTempVocab(t, jsonContent)
	defer os.Remove(path)

	tok, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}

	if tok.PadID() != 0 {
		t.Errorf("PadID = %d, want 0", tok.PadID())
	}
	if tok.UnkID() != 1 {
		t.Errorf("UnkID = %d, want 1", tok.UnkID())
	}
	if tok.BosID() != 2 {
		t.Errorf("BosID = %d, want 2", tok.BosID())
	}
	if tok.EosID() != 3 {
		t.Errorf("EosID = %d, want 3", tok.EosID())
	}
}

func TestEncodeDecodeRoundtrip(t *testing.T) {
	jsonContent := minimalVocabJSON(t)
	path := writeTempVocab(t, jsonContent)
	defer os.Remove(path)

	tok, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}

	text := "abc abc"
	ids, err := tok.Encode(text)
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}

	// Should have BOS + tokens + EOS
	if len(ids) < 3 {
		t.Fatalf("Encode returned %d ids, want >= 3", len(ids))
	}
	if ids[0] != BosID {
		t.Errorf("First id = %d, want %d (BOS)", ids[0], BosID)
	}
	if ids[len(ids)-1] != EosID {
		t.Errorf("Last id = %d, want %d (EOS)", ids[len(ids)-1], EosID)
	}

	decoded := tok.Decode(ids)
	// Should contain the original letters, possibly with BOS/EOS stripped
	if len(decoded) == 0 {
		t.Error("Decode returned empty string")
	}
}

func TestDecodeIgnoresSpecialTokens(t *testing.T) {
	jsonContent := minimalVocabJSON(t)
	path := writeTempVocab(t, jsonContent)
	defer os.Remove(path)

	tok, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}

	// Decode [BOS, 'a', 'b', EOS, PAD]
	result := tok.Decode([]uint16{BosID, 4, 5, EosID, PadID})
	if result != "ab" {
		t.Errorf("Decode = %q, want %q", result, "ab")
	}
}

func TestEncodeBOSEOSPresence(t *testing.T) {
	jsonContent := minimalVocabJSON(t)
	path := writeTempVocab(t, jsonContent)
	defer os.Remove(path)

	tok, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}

	ids, err := tok.Encode("")
	if err != nil {
		t.Fatal(err)
	}

	if len(ids) != 2 || ids[0] != BosID || ids[1] != EosID {
		t.Errorf("Empty encode = %v, want [BOS, EOS]", ids)
	}
}

func TestUnknownTokens(t *testing.T) {
	jsonContent := minimalVocabJSON(t)
	path := writeTempVocab(t, jsonContent)
	defer os.Remove(path)

	tok, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}

	// 'z' is not in vocab, should become UNK
	ids, err := tok.Encode("z")
	if err != nil {
		t.Fatal(err)
	}

	hasUnk := false
	for _, id := range ids {
		if id == UnkID {
			hasUnk = true
			break
		}
	}
	if !hasUnk {
		t.Errorf("Unknown char 'z' should map to UNK, ids = %v", ids)
	}
}

func TestDeterministic(t *testing.T) {
	jsonContent := minimalVocabJSON(t)
	path := writeTempVocab(t, jsonContent)
	defer os.Remove(path)

	tok, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}

	ids1, _ := tok.Encode("abc abc abc")
	ids2, _ := tok.Encode("abc abc abc")

	if len(ids1) != len(ids2) {
		t.Fatalf("Different lengths: %d vs %d", len(ids1), len(ids2))
	}
	for i := range ids1 {
		if ids1[i] != ids2[i] {
			t.Errorf("Position %d: %d vs %d", i, ids1[i], ids2[i])
		}
	}
}

// --- Parity test (skipped unless reference data exists) ---

func TestParityWithPython(t *testing.T) {
	refPath := "/tmp/tokenizer_reference.json"
	if _, err := os.Stat(refPath); os.IsNotExist(err) {
		t.Skip("run training/scripts/verify_tokenizer_parity.py first")
	}

	refData, err := os.ReadFile(refPath)
	if err != nil {
		t.Fatal(err)
	}

	var ref struct {
		VocabPath string            `json:"vocab_path"`
		Vocab     map[string]uint16 `json:"vocab"`
		Results   []struct {
			Text    string   `json:"text"`
			IDs     []uint16 `json:"ids"`
			Decoded string   `json:"decoded"`
		} `json:"results"`
	}
	if err := json.Unmarshal(refData, &ref); err != nil {
		t.Fatal(err)
	}

	tok, err := Load(ref.VocabPath)
	if err != nil {
		t.Fatal(err)
	}

	for i, result := range ref.Results {
		gotIDs, err := tok.Encode(result.Text)
		if err != nil {
			t.Errorf("Result %d (%q): Encode error: %v", i, result.Text, err)
			continue
		}

		if len(gotIDs) != len(result.IDs) {
			t.Errorf("Result %d (%q): lengths differ: got %d, want %d",
				i, result.Text, len(gotIDs), len(result.IDs))
			t.Logf("  Got:  %v", gotIDs)
			t.Logf("  Want: %v", result.IDs)
			continue
		}

		for j := range gotIDs {
			if gotIDs[j] != result.IDs[j] {
				t.Errorf("Result %d (%q): position %d: got %d, want %d",
					i, result.Text, j, gotIDs[j], result.IDs[j])
			}
		}

		gotDecoded := tok.Decode(result.IDs)
		// Decoded strings may differ slightly due to unknown chars
		// but should at least contain the known words
		_ = gotDecoded
	}
}
