package blmf

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func generateTestBLMF(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	outPath := filepath.Join(dir, "test.blmf")

	cmd := exec.Command("python3", "-c", `
import sys
sys.path.insert(0, "training")
from blm.model import BSMModel, BSMConfig
from blm.tokenizer import BPETokenizer
from blm.export import export_to_blmf

cfg = BSMConfig(vocab_size=64, hidden_dim=64, num_layers=1, window_size=2, seq_len=8)
model = BSMModel(cfg)
tok = BPETokenizer(vocab_size=64)
tok.train("the cat sat on the mat and the dog ran in the park hello world test")
export_to_blmf(model, tok, "`+outPath+`")
print("OK")
	`)

	// Run from project root
	cmd.Dir = "/var/www/html/BitKore"
	cmd.Stderr = os.Stderr
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("python export failed: %v\n%s", err, out)
	}

	return outPath
}

func TestLoadFile(t *testing.T) {
	path := generateTestBLMF(t)
	f, err := LoadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	if f.Header == nil {
		t.Error("expected non-nil header")
	}

	required := []string{"embedding", "layer_0_wforget", "layer_0_winput", "layer_0_wmix", "head_weight", "vocab"}
	for _, name := range required {
		if _, ok := f.Sections[name]; !ok {
			t.Errorf("missing section %q", name)
		}
	}

	if f.Header["arch"] != "BSM" {
		t.Errorf("arch = %v, want BSM", f.Header["arch"])
	}
	if v, ok := f.Header["vocab_size"].(float64); !ok || int(v) != 64 {
		t.Errorf("vocab_size = %v, want 64", f.Header["vocab_size"])
	}
}

func TestValidateFile(t *testing.T) {
	path := generateTestBLMF(t)
	info, err := ValidateFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if info["checksum_ok"] != true {
		t.Error("checksum should be valid")
	}
	if info["arch"] != "BSM" {
		t.Errorf("arch = %v, want BSM", info["arch"])
	}
}

func TestListSections(t *testing.T) {
	path := generateTestBLMF(t)
	names, err := ListSections(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(names) < 6 {
		t.Errorf("expected >=6 sections, got %d: %v", len(names), names)
	}
}

func TestLoadModel(t *testing.T) {
	path := generateTestBLMF(t)
	m, err := LoadModel(path)
	if err != nil {
		t.Fatal(err)
	}

	if m.Config.VocabSize != 64 {
		t.Errorf("VocabSize = %d, want 64", m.Config.VocabSize)
	}
	if m.Config.HiddenDim != 64 {
		t.Errorf("HiddenDim = %d, want 64", m.Config.HiddenDim)
	}
	if m.Config.NumLayers != 1 {
		t.Errorf("NumLayers = %d, want 1", m.Config.NumLayers)
	}

	if m.Embed == nil {
		t.Error("Embed is nil")
	}
	if len(m.Layers) != 1 {
		t.Errorf("Layers = %d, want 1", len(m.Layers))
	}
	if m.HeadW == nil {
		t.Error("HeadW is nil")
	}
}

func TestCorruptFile(t *testing.T) {
	dir := t.TempDir()
	badPath := filepath.Join(dir, "bad.blmf")

	if err := os.WriteFile(badPath, []byte("not a blmf file"), 0644); err != nil {
		t.Fatal(err)
	}

	_, err := LoadFile(badPath)
	if err == nil {
		t.Error("expected error for invalid file")
	}
}

func TestEmptyFile(t *testing.T) {
	dir := t.TempDir()
	emptyPath := filepath.Join(dir, "empty.blmf")

	if err := os.WriteFile(emptyPath, nil, 0644); err != nil {
		t.Fatal(err)
	}

	_, err := LoadFile(emptyPath)
	if err == nil {
		t.Error("expected error for empty file")
	}
}

func TestShortFile(t *testing.T) {
	dir := t.TempDir()
	shortPath := filepath.Join(dir, "short.blmf")

	if err := os.WriteFile(shortPath, []byte("BLMF"), 0644); err != nil {
		t.Fatal(err)
	}

	_, err := LoadFile(shortPath)
	if err == nil {
		t.Error("expected error for short file")
	}
}
