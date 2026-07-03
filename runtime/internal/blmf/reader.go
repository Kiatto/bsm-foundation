// Package blmf implements the BLMF binary format reader.
//
// BLMF is a section-based binary format for BSM models.
// Spec: see format/BLMF_SPEC.md
package blmf

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"

	"github.com/blm/runtime/internal/model"
)

// Magic bytes: "BLMF" + v1.0
var magic = []byte{0x42, 0x4C, 0x4D, 0x46, 0x00, 0x01, 0x00, 0x00}

// DType constants
const (
	DTypeU8  uint32 = 0
	DTypeU16 uint32 = 1
	DTypeI32 uint32 = 2
	DTypeF32 uint32 = 3
	DTypeU64 uint32 = 4
	DTypeStr uint32 = 5
	DTypeRaw uint32 = 0xFF
)

// Section holds raw section data from a BLMF file.
type Section struct {
	Name   string
	Data   []byte
	DType  uint32
	Shape  []uint32
}

// File holds parsed BLMF file contents.
type File struct {
	Header   map[string]any
	Sections map[string]Section
}

// LoadFile reads and parses a BLMF file, returning section data.
func LoadFile(path string) (*File, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read blmf file: %w", err)
	}

	if len(data) < 24 {
		return nil, fmt.Errorf("file too short: %d bytes", len(data))
	}

	// Verify magic
	for i, b := range magic[:4] {
		if data[i] != b {
			return nil, fmt.Errorf("invalid magic: expected BLMF, got %q", string(data[:4]))
		}
	}

	// Parse header
	version := binary.LittleEndian.Uint32(data[8:12])
	flags := binary.LittleEndian.Uint32(data[12:16])
	headerSize := binary.LittleEndian.Uint32(data[16:20])

	_ = version
	_ = flags

	if int(headerSize) > len(data) {
		return nil, fmt.Errorf("header size %d exceeds file size %d", headerSize, len(data))
	}

	// Extract JSON header (bytes 20 to headerSize, stop at first null)
	jsonBytes := data[20:headerSize]
	for i, b := range jsonBytes {
		if b == 0 {
			jsonBytes = jsonBytes[:i]
			break
		}
	}

	var header map[string]any
	if err := json.Unmarshal(jsonBytes, &header); err != nil {
		return nil, fmt.Errorf("parse header JSON: %w", err)
	}

	// Section table starts at headerSize
	stOffset := int(headerSize)
	if stOffset+4 > len(data) {
		return nil, fmt.Errorf("section table offset %d beyond file", stOffset)
	}

	numSections := binary.LittleEndian.Uint32(data[stOffset : stOffset+4])
	entryBase := stOffset + 4

	sections := make(map[string]Section)

	for i := uint32(0); i < numSections; i++ {
		entryOff := entryBase + int(i)*56

		if entryOff+56 > len(data) {
			return nil, fmt.Errorf("section entry %d at %d beyond file", i, entryOff)
		}

		nameOff := binary.LittleEndian.Uint32(data[entryOff : entryOff+4])
		nameLen := binary.LittleEndian.Uint32(data[entryOff+4 : entryOff+8])
		dataOff := binary.LittleEndian.Uint64(data[entryOff+8 : entryOff+16])
		dataSize := binary.LittleEndian.Uint64(data[entryOff+16 : entryOff+24])
		dtype := binary.LittleEndian.Uint32(data[entryOff+24 : entryOff+28])
		shapeRank := binary.LittleEndian.Uint32(data[entryOff+28 : entryOff+32])

		shape := make([]uint32, shapeRank)
		for j := uint32(0); j < shapeRank && j < 4; j++ {
			off := entryOff + 32 + int(j)*4
			shape[j] = binary.LittleEndian.Uint32(data[off : off+4])
		}

		// Read section name
		nameStart := int(nameOff) + 2 // skip 2-byte length prefix
		if nameStart+int(nameLen) > len(data) {
			return nil, fmt.Errorf("section name %d out of bounds", i)
		}
		name := string(data[nameStart : nameStart+int(nameLen)])

		// Read section data
		ds := int(dataOff)
		sz := int(dataSize)
		if ds+sz > len(data) {
			return nil, fmt.Errorf("section %q data out of bounds: offset %d size %d file %d",
				name, ds, sz, len(data))
		}
		secData := make([]byte, sz)
		copy(secData, data[ds:ds+sz])

		sections[name] = Section{
			Name:   name,
			Data:   secData,
			DType:  dtype,
			Shape:  shape,
		}
	}

	return &File{
		Header:   header,
		Sections: sections,
	}, nil
}

// LoadModel reconstructs a BSMModel from a BLMF file.
func LoadModel(path string) (*model.BSMModel, error) {
	f, err := LoadFile(path)
	if err != nil {
		return nil, err
	}

	// Parse config from header
	cfg, err := configFromHeader(f.Header)
	if err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	// Load embedding
	embSec, ok := f.Sections["embedding"]
	if !ok {
		return nil, fmt.Errorf("missing embedding section")
	}
	emb, err := model.EmbeddingFromBytes(embSec.Data, cfg.VocabSize, cfg.HiddenDim)
	if err != nil {
		return nil, fmt.Errorf("load embedding: %w", err)
	}

	// Load layers
	layers := make([]model.BSMLayerWeights, cfg.NumLayers)
	for li := 0; li < cfg.NumLayers; li++ {
		loadWeight := func(name string) (*model.BinaryWeight, error) {
			sec, ok := f.Sections[name]
			if !ok {
				return nil, fmt.Errorf("missing section %q", name)
			}
			return model.BinaryWeightFromBytes(sec.Data, cfg.HiddenDim, cfg.HiddenDim)
		}

		wf, err := loadWeight(fmt.Sprintf("layer_%d_wforget", li))
		if err != nil {
			return nil, err
		}
		wi, err := loadWeight(fmt.Sprintf("layer_%d_winput", li))
		if err != nil {
			return nil, err
		}
		wm, err := loadWeight(fmt.Sprintf("layer_%d_wmix", li))
		if err != nil {
			return nil, err
		}

		layers[li] = model.BSMLayerWeights{
			WForget: wf,
			WInput:  wi,
			WMix:    wm,
		}
	}

	// Load head weights
	headSec, ok := f.Sections["head_weight"]
	if !ok {
		return nil, fmt.Errorf("missing head_weight section")
	}
	expectedHeadBytes := cfg.VocabSize * cfg.HiddenDim * 4
	if len(headSec.Data) < expectedHeadBytes {
		return nil, fmt.Errorf("head_weight data too short: %d < %d", len(headSec.Data), expectedHeadBytes)
	}

	headW := make([]float32, cfg.VocabSize*cfg.HiddenDim)
	for i := 0; i < cfg.VocabSize*cfg.HiddenDim; i++ {
		headW[i] = float32FromBytes(headSec.Data[i*4 : i*4+4])
	}

	m := &model.BSMModel{
		Config: *cfg,
		Embed:  emb,
		Layers: layers,
		HeadW:  headW,
	}

	// Load optional tree head
	if treeSec, ok := f.Sections["tree_head"]; ok {
		tree, err := model.TreeFromBytes(treeSec.Data)
		if err != nil {
			return nil, fmt.Errorf("load tree_head: %w", err)
		}
		m.TreeHead = tree
	}

	return m, nil
}

// ValidateFile checks BLMF file integrity and returns stats.
func ValidateFile(path string) (map[string]any, error) {
	f, err := LoadFile(path)
	if err != nil {
		return nil, err
	}

	// Verify checksum
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	if len(data) < 8 {
		return nil, fmt.Errorf("file too short for checksum")
	}

	storedChecksum := binary.LittleEndian.Uint64(data[len(data)-8:])
	computed := checksum64(data[:len(data)-8])

	if storedChecksum != computed {
		return nil, fmt.Errorf("checksum mismatch: stored=0x%X computed=0x%X",
			storedChecksum, computed)
	}

	summary := make(map[string]any)
	summary["file_size"] = len(data)
	summary["num_sections"] = len(f.Sections)
	summary["checksum_ok"] = true

	sectionNames := make([]string, 0, len(f.Sections))
	for name := range f.Sections {
		sectionNames = append(sectionNames, name)
	}
	summary["sections"] = sectionNames

	if arch, ok := f.Header["arch"]; ok {
		summary["arch"] = arch
	}
	if vs, ok := f.Header["vocab_size"]; ok {
		summary["vocab_size"] = vs
	}
	if hd, ok := f.Header["hidden_dim"]; ok {
		summary["hidden_dim"] = hd
	}
	if nl, ok := f.Header["num_layers"]; ok {
		summary["num_layers"] = nl
	}

	return summary, nil
}

// --- Internal helpers ---

func configFromHeader(h map[string]any) (*model.BSMConfig, error) {
	getInt := func(key string) (int, error) {
		v, ok := h[key]
		if !ok {
			return 0, fmt.Errorf("missing header field %q", key)
		}
		switch val := v.(type) {
		case float64:
			return int(val), nil
		case int:
			return val, nil
		default:
			return 0, fmt.Errorf("field %q has unexpected type %T", key, v)
		}
	}

	vs, err := getInt("vocab_size")
	if err != nil {
		return nil, err
	}
	hd, err := getInt("hidden_dim")
	if err != nil {
		return nil, err
	}
	nl, err := getInt("num_layers")
	if err != nil {
		return nil, err
	}
	ws, err := getInt("window_size")
	if err != nil {
		return nil, err
	}
	sl, err := getInt("seq_len")
	if err != nil {
		return nil, err
	}

	cfg := &model.BSMConfig{
		VocabSize:  vs,
		HiddenDim:  hd,
		NumLayers:  nl,
		WindowSize: ws,
		MaxSeqLen:  sl,
	}

	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	return cfg, nil
}

func float32FromBytes(b []byte) float32 {
	return math.Float32frombits(binary.LittleEndian.Uint32(b))
}


func checksum64(data []byte) uint64 {
	h := sha256.Sum256(data)
	return binary.LittleEndian.Uint64(h[:8])
}

// ListSections returns the names of all sections in a BLMF file.
func ListSections(path string) ([]string, error) {
	f, err := LoadFile(path)
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(f.Sections))
	for name := range f.Sections {
		names = append(names, name)
	}
	return names, nil
}

