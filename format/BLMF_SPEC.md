# BLMF — Binary Language Model Format Specification v1

## 1. Overview

BLMF is a binary container format designed exclusively for binary neural network models.
It supports memory-mapped loading, section-based organization, and integrity verification.

Key design decisions:
- **Section-based**: each type of data (embeddings, weights, metadata) lives in its own section
- **mmap-friendly**: all offsets are absolute from file start; no pointer fixup needed
- **Self-contained**: includes tokenizer vocabulary for standalone inference
- **Checksummed**: xxHash64 integrity verification

## 2. File Layout

```
┌─────────────────────────────────────────────────┐
│ Magic Header      (8 bytes)                      │
├─────────────────────────────────────────────────┤
│ Version           (4 bytes)                      │
├─────────────────────────────────────────────────┤
│ Flags             (4 bytes)                      │
├─────────────────────────────────────────────────┤
│ HeaderSize        (4 bytes)                      │
├─────────────────────────────────────────────────┤
│ HeaderJSON        (variable, UTF-8)              │
├─────────────────────────────────────────────────┤
│ Padding           (to 512-byte boundary)         │
├─────────────────────────────────────────────────┤
│ SectionTable      (variable)                     │
├─────────────────────────────────────────────────┤
│ Section Data      (variable)                     │
├─────────────────────────────────────────────────┤
│ Padding           (to 8-byte alignment)          │
├─────────────────────────────────────────────────┤
│ Checksum          (8 bytes, xxHash64)            │
└─────────────────────────────────────────────────┘
```

## 3. Field Specifications

### 3.1 Magic Header (8 bytes)

```
Offset: 0
Value:  0x00 0x00 0x01 0x00 0x46 0x4D 0x4C 0x42

Interpretation: "BLMF\x00\x01\x00\x00"
  - Bytes 0-3: "BLMF" (0x42 0x4C 0x4D 0x46)
  - Bytes 4-5: major version (0x0001)
  - Bytes 6-7: minor version (0x0000)
```

### 3.2 Version (4 bytes, uint32 LE)

```
Currently: 1
```

### 3.3 Flags (4 bytes, uint32 LE, bitmask)

```
Bit 0: compressed (1 = sections are zstd-compressed)
Bit 1: has_tokenizer (1 = tokenizer section present)
Bits 2-31: reserved, must be 0
```

### 3.4 HeaderSize (4 bytes, uint32 LE)

Total size of the header region (Magic + Version + Flags + HeaderSize + HeaderJSON + padding).

### 3.5 HeaderJSON (variable, UTF-8)

JSON object containing model metadata:

```json
{
  "arch": "BSM",
  "vocab_size": 4096,
  "hidden_dim": 256,
  "num_layers": 4,
  "window_size": 8,
  "seq_len": 128,
  "created_at": "2026-06-26T12:00:00Z",
  "training_steps": 5000,
  "final_loss": 1.234,
  "description": "BLM BSM model trained on TinyStories"
}
```

Required fields: `arch`, `vocab_size`, `hidden_dim`, `num_layers`, `window_size`, `seq_len`
Optional fields: `created_at`, `training_steps`, `final_loss`, `description`

### 3.6 Padding (to 512-byte boundary)

Zero-filled padding after HeaderJSON.
Ensures the file header fits in a single sector for efficient I/O.

Calculation: `pad = (512 - (header_start + header_size) % 512) % 512`

### 3.7 SectionTable (variable)

Section table format:

```
┌─────────────────────────────────────────────────┐
│ NumSections       (4 bytes, uint32 LE)           │
├─────────────────────────────────────────────────┤
│ Section Entry 0   (48 bytes)                     │
│ Section Entry 1   (48 bytes)                     │
│ ...                                              │
└─────────────────────────────────────────────────┘
```

Each section entry (48 bytes):

```
┌─────────────────────────────────────────────────┐
│ NameOffset       (4 bytes, uint32 LE)            │  offset of section name string
│ NameLen          (4 bytes, uint32 LE)            │  length of name (max 64)
│ DataOffset       (8 bytes, uint64 LE)            │  absolute file offset of data
│ DataSize         (8 bytes, uint64 LE)            │  size in bytes
│ DType            (4 bytes, uint32 LE)            │  data type enum
│ ShapeRank        (4 bytes, uint32 LE)            │  number of shape dimensions
│ Shape            (16 bytes, 4 x uint32 LE)       │  shape dimensions [0,0,0,0] for unused
│ Flags            (4 bytes, uint32 LE)            │  section-specific flags
│ Padding          (4 bytes)                       │  zero-filled
└─────────────────────────────────────────────────┘
```

DType values:
- 0 = uint8 (binary packed)
- 1 = uint16
- 2 = int32
- 3 = float32
- 4 = uint64
- 5 = string (UTF-8)
- 0xFF = raw bytes

Section name strings follow the section table:
```
StringBlock:
  [NameLenBytes][UTF-8 Name]...
  Each entry: 2 bytes name length + UTF-8 name bytes
```

### 3.8 Section Data

For each section, data starts at `DataOffset` and spans `DataSize` bytes.

Standard sections:
- `"embedding"` — uint8 packed, shape [vocab_size, hidden_dim/8]
- `"layer_N_wforget"` — uint8 packed, shape [hidden_dim, hidden_dim/8]
- `"layer_N_winput"` — uint8 packed, shape [hidden_dim, hidden_dim/8]
- `"layer_N_wmix"` — uint8 packed, shape [hidden_dim, hidden_dim/8]
- `"head_weight"` — float32, shape [vocab_size, hidden_dim]
- `"vocab"` — JSON string, tokenizer vocabulary
- `"merges"` — raw bytes, BPE merges

### 3.9 Checksum (8 bytes, uint64 LE)

xxHash64 of the entire file except the last 8 bytes (the checksum itself).

## 4. Binary Packing

All binary weights are stored as packed uint8 arrays (8 bits per byte, LSB first).

For a weight matrix with logical shape [M, N]:
- Physical shape: [M, ceil(N/8)] bytes
- Packing order: row-major
- Bit `i` of byte `j` corresponds to weight index `j*8 + i`

Example: M=1, N=10
- 2 bytes: [b7 b6 b5 b4 b3 b2 b1 b0] [0 0 0 0 0 b9 b8 b7] (LSB = bit 0)

## 5. Example Hex Dump (annotated)

```
Offset  Bytes              Description
────────────────────────────────────────────────────
0x0000  42 4C 4D 46        Magic "BLMF"
0x0004  00 01 00 00        Version 1
0x0008  00 00 00 00        Flags
0x000C  00 01 00 00        HeaderSize = 256
0x0010  7B 22 61 72 63...  HeaderJSON: {"arch":"BSM",...}
0x0100  ...                Padding zeros
0x0200  ...                SectionTable
```

## 6. Versioning

- Backward compatible: readers must accept files with any minor version
- Breaking changes: increment major version
- Unknown sections: must be ignored (not errored)

## 7. Implementation Notes

- All multi-byte values are little-endian
- Padding bytes must be zero; readers should NOT depend on specific values
- Section names are case-sensitive
- A valid file must have at minimum: `embedding`, `head_weight`, `vocab` sections
- Maximum section name length: 64 bytes
- Maximum number of sections: 1024
