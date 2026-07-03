# BLMF — Binary Language Model Format

## 1. Specifica v1

```
Binary Language Model Format v1

Tutto little-endian.

+----------------------------------------------------------------+
| HEADER (64 byte)                                                 |
+------+------+-----------------------+--------------------------+
| Off  | Size | Field                 | Descrizione              |
+------+------+-----------------------+--------------------------+
| 0    | 4    | magic                 | "BLMF" (0x464D4C42)     |
| 4    | 4    | version               | 1                        |
| 8    | 2    | arch_id               | 0x0001 = BCM-2048       |
| 10   | 2    | reserved              | padding                  |
| 12   | 4    | config_size           | byte del JSON config     |
| 16   | 4    | vocab_size            | V                        |
| 20   | 4    | dimension             | D                        |
| 24   | 4    | num_layers            | N                        |
| 28   | 4    | context_len           | L                        |
| 32   | 4    | d_hidden              | D_hidden                 |
| 36   | 28   | reserved              | zero padding             |
+------+------+-----------------------+--------------------------+

+----------------------------------------------------------------+
| CONFIG (config_size byte) — JSON UTF-8                          |
+----------------------------------------------------------------+
| Esempio: {"tokenizer":"bpe","merge_ops":4096,"name":"blm-v1"}  |
+----------------------------------------------------------------+

+----------------------------------------------------------------+
| EMBED TABLE                                                     |
+------+------+-----------------------+--------------------------+
| Off  | Size | Field                 | Descrizione              |
+------+------+-----------------------+--------------------------+
| 0    | 4    | embed_bit_count       | D bit per embedding      |
| 4    | 4    | embed_byte_size       | D/8 bytes per embedding  |
| 8    | V*(D/8) | embed_data        | packed binary vectors    |
+------+------+-----------------------+--------------------------+

+----------------------------------------------------------------+
| LAYERS (N layers, sequenziali)                                  |
+------+------+-----------------------+--------------------------+
| Layer header (32 byte per layer):                               |
| 0    | 1    | layer_type            | 0x01=BinMLP, 0x02=Conv  |
| 1    | 3    | reserved              | padding                  |
| 4    | 4    | in_dim                | D input dimension        |
| 8    | 4    | out_dim               | D output dimension       |
| 12   | 4    | weight_bytes          | size of weight data      |
| 16   | 4    | threshold_bytes       | size of threshold data   |
| 20   | 12   | reserved              | padding                  |
|      |      |                       |                          |
| Layer data (dopo header):                                       |
|      | weight_bytes | weights      | packed binary weights    |
|      | threshold_bytes | thresholds| packed thresholds        |
+------+------+-----------------------+--------------------------+

+----------------------------------------------------------------+
| TOKEN TABLE (vocabolario)                                       |
+------+------+-----------------------+--------------------------+
| Per ogni token:                                                 |
| 0    | 2    | token_len             | byte del token UTF-8    |
| 2    | var  | token_text            | UTF-8 encoded text      |
+------+------+-----------------------+--------------------------+

+----------------------------------------------------------------+
| FOOTER (16 byte)                                                |
+------+------+-----------------------+--------------------------+
| 0    | 4    | checksum              | CRC32 of file - footer  |
| 4    | 4    | magic_end             | "FM LB" (0x424C4D46)    |
| 8    | 8    | file_size             | total file size         |
+------+------+-----------------------+--------------------------+
```

## 2. Struttura Pesi in Memoria

```
Layer BinMLP:
  W1: D_hidden x D bit
    packed: D_hidden * ceil(D/8) byte
    access: W1[neuron * D/8 + byte_offset]

  t1: D_hidden bit
    packed: ceil(D_hidden/8) byte
    access: (t1[byte_offset] >> bit) & 1

  W2: D x D_hidden bit
    packed: D * ceil(D_hidden/8) byte

  t2: D bit
    packed: ceil(D/8) byte

  Wg: D x D bit
    packed: D * ceil(D/8) byte

  tg: D bit
    packed: ceil(D/8) byte

Layer Conv1D:
  W_conv: K x D bit
    packed: K * ceil(D/8) byte

  t_conv: D bit
    packed: ceil(D/8) byte
```

## 3. Memory Map per Load Zero-copy

```
File BLMF:

+----------+
| Header   | 64 byte
+----------+
| Config   | JSON
+----------+
| Embed    | V*(D/8) byte
+----------+
| Layer 0  | header + data
+----------+
| Layer 1  |
+----------+
| ...      |
+----------+
| Tokens   | token table
+----------+
| Footer   | 16 byte
+----------+

Caricamento: mmap del file intero.
Puntatori diretti alle regioni di memoria.
Nessuna copia dei pesi binari.

embed_data = mmap_base + header_size
layer_W1 = layer_base + layer_header_size
```

## 4. Implementazione Go — Lettura

```go
type Header struct {
    Magic       uint32
    Version     uint32
    ArchID      uint16
    _           [2]byte
    ConfigSize  uint32
    VocabSize   uint32
    Dimension   uint32
    NumLayers   uint32
    ContextLen  uint32
    DHidden     uint32
    _           [28]byte
}

type BLMFile struct {
    data   []byte  // mmap'd data
    header Header
    layers []LayerReader
}

func Load(path string) (*BLMFile, error) {
    data, err := mmap(path)
    // ...
    h := (*Header)(unsafe.Pointer(&data[0]))
    // verify magic, version, checksum
    // return reader with pointers into data
}

func (f *BLMFile) Embeddings() []uint64 {
    off := 64 + f.header.ConfigSize
    count := int(f.header.VocabSize * f.header.Dimension / 64)
    return unsafe.Slice((*uint64)(unsafe.Pointer(&f.data[off])), count)
}
```

## 5. Checksum e Validazione

```
CRC32 dell'intero file prima del footer (byte 0 fino a -16).
Il footer contiene il CRC32 atteso.

Alla validazione:
  - Verificare magic "BLMF"
  - Verificare magic_end "FM LB"
  - Verificare CRC32 match
  - Verificare version == 1
  - Verificare arch_id noto
  - Verificare config JSON parsabile
```
