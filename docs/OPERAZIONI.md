# BLM — Binary Operations

## 1. Core Operations

Tutte definite su `BitVector = []uint64` con `N = D/64`.

```
Dimensione: D = 2048 bit = 32 x uint64 = 256 byte

Operazione   Input      Output     Istruzione      Costo
----------   -----      ------     ----------      ----
POPCOUNT     uint64     int(0..64)  POPCNT (x86)    3-5 cicli
                                    VCNT (ARM)
XOR          a, b       uint64     XOR              1 ciclo
XNOR         a, b       uint64     NOT(XOR)         1 ciclo
AND          a, b       uint64     AND              1 ciclo
OR           a, b       uint64     OR               1 ciclo
NOT          a          uint64     NOT              1 ciclo
SHIFT        a, n       uint64     SHL/SHR          1 ciclo
COMPARE      v, t       bit        CMP              1 ciclo
```

## 2. Binary Linear Layer — Kernel C

```c
// Processa 64 neuroni alla volta (un uint64 di output)
// Input:  x[D_words]
// Weight: W[M][D_words]
// Output: out[M/64]

void binary_linear(
    uint64_t* x,
    uint64_t* W,
    uint64_t* t,
    uint64_t* out,
    int M,
    int D_words
) {
    for (int i = 0; i < M/64; i++) {
        uint64_t result = 0;

        for (int bit = 0; bit < 64; bit++) {
            int total = 0;
            int ni = i * 64 + bit;

            for (int c = 0; c < D_words; c++) {
                uint64_t match = ~(x[c] ^ W[ni * D_words + c]);
                total += __builtin_popcountll(match);
            }

            // threshold: 1 = D/2, 0 = D/4 (esempio)
            uint64_t thr_bit = (t[i] >> bit) & 1;
            int thr = thr_bit ? D/2 : D/4;
            if (total > thr) {
                result |= (1ULL << bit);
            }
        }

        out[i] = result;
    }
}
```

## 3. Binary Linear Layer — Go

```go
func BinaryLinear(x, w, t, out []uint64, m, dw int) {
    for i := 0; i < m/64; i++ {
        var result uint64
        for bit := 0; bit < 64; bit++ {
            total := 0
            ni := i*64 + bit
            base := ni * dw

            for c := 0; c < dw; c++ {
                match := ^(x[c] ^ w[base+c])
                total += bits.OnesCount64(match)
            }

            thr := (t[i] >> bit) & 1
            threshold := D / 2
            if thr == 1 && total > threshold {
                result |= 1 << bit
            } else if thr == 0 && total > threshold/2 {
                result |= 1 << bit
            }
        }
        out[i] = result
    }
}
```

## 4. Binary Matrix Multiply (Bit-packed)

Per D=2048:
```
W: [M x 32] uint64   -> M * 32 * 8 = M * 256 bytes
x: [32] uint64       -> 256 bytes
out: [M/64] uint64   -> M/8 bytes

Pattern di accesso:
  - x sta in L1 cache (256 byte)
  - W streama dal main memory
  - Bottleneck: bandwidth, non compute
```

## 5. POPCOUNT su ARM NEON

```c
// ARM NEON: 128 bit = 2 x uint64 per iterazione
// Processa 2 parole alla volta

uint64x2_t v = veorq_u64(a, b);          // XOR
uint8x16_t vcnt = vcntq_u8(vreinterpretq_u8_u64(v));  // POPCOUNT per byte
uint8x16_t sum = vpaddlq_u8(vcnt);       // riduzione
```

## 6. POPCOUNT su x86 AVX-512

```c
// VPOPCNTQ (AVX-512 VPOPCNTDQ): processa 8 uint64 in 1 ciclo
// Solo su Skylake-X, Ice Lake+, Zen4+

__m512i v = _mm512_xor_si512(a, b);           // XOR 512 bit
__m512i pop = _mm512_popcnt_epi64(v);          // 8 popcount
__m256i lo = _mm512_extracti64x4_epi64(pop, 0);
__m256i hi = _mm512_extracti64x4_epi64(pop, 1);
__m256i sum = _mm256_add_epi64(lo, hi);
```

## 7. Costi Computazionali

```
Forward completo per token (D=2048, D_hidden=1024):

+----------------------+--------------------+
| Operazione           | Costo stimato      |
+----------------------+--------------------+
| Embed lookup         | 1 load (256 byte)  |
| CellularBlock x 6    | ~30K-50K POPCOUNT  |
|   - BinMLP gate      |  ~2048*32 ops      |
|   - BinMLP hidden    |  ~1024*32 ops      |
|   - BinMLP output    |  ~2048*16 ops      |
| Output head (V=4096) | ~4096*32 POPCOUNT  |
| Total                | ~50K-130K POPCOUNT |
+----------------------+--------------------+

Throughput stimato:
  -50K-130K POPCOUNT @ 3-5 cicli cad. = 150K-650K cicli
  - CPU 2 GHz: ~75-325 us per token
  - Throughput: ~3000-13000 token/sec
```

## 8. Memory Footprint

```
+-----------------+-------------+---------------------+
| Componente      | Dimensione  | Note                |
+-----------------+-------------+---------------------+
| Embed Table     | V*D/8       | 4096*256 = 1MB      |
| BinMLP weights  | N*2*D*D_h   | 6*2*2048*1024 = ~24Mbit = 3MB |
| Thresholds      | N*D         | 6*2048 bit = ~12Kbit |
| States buffer   | L*D/8       | 256*256 = 64KB      |
| Totale modello  | ~4-5 MB     |                     |
+-----------------+-------------+---------------------+
```

## 9. Ottimizzazioni Chiave

1. **Loop unrolling**: Srotolare il loop su D_words (fisso a 32 per D=2048)
2. **Cache blocking**: W e' ~3MB, non sta in L2 (tipico 512KB-1MB). 
   Dividere M in blocchi da 512 neuroni per sfruttare L2.
3. **SIMD**: Usare AVX2/NEON per XOR in vettori larghi
4. **Branchless**: Evitare if nel loop interno (POPCOUNT e' branchless)
5. **Prefetch**: Prefetchare il prossimo blocco di W mentre si elabora il corrente
