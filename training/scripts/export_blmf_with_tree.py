#!/usr/bin/env python3
"""Export checkpoint + trained tree head to BLMF with tree_head section."""

import argparse
import json
import struct
import sys
import os
import hashlib
import io
import torch
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from blm.model import BSMModel, BSMConfig
from blm.export import export_to_blmf, _raw_section, checksum64

DTYPE_RAW = 0xFF
MAGIC = b"BLMF\x00\x01\x00\x00"
VERSION = 1
FLAG_HAS_TOKENIZER = 2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tree-head", required=True, help=".bin file from train_tree_head.py")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default="model.blmf")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if isinstance(config, dict):
        config = BSMConfig(**config)
    elif not isinstance(config, BSMConfig):
        print("Error: unknown config type", file=sys.stderr)
        sys.exit(1)

    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config_data = json.load(f)
            config = BSMConfig(**config_data)

    from blm.tokenizer import BPETokenizer
    tokenizer = BPETokenizer.load(args.tokenizer)
    model = BSMModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load trained tree head bytes
    with open(args.tree_head, 'rb') as f:
        trained_tree_bytes = f.read()
    
    hdr = struct.unpack_from('<IIII', trained_tree_bytes, 0)
    print(f"Trained tree: numNodes={hdr[0]}, nw={hdr[1]}, vocabSize={hdr[2]}, leafBase={hdr[3]}")

    # Export to BLMF, replacing the default tree_head section with trained one
    export_to_blmf_with_tree(model, tokenizer, args.output, trained_tree_bytes)
    print(f"Exported to {args.output}")

def export_to_blmf_with_tree(model, tokenizer, output_path, tree_bytes):
    """Export model + trained tree head to BLMF."""
    from blm.export import _uint8_section, _float32_section, _string_section
    from blm.export import binarize_weight, pack_bits

    model.eval()
    cfg = model.config

    sections = []

    emb_bits = model.embedding.export_binary()
    sections.append(_uint8_section(emb_bits.cpu(), "embedding"))

    for li, layer in enumerate(model.layers):
        wf = binarize_weight(layer.state_update.W_forget.weight)
        sections.append(_uint8_section(pack_bits(wf).cpu(), f"layer_{li}_wforget"))
        wi = binarize_weight(layer.state_update.W_input.weight)
        sections.append(_uint8_section(pack_bits(wi).cpu(), f"layer_{li}_winput"))
        wm = binarize_weight(layer.mixer.W_mix.weight)
        sections.append(_uint8_section(pack_bits(wm).cpu(), f"layer_{li}_wmix"))

    head_w = model.head.weight.data.cpu()
    sections.append(_float32_section(head_w, "head_weight"))

    sections.append(_raw_section(tree_bytes, "tree_head"))

    vocab_json = json.dumps({
        "vocab_size": tokenizer._vocab_size,
        "vocab": dict(tokenizer.vocab),
        "merges": [[p[0], p[1]] for p in tokenizer.merges],
    }, ensure_ascii=False)
    sections.append(_string_section(vocab_json, "vocab"))

    header_meta = {
        "arch": "BSM",
        "vocab_size": cfg.vocab_size,
        "hidden_dim": cfg.hidden_dim,
        "num_layers": cfg.num_layers,
        "window_size": cfg.window_size,
        "seq_len": cfg.seq_len,
    }

    # Write BLMF file
    buf = io.BytesIO()
    header_json = json.dumps(header_meta, indent=2)
    hdr_json_bytes = header_json.encode('utf-8')
    json_start = 20
    json_end = json_start + len(hdr_json_bytes)
    pad_to_512 = (512 - (json_end % 512)) % 512
    total_header_size = json_end + pad_to_512

    buf.write(MAGIC)
    buf.write(struct.pack('<I', VERSION))
    buf.write(struct.pack('<I', FLAG_HAS_TOKENIZER))
    buf.write(struct.pack('<I', total_header_size))
    buf.write(hdr_json_bytes)
    buf.write(b'\x00' * pad_to_512)

    section_table_offset = buf.tell()
    num_sections = len(sections)
    buf.write(struct.pack('<I', num_sections))
    section_entry_size = 56
    section_table_header_size = 4 + num_sections * section_entry_size
    buf.write(b'\x00' * section_table_header_size)

    name_offsets = []
    for s in sections:
        name_bytes = s.name.encode('utf-8')
        name_len = len(name_bytes)
        name_offsets.append(buf.tell() - section_table_offset)
        buf.write(struct.pack('<H', name_len))
        buf.write(name_bytes)

    data_start = buf.tell()
    padding = (8 - (data_start % 8)) % 8
    buf.write(b'\x00' * padding)
    data_start = buf.tell()

    section_offsets = []
    offset = data_start
    for s in sections:
        section_offsets.append(offset)
        buf.write(s.data)
        offset += len(s.data)
        align = (8 - (len(s.data) % 8)) % 8
        buf.write(b'\x00' * align)
        offset += align

    buf.seek(section_table_offset + 4)
    for i, s in enumerate(sections):
        name_bytes = s.name.encode('utf-8')
        shape_vals = list(s.shape[:4]) + [0] * (4 - len(s.shape))
        name_abs_offset = name_offsets[i] + section_table_offset
        entry = struct.pack(
            '<IIQQIIIIIIII',
            name_abs_offset, len(name_bytes),
            section_offsets[i], len(s.data),
            s.dtype, len(s.shape),
            shape_vals[0], shape_vals[1], shape_vals[2], shape_vals[3],
            0, 0,
        )
        buf.write(entry)

    buf.seek(0)
    all_data = buf.read()
    checksum = checksum64(all_data)
    buf.write(struct.pack('<Q', checksum))

    with open(output_path, 'wb') as f:
        buf.seek(0)
        f.write(buf.read())
    return output_path

if __name__ == "__main__":
    main()
