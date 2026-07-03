"""
Build a binary tree head from a trained BSM model's FP32 output head.

Usage:
    python3 -m training.scripts.build_tree_head \
        --checkpoint checkpoint.pt \
        --output model_with_tree.blmf

Reads the FP32 head weights from the checkpoint, constructs a complete
binary tree over the vocabulary, and saves the tree as a new BLMF section.
"""

import argparse
import json
import struct
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np


def next_pow2(n):
    """Return the smallest power of 2 >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


def build_tree_from_head(head_weight: torch.Tensor) -> tuple:
    """
    Build a binary tree from FP32 head weights.

    Args:
        head_weight: [vocab_size, hidden_dim] FP32 tensor

    Returns:
        tree_nodes: list of packed binary weight vectors (list of bytes)
        (num_nodes, nw, vocab_size, leaf_base) metadata
    """
    vocab_size, hidden_dim = head_weight.shape
    leaf_count = next_pow2(vocab_size)
    num_nodes = 2 * leaf_count - 1
    leaf_base = leaf_count - 1

    nw = (hidden_dim + 63) // 64
    head_np = head_weight.float().cpu().numpy()

    # Precompute bit-mask split per level (token ID bit)
    token_sets = {}
    for t in range(vocab_size):
        bit = t & 1
        token_sets.setdefault((0, bit), []).append(t)

    # Nodes stored as list of packed uint64 bytes
    nodes = [None] * num_nodes

    def build_node(node_idx, lo, hi):
        if node_idx >= num_nodes or hi - lo <= 1:
            return

        mid = (lo + hi) // 2
        left_lo, left_hi = lo, min(mid, vocab_size)
        right_lo, right_hi = mid, min(hi, vocab_size)

        if left_lo >= left_hi or right_lo >= right_hi:
            build_node(2 * node_idx + 1, lo, mid)
            build_node(2 * node_idx + 2, mid, hi)
            return

        centroid = np.zeros(hidden_dim, dtype=np.float32)
        centroid += head_np[right_lo:right_hi].mean(axis=0)
        centroid -= head_np[left_lo:left_hi].mean(axis=0)

        node_bits = bytearray(nw * 8)
        for j in range(hidden_dim):
            if centroid[j] > 0:
                word = j // 64
                bit = j % 64
                struct.pack_into('<Q', node_bits, word * 8,
                                 struct.unpack_from('<Q', node_bits, word * 8)[0] | (1 << bit))
        nodes[node_idx] = bytes(node_bits)

        build_node(2 * node_idx + 1, lo, mid)
        build_node(2 * node_idx + 2, mid, hi)

    build_node(0, 0, leaf_count)

    return nodes, num_nodes, nw, vocab_size, leaf_base


def serialize_tree(nodes, nw):
    """Serialize tree nodes into a byte string."""
    node_size = nw * 8
    data = bytearray()
    for node in nodes:
        if node is None:
            data.extend(b'\x00' * node_size)
        else:
            data.extend(node)
    return bytes(data)


def export_tree_blmf(input_path: str, output_path: str):
    """Read a BLMF file, build a tree head, and write it back."""
    # Read the existing BLMF file
    with open(input_path, 'rb') as f:
        data = bytearray(f.read())

    if len(data) < 24:
        print("Error: file too small")
        sys.exit(1)

    # Parse header to find head_weight section
    magic = data[8:12]
    version = int.from_bytes(data[8:12], 'little')
    flags = int.from_bytes(data[12:16], 'little')
    header_size = int.from_bytes(data[16:20], 'little')

    json_bytes = data[20:header_size]
    null_pos = json_bytes.find(b'\0')
    if null_pos >= 0:
        json_bytes = json_bytes[:null_pos]
    header = json.loads(json_bytes.decode('utf-8'))

    vocab_size = int(header.get('vocab_size', 0))
    hidden_dim = int(header.get('hidden_dim', 0))

    if vocab_size == 0 or hidden_dim == 0:
        print("Error: could not determine vocab_size or hidden_dim from header")
        sys.exit(1)

    # Parse sections to find head_weight
    st_offset = header_size
    num_sections = int.from_bytes(data[st_offset:st_offset + 4], 'little')
    entry_base = st_offset + 4

    head_weight_bytes = None
    head_name_offset = None

    for i in range(num_sections):
        entry_off = entry_base + i * 56
        if entry_off + 56 > len(data):
            break

        name_off = int.from_bytes(data[entry_off:entry_off + 4], 'little')
        name_len = int.from_bytes(data[entry_off + 4:entry_off + 8], 'little')
        data_off = int.from_bytes(data[entry_off + 8:entry_off + 16], 'little')
        data_size = int.from_bytes(data[entry_off + 16:entry_off + 24], 'little')
        dtype = int.from_bytes(data[entry_off + 24:entry_off + 28], 'little')
        shape_rank = int.from_bytes(data[entry_off + 28:entry_off + 32], 'little')

        name_start = int(name_off) + 2
        if name_start + int(name_len) <= len(data):
            name = data[name_start:name_start + int(name_len)].decode('utf-8', errors='replace')
            if name == 'head_weight':
                ds = int(data_off)
                sz = int(data_size)
                if ds + sz <= len(data):
                    head_weight_bytes = data[ds:ds + sz]

    if head_weight_bytes is None:
        print("Error: could not find head_weight section")
        sys.exit(1)

    expected_bytes = vocab_size * hidden_dim * 4
    if len(head_weight_bytes) < expected_bytes:
        print(f"Error: head_weight too small: {len(head_weight_bytes)} < {expected_bytes}")
        sys.exit(1)

    # Convert to torch tensor
    head_flat = np.frombuffer(head_weight_bytes[:expected_bytes], dtype=np.float32)
    head_w = torch.from_numpy(head_flat.reshape(vocab_size, hidden_dim))

    print(f"Building binary tree head: V={vocab_size} D={hidden_dim}")
    nodes, num_nodes, nw, vs, leaf_base = build_tree_from_head(head_w)
    print(f"  Tree: {num_nodes} nodes, {nw} words/node, {num_nodes * nw * 8} bytes")

    tree_bytes = serialize_tree(nodes, nw)

    # Build tree section header: numNodes(4) + nw(4) + vocabSize(4) + leafBase(4)
    tree_section_data = bytearray()
    tree_section_data.extend(struct.pack('<IIII', num_nodes, nw, vs, leaf_base))
    tree_section_data.extend(tree_bytes)

    # Write output BLMF with tree_head section appended
    with open(input_path, 'rb') as f:
        original = f.read()

    # We need to append the new section to the file.
    # First, update the section table, then append the data.

    # Read the original file's structure
    orig_data = bytearray(original)

    # Find end of last section data
    st_offset = header_size
    num_sections = int.from_bytes(orig_data[st_offset:st_offset + 4], 'little')
    entry_base = st_offset + 4

    max_end = len(orig_data) - 8  # last 8 bytes are checksum

    for i in range(num_sections):
        entry_off = entry_base + i * 56
        if entry_off + 56 > len(orig_data):
            break
        data_off = int.from_bytes(orig_data[entry_off + 8:entry_off + 16], 'little')
        data_size = int.from_bytes(orig_data[entry_off + 16:entry_off + 24], 'little')
        end = int(data_off) + int(data_size)
        if end > max_end:
            max_end = end

    # New data starts after all existing data (before checksum)
    section_data_offset = max_end

    # Add name to string table
    tree_name = "tree_head"
    name_start_abs = len(orig_data) - 8  # after checksum area? no, before
    # Actually, names are stored in the string table area
    # We need to find where the string table ends or reuse a position

    # Simpler approach: reuse the string table at the very beginning
    # Name offset is relative to... actually absolute in this format
    # Let's put the name right after the section table data

    # The original format: names are stored starting at some offset
    # We can just reuse existing name area or find the end
    # For simplicity, let's find the end of the string table

    # The names are stored in a string table before section data.
    # Let's find where section data starts

    first_data_offset = max_end
    for i in range(num_sections):
        entry_off = entry_base + i * 56
        data_off = int.from_bytes(orig_data[entry_off + 8:entry_off + 16], 'little')
        if int(data_off) < first_data_offset:
            first_data_offset = int(data_off)

    # Names are stored between entry ends and first data
    # Let's put new name after the last existing name
    # Find the last name's end
    last_name_end = first_data_offset
    for i in range(num_sections):
        entry_off = entry_base + i * 56
        name_off = int.from_bytes(orig_data[entry_off:entry_off + 4], 'little')
        name_len = int.from_bytes(orig_data[entry_off + 4:entry_off + 8], 'little')
        ns = int(name_off) + 2 + int(name_len)
        if ns > last_name_end:
            last_name_end = ns

    new_name_offset = last_name_end
    tree_name_bytes = tree_name.encode('utf-8')
    name_with_len = struct.pack('<H', len(tree_name_bytes)) + tree_name_bytes

    # Build new section table entry (56 bytes)
    new_entry = bytearray(56)
    # name_offset (absolute position)
    new_entry[0:4] = struct.pack('<I', new_name_offset)
    new_entry[4:8] = struct.pack('<I', len(tree_name_bytes))
    new_entry[8:16] = struct.pack('<Q', section_data_offset)
    new_entry[16:24] = struct.pack('<Q', len(tree_section_data))
    new_entry[24:28] = struct.pack('<I', 0xFF)  # dtype: raw
    new_entry[28:32] = struct.pack('<I', 1)  # shape rank
    new_entry[32:36] = struct.pack('<I', num_nodes)  # shape[0]
    # rest is zero (shape[1..3] and padding)

    # Build new file:
    # [original header + section table (unchanged)]
    # [new section name string]
    # [existing section data (as-is)]
    # [new section data]
    # [updated section count + new entry at the end of section table]
    # [checksum]

    # Actually this is getting complicated. Let me use a simpler approach:
    # Rewrite the entire file with the new section.

    # Remove old checksum
    file_without_checksum = orig_data[:-8]

    # Build updated section table area
    # Original: [header][section_table: numSections(4) + entries(numSections*56)][names][data][checksum]
    # We'll: append new entry after existing entries, add new name, add new data

    # Reconstruct:
    # [header (unchanged)]
    # [section table: numSections+1, original entries, new entry]
    # [original names + new name]
    # [original data]
    # [new section data]
    # [checksum]

    new_num_sections = num_sections + 1

    out = bytearray()
    # Header (up to section table)
    out.extend(orig_data[:st_offset])

    # Section table header
    out.extend(struct.pack('<I', new_num_sections))

    # Original entries
    for i in range(num_sections):
        entry_off = entry_base + i * 56
        out.extend(orig_data[entry_off:entry_off + 56])

    # New entry
    out.extend(new_entry)

    # Names area: original names
    name_area_start = len(out)
    original_name_area = bytes(orig_data[st_offset + 4 + num_sections * 56:first_data_offset])
    out.extend(original_name_area)

    # Add new name
    new_name_offset_in_file = len(out)
    out.extend(name_with_len)

    # Now we need to update the new entry's name_offset field
    entry_start = st_offset + 4 + num_sections * 56
    out[entry_start + 0:entry_start + 4] = struct.pack('<I', new_name_offset_in_file)
    out[entry_start + 4:entry_start + 8] = struct.pack('<I', len(tree_name_bytes))

    # Original data (starting after names)
    original_data_start = len(out)
    out.extend(bytes(file_without_checksum[first_data_offset:]))

    # Update data_offset and data_size in the new entry
    new_data_offset = len(out)
    out[entry_start + 8:entry_start + 16] = struct.pack('<Q', new_data_offset)
    out[entry_start + 16:entry_start + 24] = struct.pack('<Q', len(tree_section_data))

    # Append new section data
    out.extend(tree_section_data)

    # Compute checksum
    import hashlib
    checksum = hashlib.sha256(bytes(out)).digest()[:8]

    # Append checksum
    out.extend(checksum)

    with open(output_path, 'wb') as f:
        f.write(out)

    tree_size = len(tree_section_data)
    total_size = len(out)
    print(f"  Tree section: {tree_size} bytes")
    print(f"  Output: {output_path} ({total_size} bytes, +{tree_size + 56 + len(name_with_len) + 4} bytes)")
    print(f"  FP32 head was {vocab_size * hidden_dim * 4} bytes ({vocab_size * hidden_dim * 4 / tree_size:.1f}x larger)")


def main():
    parser = argparse.ArgumentParser(description="Build binary tree head from trained model")
    parser.add_argument("--input", required=True, help="Input BLMF file path")
    parser.add_argument("--output", required=True, help="Output BLMF file path")
    args = parser.parse_args()

    export_tree_blmf(args.input, args.output)


if __name__ == "__main__":
    main()
