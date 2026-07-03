#!/usr/bin/env python3
"""Evaluate improved STE tree (100K states, 500 epochs)."""
import sys, os, struct
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from blm.model import BSMModel, BSMConfig
from blm.tokenizer import load_tokenizer
from blm.export import load_from_blmf
from scripts.train_tree_head import collect_data
from scripts.bench_tree_accuracy import pack_vector, tree_predict

HPATH = "/var/www/html/BitKore"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(path):
    info = load_from_blmf(path)
    hdr = info["header"]
    config = BSMConfig(**{k: hdr[k] for k in ["vocab_size","hidden_dim","num_layers","seq_len","window_size"]})
    model = BSMModel(config)
    model.eval()
    pm = {"embedding":"embedding.weight","head_weight":"head.weight"}
    for li in range(config.num_layers):
        pm[f"layer_{li}_wforget"]=f"layers.{li}.state_update.W_forget.weight"
        pm[f"layer_{li}_winput"]=f"layers.{li}.state_update.W_input.weight"
        pm[f"layer_{li}_wmix"]=f"layers.{li}.mixer.W_mix.weight"
    sd = model.state_dict()
    for sn, sec in info["sections"].items():
        if sn not in pm: continue
        pn = pm[sn]; 
        if pn not in sd: continue
        arr = np.frombuffer(sec["data"], dtype={3: np.float32}.get(sec["dtype"], np.uint8))
        if sec["shape"]: arr = arr.reshape(sec["shape"])
        p = torch.from_numpy(arr.copy())
        if p.shape != sd[pn].shape:
            tgt = sd[pn]
            if p.dtype == torch.uint8 and tgt.dtype in (torch.float32, torch.float64):
                bits = torch.zeros(tgt.shape, dtype=torch.float32)
                for j in range(tgt.numel()):
                    byte_idx, bit_idx = j//8, j%8
                    bits.view(-1)[j] = 1.0 if (p.view(-1)[byte_idx].item()>>bit_idx)&1 else -1.0
                p = bits
            else: continue
        sd[pn] = p
    model.load_state_dict(sd, strict=False)
    return model, config

def load_tree(bin_path, num_nodes, nw):
    with open(bin_path,"rb") as f:
        data = f.read()
    node_bytes = nw*8; hdr_size = 16
    expected_no_bm = hdr_size + num_nodes*node_bytes
    bm_size = (num_nodes+7)//8
    has_bm = len(data) == expected_no_bm + bm_size
    nodes = []
    for i in range(num_nodes):
        off = hdr_size + i*node_bytes
        chunk = data[off:off+node_bytes]
        if has_bm:
            bm_byte = data[expected_no_bm + i//8]
            bm_bit = (bm_byte >> (i%8)) & 1
            nodes.append(chunk if bm_bit else None)
        else:
            nodes.append(None if all(b==0 for b in chunk) else chunk)
    return nodes, struct.unpack('<I', data[12:16])[0]

print("[*] Loading Fast model...")
model, config = load_model(f"{HPATH}/checkpoints/tinystories_fast.blmf")
model.to(DEVICE)
V, D, L = config.vocab_size, config.hidden_dim, config.num_layers
print(f"    V={V} D={D} L={L}")

leaf_count = 2
while leaf_count < V: leaf_count <<= 1
num_nodes = 2*leaf_count-1; nw = (D+63)//64

print("[*] Loading improved STE tree (100K, 500 epochs)...")
nodes_old, _ = load_tree("/tmp/ste_improved_tree.bin", num_nodes, nw)
trained_old = sum(1 for n in nodes_old if n is not None)
print(f"    {trained_old}/{num_nodes} trained")

print("[*] Loading original STE tree (20K, 200 epochs)...")
nodes_new, leaf_base = load_tree("/tmp/ste_trained_tree.bin", num_nodes, nw)
trained_new = sum(1 for n in nodes_new if n is not None)
print(f"    {trained_new}/{num_nodes} trained")

# Collect 2000 eval samples
print("[*] Collecting 2000 eval samples...")
tok = load_tokenizer(f"{HPATH}/checkpoints/tinystories_vocab4096.json")
H, T = collect_data(model, tok, f"{HPATH}/data/tinystories_val.txt", max_steps=2000)

# Evaluate both trees
leaf_map = {i:i for i in range(V)}
head_w = model.head.weight.data.cpu().numpy()

for label, nodes in [("Improved (100K, 500ep)", nodes_old), ("Original (20K, 200ep)", nodes_new)]:
    total = 0; t1k = 0; t5k = 0; t40k = 0
    with torch.no_grad():
        for i in range(min(500, len(H))):
            h = H[i]
            logits = head_w @ h.numpy()
            fp32_top1 = int(np.argmax(logits))
            fp32_top5 = set(np.argsort(logits)[-5:].tolist())
            fp32_top40 = set(np.argsort(logits)[-40:].tolist())
            
            packed = pack_vector(h.numpy())
            leaf = tree_predict(nodes, leaf_map, leaf_base, nw, packed, D)
            
            total += 1
            if leaf == fp32_top1: t1k += 1
            if leaf in fp32_top5: t5k += 1
            if leaf in fp32_top40: t40k += 1
    
    print(f"\n  {label}:")
    print(f"    Top-1: {100*t1k/total:.1f}%")
    print(f"    Top-5: {100*t5k/total:.1f}%")
    print(f"    Top-40: {100*t40k/total:.1f}%")

print("\nDone.")
