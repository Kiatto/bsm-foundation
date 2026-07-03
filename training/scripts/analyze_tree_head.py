#!/usr/bin/env python3
"""Analyze STE-trained tree head accuracy by token frequency quartile."""

import sys, os, struct, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from blm.model import BSMModel, BSMConfig
from blm.tokenizer import load_tokenizer
from blm.export import load_from_blmf
from scripts.train_tree_head import collect_data
from scripts.bench_tree_accuracy import pack_vector, tree_predict

HPATH = "/var/www/html/BitKore"

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
    node_bytes = nw * 8
    hdr_size = 16
    # Check for bitmask at end of file
    expected_no_bm = hdr_size + num_nodes * node_bytes
    bitmask_size = (num_nodes + 7) // 8
    has_bitmask = (len(data) == expected_no_bm + bitmask_size)
    
    nodes = []
    for i in range(num_nodes):
        off = hdr_size + i * node_bytes
        chunk = data[off:off+node_bytes]
        if has_bitmask:
            bm_byte = data[expected_no_bm + i // 8]
            bm_bit = (bm_byte >> (i % 8)) & 1
            if not bm_bit:
                nodes.append(None)
            else:
                nodes.append(chunk)
        else:
            # Old format: assume non-zero means trained
            if all(b == 0 for b in chunk):
                nodes.append(None)
            else:
                nodes.append(chunk)
    return nodes, struct.unpack('<I', data[12:16])[0]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mp = f"{HPATH}/checkpoints/tinystories_fast.blmf"
    tp = f"{HPATH}/checkpoints/tinystories_vocab4096.json"
    dp = f"{HPATH}/data/tinystories_val.txt"

    print("[*] Loading model...")
    model, config = load_model(mp); model.to(device)
    V, D, L = config.vocab_size, config.hidden_dim, config.num_layers
    print(f"    V={V} D={D} L={L}")
    
    leaf_count = 2
    while leaf_count < V: leaf_count <<= 1
    num_nodes = 2*leaf_count-1; nw = (D+63)//64
    nodes, leaf_base = load_tree("/tmp/ste_trained_tree.bin", num_nodes, nw)
    trained = sum(1 for n in nodes if n is not None and any(b!=0 for b in n))
    print(f"    Tree: {trained}/{num_nodes} trained")
    
    # Load precomputed frequency quartiles
    with open("/tmp/freq_quartiles.json") as f:
        fq = json.load(f)
    quartiles = fq["quartiles"]
    tok_to_q = {}
    for qi, q in enumerate(quartiles):
        for tok in q: tok_to_q[tok] = qi
    for qi, q in enumerate(quartiles):
        print(f"  Q{qi+1}: {len(q)} tokens")
    
    # Collect eval data
    print("[*] Collecting 2000 eval pairs...")
    H, T = collect_data(model, load_tokenizer(tp), dp, max_steps=2000)
    
    # Quick neighbor-target accuracy by quartile
    print("\n[*] Tree==Target by frequency quartile:")
    leaf_map = {i:i for i in range(V)}
    q_acc = {qi:{"n":0,"ok":0} for qi in range(4)}
    overall = 0; overall_ok = 0
    with torch.no_grad():
        for i in range(len(H)):
            h, t = H[i], T[i].item()
            packed = pack_vector(h.numpy())
            leaf = tree_predict(nodes, leaf_map, leaf_base, nw, packed, D)
            ok = leaf == t
            overall += 1; overall_ok += ok
            qi = tok_to_q.get(t, 3)
            q_acc[qi]["n"] += 1; q_acc[qi]["ok"] += ok
    
    print(f"  {'Q':<6} {'Samples':<10} {'Acc%':<10}")
    for qi in range(4):
        s = q_acc[qi]; p = 100*s["ok"]/s["n"] if s["n"] else 0
        print(f"  Q{qi+1:<5} {s['n']:<10} {p:<9.1f}%")
    print(f"  {'All':<6} {overall:<10} {100*overall_ok/overall:<9.1f}%")
    
    # FP32 comparison (500 samples)
    print("\n[*] Tree vs FP32 argmax (500 samples):")
    head_w = model.head.weight.data.cpu().numpy()
    total = 0; t1k = 0; t5k = 0; t40k = 0
    q_fp32 = {qi:{"n":0,"t1":0,"t5":0,"t40":0} for qi in range(4)}
    
    for i in range(min(500, len(H))):
        h, t = H[i], T[i].item()
        logits = head_w @ h.numpy()
        fp32_top1 = int(np.argmax(logits))
        fp32_top5 = set(np.argsort(logits)[-5:].tolist())
        fp32_top40 = set(np.argsort(logits)[-40:].tolist())
        
        packed = pack_vector(h.numpy())
        leaf = tree_predict(nodes, leaf_map, leaf_base, nw, packed, D)
        total += 1
        qi = tok_to_q.get(t, 3)
        if leaf == fp32_top1: t1k += 1; q_fp32[qi]["t1"] += 1
        if leaf in fp32_top5: t5k += 1; q_fp32[qi]["t5"] += 1
        if leaf in fp32_top40: t40k += 1; q_fp32[qi]["t40"] += 1
        q_fp32[qi]["n"] += 1
    
    print(f"  {'Metric':<15} {'Value':<10}")
    print(f"  {'Top-1 (vs FP32)':<15} {100*t1k/total:<7.1f}%")
    print(f"  {'Top-5 (vs FP32)':<15} {100*t5k/total:<7.1f}%")
    print(f"  {'Top-40 (vs FP32)':<15} {100*t40k/total:<7.1f}%")
    
    print(f"\n  {'Quartile':<8} {'Top-1%':<8} {'Top-5%':<8} {'Top-40%':<9} {'Samples':<8}")
    for qi in range(4):
        s = q_fp32[qi]; st = s["n"] or 1
        print(f"  Q{qi+1:<7} {100*s['t1']/st:<7.1f}% {100*s['t5']/st:<7.1f}% {100*s['t40']/st:<8.1f}% {st:<8}")

if __name__ == "__main__":
    main()
