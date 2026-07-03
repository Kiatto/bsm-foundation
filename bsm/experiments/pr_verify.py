"""
Verify intrinsic dimension: PR(binary) vs PR(raw), eigenvalue spectra.
Focus on D=128 only for speed, then check D=1024.
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, time

torch.manual_seed(42); np.random.seed(42)
IN_DIM=48; C=4; STEPS=2000; BATCH_SIZE=16; SEQ_LEN=64; LR=1e-3

from tokenizers import Tokenizer
tok=Tokenizer.from_file("data/tokenizer.json")
with open("data/tinystories_train.txt") as f: text=f.read()
lines=[l.strip() for l in text.split("\n") if l.strip() and not l.startswith("<|endoftext|>")]
all_text=" ".join(lines); words=all_text.split()
stories_text=[]
for i in range(0,len(words),2000):
    chunk=" ".join(words[i:i+2000])
    if len(chunk)>200: stories_text.append(chunk)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self,tok,texts,seq_len=64):
        self.tok=tok; self.seq_len=seq_len; self.tokens=[]
        for s in texts:
            ids=tok.encode(s).ids
            if len(ids)>=seq_len+1:
                for start in range(0,len(ids)-seq_len,seq_len//2):
                    self.tokens.append(torch.tensor(ids[start:start+seq_len+1]))
    def __len__(self): return len(self.tokens)
    def __getitem__(self,i):
        t=self.tokens[i]; return t[:self.seq_len],t[1:self.seq_len+1]

ds=TextDataset(tok,stories_text,seq_len=SEQ_LEN)
loader=torch.utils.data.DataLoader(ds,batch_size=BATCH_SIZE,shuffle=True,drop_last=True,num_workers=0)
def make_iter(): return iter(loader)
def next_batch(it):
    while True:
        try: x,y=next(it[0])
        except StopIteration: it[0]=iter(loader); continue
        break
    B,T=x.shape
    ctx_list,tgt_list=[],[]
    for t in range(C,T):
        ctx_list.append(x[:,t-C:t]); tgt_list.append(x[:,t])
    ctx=torch.stack(ctx_list,dim=1).reshape(-1,C)
    tgt=torch.stack(tgt_list,dim=1).reshape(-1)
    bits=int_to_bits(ctx,12).reshape(-1,IN_DIM)
    return bits,tgt
def int_to_bits(x,bits=12):
    return ((x.unsqueeze(-1)>>torch.arange(bits,device=x.device))&1).float()

class DSGModel(nn.Module):
    def __init__(self,state_dim,hid_dim):
        super().__init__()
        self.e1=nn.Linear(IN_DIM,hid_dim); self.e2=nn.Linear(hid_dim,state_dim)
        self.decoder=nn.Linear(state_dim,12)
    def forward(self,bits):
        h=torch.tanh(self.e1(bits)); sp=self.e2(h); sb=torch.sign(sp)
        return self.decoder(sp),sb,sp

def participation_ratio(X):
    C=np.cov(X.T) if X.shape[1] <= 2048 else None
    if C is None: return None
    ev=np.linalg.eigvalsh(C)[::-1]
    ev=ev[ev>1e-12]
    tv=np.sum(ev)
    return tv**2/np.sum(ev**2) if np.sum(ev**2)>0 else 0, ev

for D, H, label in [(128, 192, "D=128,H=192"), (1024, 1536, "D=1024,H=1536")]:
    print(f"\n{'-'*60}")
    print(f"  {label}")
    print(f"{'-'*60}")
    
    model=DSGModel(D,H)
    opt=torch.optim.AdamW(model.parameters(),lr=LR)
    it=[make_iter()]
    for step in range(STEPS):
        bits,tgt=next_batch(it)
        logits,_,_=model(bits)
        loss=F.binary_cross_entropy_with_logits(logits.reshape(-1,12),
            ((int_to_bits(tgt,12)+1)//2).float().reshape(-1,12))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    model.eval()
    
    # Collect states
    bin_list,raw_list=[],[]
    it_data=[make_iter()]; n=0
    with torch.no_grad():
        while n<2000:
            bits,_ =next_batch(it_data)
            _,sb,sp=model(bits)
            for i in range(bits.shape[0]):
                bin_list.append(sb[i].cpu()); raw_list.append(sp[i].cpu())
                n+=1
                if n>=2000: break
    
    X_bin=torch.stack(bin_list).numpy().astype(np.float64)
    X_raw=torch.stack(raw_list).numpy().astype(np.float64)
    
    # PR for binary
    pr_bin, ev_bin = participation_ratio(X_bin)
    
    # PR for raw
    pr_raw, ev_raw = participation_ratio(X_raw)
    
    # PR for random ±1
    pr_rand, _ = participation_ratio(np.random.choice([-1,1],size=(2000,D)).astype(np.float64))
    
    # PR for raw whitened (normalized to zero mean, unit variance per dimension)
    X_wn = (X_raw - X_raw.mean(0)) / (X_raw.std(0) + 1e-12)
    pr_wn, ev_wn = participation_ratio(X_wn)
    
    # Spectrum analysis
    cum_bin=np.cumsum(ev_bin)/np.sum(ev_bin)
    cum_raw=np.cumsum(ev_raw)/np.sum(ev_raw)
    
    n90_bin=int(np.searchsorted(cum_bin,0.9)+1)
    n95_bin=int(np.searchsorted(cum_bin,0.95)+1)
    n90_raw=int(np.searchsorted(cum_raw,0.9)+1)
    n95_raw=int(np.searchsorted(cum_raw,0.95)+1)
    
    # Effective rank
    eff_bin=np.sum(ev_bin)**2/np.sum(ev_bin**2)
    eff_raw=np.sum(ev_raw)**2/np.sum(ev_raw**2)
    
    print(f"  Binary states:  PR={pr_bin:.1f}  eff_rank={eff_bin:.1f}  n90={n90_bin}  n95={n95_bin}")
    print(f"  Raw states:     PR={pr_raw:.1f}  eff_rank={eff_raw:.1f}  n90={n90_raw}  n95={n95_raw}")
    print(f"  Whitened raw:   PR={pr_wn:.1f}")
    print(f"  Random ±1:      PR={pr_rand:.1f}")
    print(f"  D={D}: Binary PR/D={pr_bin/D:.3f}  Raw PR/D={pr_raw/D:.3f}")
    
    # Top eigenvalue analysis
    print(f"  Top 5 eigenvalues (binary): {ev_bin[:5]}")
    print(f"  Top 5 eigenvalues (raw):    {ev_raw[:5]}")
    print(f"  Num eigenvalues > 1% (bin): {np.sum(ev_bin/ev_bin[0]>0.01)}")
    print(f"  Num eigenvalues > 1% (raw): {np.sum(ev_raw/ev_raw[0]>0.01)}")
EOF
