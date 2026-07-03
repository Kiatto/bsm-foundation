"""
BSM Augmentation Demo — Memory Layer for an LSTM Language Model

Architecture:
  Input → LSTM → hidden state → BSM Encoder → Memory Engine → Recall → Augmented Prediction

Demonstrates that BSM memory improves a NON-BSM model (an LSTM).
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, time, math
from collections import Counter

torch.manual_seed(42); np.random.seed(42)

IN_DIM = 48; C = 4; D = 128; H = 192
STEPS = 3000; BATCH_SIZE = 16; SEQ_LEN = 64; LR = 1e-3
N_BUCKET_BITS = 12; N_BUCKETS = 1 << N_BUCKET_BITS

from tokenizers import Tokenizer
tok = Tokenizer.from_file("data/tokenizer.json")

# ============ Data ============
with open("data/tinystories_train.txt") as f: text = f.read()
lines = [l.strip() for l in text.split("\n") if l.strip() and not l.startswith("<|endoftext|>")]
all_text = " ".join(lines); words = all_text.split()
stories_text = []
for i in range(0, len(words), 2000):
    chunk = " ".join(words[i:i+2000])
    if len(chunk) > 200: stories_text.append(chunk)

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

ds = TextDataset(tok,stories_text,seq_len=SEQ_LEN)
loader = torch.utils.data.DataLoader(ds,batch_size=16,shuffle=True,drop_last=True,num_workers=0)
def make_iter(): return iter(loader)

def int_to_bits(x,bits=12):
    return ((x.unsqueeze(-1)>>torch.arange(bits,device=x.device))&1).float()

def bits_to_int(bits,nb=12):
    return ((bits>0).float()*(2**torch.arange(bits.shape[-1],device=bits.device)).float()).sum(-1).long()

# ============ LSTM Language Model ============
class LSTMLM(nn.Module):
    """Tiny LSTM that reads 4 tokens and predicts the next."""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(12, 128, batch_first=True)
        self.decoder = nn.Linear(128, 12)
    
    def forward(self, bits, return_hidden=False):
        # bits: [B, 4, 12] = [B, 48] in our encoding
        # Reshape to (B, 4, 12) for LSTM
        B = bits.shape[0]
        x = bits.view(B, C, 12)  # [B, 4, 12]
        out, (h, c) = self.lstm(x)
        logits = self.decoder(out[:, -1])  # [B, 12]
        if return_hidden:
            return logits, out[:, -1]  # last hidden
        return logits

# ============ TGAM ============
class TGAM:
    def __init__(self, simhash):
        self.simhash=simhash
        self.max_states=50000
        self._states=torch.zeros(self.max_states, D)
        self._tokens=np.zeros(self.max_states,dtype=np.int32)
        self._n=0; self.buckets=[[] for _ in range(N_BUCKETS)]
    def bucket(self,s):
        proj=s@self.simhash.T; b=0
        for i in range(N_BUCKET_BITS):
            if proj[i]>0: b|=1<<i
        return b
    def add(self,s,tok):
        idx=self._n
        if idx>=self.max_states: return
        self._states[idx]=s.cpu(); self._tokens[idx]=tok
        self.buckets[self.bucket(s)].append(idx); self._n+=1
    def build(self, model, make_batch, max_examples=50000):
        model.eval(); n=0
        with torch.no_grad():
            while n<max_examples:
                bits,tgt,lstm_hidden = make_batch(return_hidden=True)
                state = lstm_hidden  # use LSTM hidden as the state to memorize
                # Convert to binary via sign (simplest binarization)
                sb = torch.sign(state)
                for i in range(bits.shape[0]):
                    self.add(sb[i], tgt[i].item())
                    n+=1
                    if n>=max_examples: break
    def query_batch(self,sb,cand=200,neigh=4):
        results=[]
        for i in range(sb.shape[0]):
            s=sb[i]; b=self.bucket(s)
            cs=set()
            for idx in self.buckets[b]: cs.add(idx)
            if len(cs)<cand:
                for bb in range(N_BUCKET_BITS):
                    nb=b^(1<<bb)
                    for idx in self.buckets[nb]: cs.add(idx)
                    if len(cs)>=cand: break
            cs=list(cs)[:cand]
            if not cs: results.append(Counter()); continue
            dd=(s.unsqueeze(0)!=self._states[cs].to(s.device)).float().sum(1).cpu()
            top=sorted(enumerate(dd.tolist()),key=lambda x:x[1])[:neigh]
            votes=Counter()
            for wi,ddd in top:
                votes[self._tokens[cs[wi]]]+=1.0/(1.0+ddd/(D*2))
            results.append(votes)
        return results

# ============ Train LSTM ============
print(f"Training LSTM...", flush=True)
t0=time.time()
model = LSTMLM()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
it = [make_iter()]

for step in range(STEPS):
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
    
    logits = model(bits)
    loss = F.binary_cross_entropy_with_logits(logits.reshape(-1,12),
        ((int_to_bits(tgt,12)+1)//2).float().reshape(-1,12))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    if step%1000==0: print(f"  step {step}: loss={loss.item():.4f}", flush=True)

train_time=time.time()-t0
model.eval()
print(f"  Trained in {train_time:.0f}s", flush=True)

# ============ Evaluate LSTM alone ============
print(f"\nEvaluating LSTM baseline...", flush=True)
it_eval=[make_iter()]
cor=tot=0
with torch.no_grad():
    for _ in range(100):
        while True:
            try: x,y=next(it_eval[0])
            except StopIteration: it_eval[0]=iter(loader); continue
            break
        B,T=x.shape
        ctx_list,tgt_list=[],[]
        for t in range(C,T):
            ctx_list.append(x[:,t-C:t]); tgt_list.append(x[:,t])
        ctx=torch.stack(ctx_list,dim=1).reshape(-1,C)
        tgt=torch.stack(tgt_list,dim=1).reshape(-1)
        bits=int_to_bits(ctx,12).reshape(-1,IN_DIM)
        logits = model(bits)
        pred = bits_to_int(torch.sign(logits),12)
        cor+=(pred==tgt).sum().item(); tot+=tgt.shape[0]
acc_lstm = cor/tot*100
print(f"  LSTM alone: {acc_lstm:.2f}%", flush=True)

# ============ Build BSM Memory on LSTM hidden states ============
print(f"\nBuilding BSM Memory on LSTM hidden states...", flush=True)
simhash=torch.randn(N_BUCKET_BITS,D)
gam=TGAM(simhash)

def make_lstm_batch(return_hidden=False):
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
    logits, hidden = model(bits, return_hidden=True)
    if return_hidden:
        return bits, tgt, hidden
    return bits, tgt

it[0]=iter(loader)
gam.build(model, make_lstm_batch, max_examples=50000)
print(f"  Memory: {gam._n} states", flush=True)

# ============ Evaluate LSTM + BSM ============
print(f"\nEvaluating LSTM + BSM Memory...", flush=True)
it_aug=[make_iter()]
cor_lstm=cor_aug=0; tot=0
t_latency=0
with torch.no_grad():
    for _ in range(100):
        while True:
            try: x,y=next(it_aug[0])
            except StopIteration: it_aug[0]=iter(loader); continue
            break
        B,T=x.shape
        ctx_list,tgt_list=[],[]
        for t in range(C,T):
            ctx_list.append(x[:,t-C:t]); tgt_list.append(x[:,t])
        ctx=torch.stack(ctx_list,dim=1).reshape(-1,C)
        tgt=torch.stack(tgt_list,dim=1).reshape(-1)
        bits=int_to_bits(ctx,12).reshape(-1,IN_DIM)
        
        # LSTM prediction
        logits, hidden = model(bits, return_hidden=True)
        pred_lstm = bits_to_int(torch.sign(logits),12)
        
        # BSM memory query (on binarized hidden state)
        tq=time.perf_counter()
        sb = torch.sign(hidden)
        votes_list = gam.query_batch(sb)
        t_latency += time.perf_counter()-tq
        
        for i in range(bits.shape[0]):
            t_tok = tgt[i].item()
            lstm_tok = pred_lstm[i].item()
            cor_lstm += lstm_tok == t_tok
            
            # Augmented: preference for LSTM, but consult memory
            v = votes_list[i]
            if v:
                mem_tok = v.most_common(1)[0][0]
                # Weighted: 60% memory, 40% LSTM
                # (hyperparameter: can tune)
                if v[mem_tok] >= len(v) * 0.3:  # if memory is confident
                    aug_tok = mem_tok
                else:
                    aug_tok = lstm_tok
            else:
                aug_tok = lstm_tok
            
            cor_aug += aug_tok == t_tok
            tot += 1

acc_aug = cor_aug/tot*100
print(f"  LSTM alone:     {cor_lstm/tot*100:.2f}%", flush=True)
print(f"  LSTM + BSM:     {acc_aug:.2f}%", flush=True)
print(f"  Improvement:    +{acc_aug - cor_lstm/tot*100:.2f} pp", flush=True)
print(f"  Memory latency: {t_latency/tot*1e6:.0f}µs/query", flush=True)

# ============ Knowledge Density ============
# Memory size
mem_bytes = (gam._n * D + gam._n * 12) / 8  # states + tokens
model_bytes = sum(p.numel() for p in model.parameters()) * 4  # float32
total_bytes = model_bytes + mem_bytes
total_kb = total_bytes / 1024
total_mb = total_kb / 1024

# Information gain
lstm_unc = -math.log2(max(cor_lstm/tot/100, 0.001))
aug_unc = -math.log2(max(cor_aug/tot/100, 0.001))
bits_gained = lstm_unc - aug_unc
kd = bits_gained / total_mb if total_mb > 0 else 0

print(f"\n{'='*65}")
print(f"  BSM AUGMENTATION — FINAL REPORT")
print(f"{'='*65}")
print(f"\n  {'Metric':25s} {'LSTM alone':>12s} {'LSTM+BSM':>12s} {'Δ':>10s}")
print(f"  {'-'*59}")
print(f"  {'Accuracy':25s} {cor_lstm/tot*100:>11.2f}% {acc_aug:>11.2f}% +{acc_aug-cor_lstm/tot*100:>7.2f}pp")
print(f"  {'Uncertainty (bits)':25s} {lstm_unc:>11.3f}  {aug_unc:>11.3f}  -{lstm_unc-aug_unc:>7.3f}")
print(f"\n  Memory (BSM): {mem_bytes/1024:.0f} KB")
print(f"  Model (LSTM): {model_bytes/1024:.0f} KB")
print(f"  Total: {total_kb:.0f} KB = {total_mb:.3f} MB")
print(f"  KD: {bits_gained:.3f} bits / {total_mb:.4f} MB = {kd:.3f} bits/MB")
print(f"  Latency: {t_latency/tot*1e6:.0f} µs/query")
