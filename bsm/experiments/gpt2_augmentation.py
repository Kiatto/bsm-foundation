"""
GPT-2 + BSM augmentation with CORRECT architecture.
Hidden state = pooling of LAST 4 tokens' GPT-2 states (matches BSM's fixed context).
"""

import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, time
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.manual_seed(42); np.random.seed(42)

class BSMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
        )
    def forward(self, x):
        return self.net(x), torch.sign(self.net(x))

class MiniMemory:
    def __init__(self, dim=128, capacity=10000):
        self.dim=dim; self.capacity=capacity
        self.states=torch.zeros(capacity,dim)
        self.toks=np.zeros(capacity,dtype=np.int32)
        self.n=0
    def add(self,s,t):
        if self.n<self.capacity:
            self.states[self.n]=s; self.toks[self.n]=t; self.n+=1
    def recall(self,s,k=4):
        d=(s.unsqueeze(0)!=self.states[:self.n]).float().sum(1)
        top=d.topk(min(k,self.n),largest=False)
        return [(self.toks[top.indices[i].item()],top.values[i].item()) for i in range(top.indices.shape[0])]

print("Loading DistilGPT-2...", flush=True)
t0=time.time()
tok=AutoTokenizer.from_pretrained("distilgpt2")
tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained("distilgpt2")
model.eval()
print(f"  {time.time()-t0:.0f}s", flush=True)

encoder=BSMEncoder()
mem=MiniMemory(dim=128, capacity=10000)

# Data — use our BSM tokenizer for proper 4-token alignment
from tokenizers import Tokenizer
bsm_tok = Tokenizer.from_file("data/tokenizer.json")
with open("data/tinystories_train.txt") as f: text=f.read()
lines=[l.strip() for l in text.split("\n") if l.strip() and not l.startswith("<|endoftext|>")]
all_text=" ".join(lines); words=all_text.split()
stories_text=[" ".join(words[i:i+2000]) for i in range(0,len(words),2000) if len(" ".join(words[i:i+2000]))>200]

# Encode with BSM tokenizer for 4-token contexts
all_bsm_ids=[]
for s in stories_text:
    ids=bsm_tok.encode(s).ids
    if len(ids)>20: all_bsm_ids.append(ids)
print(f"BSM-tokenized sequences: {len(all_bsm_ids)}")

# === PHASE 1: Train encoder on GPT-2 states aligned with 4-token windows ===
print("Phase 1: Collecting aligned training data...")
train_pairs = []
with torch.no_grad():
    for seq in all_bsm_ids[:20]:
        # Convert to text, then to GPT-2 tokens for hidden state extraction
        # Simpler: use BSM tokenizer IDs directly as text? No, different vocabs.
        # Instead: use the original text for GPT-2, and 4-token sliding window for targets
        pass

# Alternative simpler approach: 
# 1. Use raw text positions
# 2. For each 4-word window, get GPT-2's hidden states (last 4 positions, mean pool)
# 3. Train encoder contrastively

print("Collecting hidden states from 4-token windows...")
train_raw = []  # (pooled_4token_hidden, next_token)

with torch.no_grad():
    for story in stories_text[:30]:
        gpt_ids = tok.encode(story)[:200]
        if len(gpt_ids) < 8: continue
        gpt_tensor = torch.tensor(gpt_ids).unsqueeze(0)
        outputs = model(gpt_tensor, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][0]  # [L, 768]
        
        for pos in range(4, len(gpt_ids)-1):
            # Pool last 4 positions: mean of hidden[pos-3:pos+1]
            ctx_hidden = hidden[max(0,pos-3):pos+1].mean(dim=0)  # [768]
            next_tok = gpt_ids[pos+1]
            train_raw.append((ctx_hidden, next_tok))

print(f"  {len(train_raw)} training samples")

# Train encoder
opt=torch.optim.AdamW(encoder.parameters(), lr=3e-4)
for step in range(500):
    batch=np.random.choice(len(train_raw), min(128,len(train_raw)), replace=False)
    hiddens=torch.stack([train_raw[i][0] for i in batch])
    tokens=[train_raw[i][1] for i in batch]
    
    raw, binary = encoder(hiddens)
    
    loss=0.0; n_pairs=0
    for i in range(len(batch)):
        for j in range(i+1,len(batch)):
            d=(binary[i]!=binary[j]).float().sum()
            same=(tokens[i]==tokens[j])
            if same:
                loss+=d/128.0
            else:
                loss+=max(0, 0.5-d/128.0)
            n_pairs+=1
    loss=loss/n_pairs
    loss+=0.01*((binary.abs()-1)**2).mean()
    
    opt.zero_grad(); loss.backward(); opt.step()
    if step%200==0: print(f"  step {step}: loss={loss.item():.4f}", flush=True)

print(f"  Final loss={loss.item():.4f}")

# === PHASE 2: Build memory ===
print("Phase 2: Building memory...")
n_stored=0
with torch.no_grad():
    for story in stories_text[30:45]:
        gpt_ids=tok.encode(story)[:200]
        if len(gpt_ids)<8: continue
        gpt_tensor=torch.tensor(gpt_ids).unsqueeze(0)
        outputs=model(gpt_tensor, output_hidden_states=True)
        hidden=outputs.hidden_states[-1][0]
        for pos in range(4,len(gpt_ids)-1):
            ctx_hidden=hidden[max(0,pos-3):pos+1].mean(dim=0).unsqueeze(0)
            _,state=encoder(ctx_hidden)
            mem.add(state[0],gpt_ids[pos+1]); n_stored+=1
            if n_stored>=mem.capacity: break
        if n_stored>=mem.capacity: break
print(f"  {mem.n} states")

# === PHASE 3: Evaluate ===
print("Phase 3: Evaluating...", flush=True)
cor_gpt=cor_aug=0; tot=0; t_lat=0
with torch.no_grad():
    for story in stories_text[45:55]:
        gpt_ids=tok.encode(story)[:200]
        if len(gpt_ids)<8: continue
        gpt_tensor=torch.tensor(gpt_ids).unsqueeze(0)
        outputs=model(gpt_tensor, output_hidden_states=True)
        logits=outputs.logits[0]; hidden=outputs.hidden_states[-1][0]
        for pos in range(4,len(gpt_ids)-1):
            t_tok=gpt_ids[pos+1]
            gpt_tok=logits[pos].argmax().item()
            cor_gpt+=gpt_tok==t_tok
            tq=time.perf_counter()
            ctx_hidden=hidden[max(0,pos-3):pos+1].mean(dim=0).unsqueeze(0)
            _,state=encoder(ctx_hidden)
            exps=mem.recall(state[0],k=4)
            t_lat+=time.perf_counter()-tq
            if exps:
                votes=Counter()
                for tid,d in exps:
                    votes[tid]+=1.0/(1.0+d/(128*2))
                mt=votes.most_common(1)[0][0]
                aug_tok=mt if votes[mt]>=sum(votes.values())*0.3 else gpt_tok
            else:
                aug_tok=gpt_tok
            cor_aug+=aug_tok==t_tok; tot+=1

mem_kb=(mem.n*mem.dim+mem.n*4)/1024
print(f"\n{'='*60}")
print("  GPT-2 + BSM (4-token pooled hidden states)")
print("="*60)
print(f"\n  {'Metric':25s} {'GPT-2':>12s} {'+BSM':>12s} {'Δ':>10s}")
print(f"  {'-'*59}")
print(f"  {'Accuracy':25s} {cor_gpt/tot*100:>10.2f}%  {cor_aug/tot*100:>10.2f}%  +{(cor_aug-cor_gpt)/tot*100:>6.2f}pp")
print(f"\n  BSM memory: {mem.n} states, {mem_kb:.0f} KB")
print(f"  Latency: {t_lat/tot*1e6:.0f} µs/query")
