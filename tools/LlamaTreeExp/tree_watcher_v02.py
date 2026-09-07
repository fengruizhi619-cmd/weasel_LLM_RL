#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cli_emojiless_exp_v0.2 - context reader + candidate tree + online RL.

Pipeline:
  C# reader detects commit → writes to log file
  This script tails log → parses context → builds candidate tree
    (caches hidden state) → user types → reward → zero-cost lm_head update
"""

import argparse, concurrent.futures, json, math, os, re, socket, subprocess, sys, threading, time, unicodedata

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import requests
except ImportError:
    print("[ERROR] pip install requests"); sys.exit(1)

# ---- config ----
MODEL_PATH   = r"E:\codex_data\研究\models\Qwen3-0.6B-Base"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
LR           = 1e-4
WIDTH        = 5
DEPTH        = 5
TOP_N        = 5
CTX_CHARS    = 100
CTX_SIZE     = 1024
PARALLEL     = 8
MAX_WORKERS  = 8
REQ_TIMEOUT  = 30
LOG_POLL     = 0.15

# ---- node ----
class Node:
    __slots__ = ("token","prob","cum_prob","parent","children","is_leaf","stop","depth","hidden")
    def __init__(self, tok="", p=1.0, cum=1.0, parent=None, depth=0):
        self.token=tok; self.prob=p; self.cum_prob=cum
        self.parent=parent; self.children=[]; self.is_leaf=False
        self.stop=""; self.depth=depth; self.hidden=None
    @property
    def path(self):
        parts=[]; n=self
        while n and n.parent: parts.append(n.token); n=n.parent
        return "".join(reversed(parts))
    @property
    def is_root(self): return self.parent is None

# ---- utils ----
def is_punct(t):
    return any(unicodedata.category(c).startswith("P") for c in t)
def is_eos(t):
    lo=t.lower()
    return "endoftext" in lo or "eos" in lo or "<|im_end|>" in lo or t in("\n","\r\n")
def free_port():
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s: s.bind(("",0)); return s.getsockname()[1]

# ---- llama-server (for candidate tree API) ----
class TreeServer:
    def __init__(self, exe, model, port):
        self.port=port
        self.base=f"http://127.0.0.1:{port}"; self._proc=None
        self._exe=exe; self._model=model
    def start(self):
        self._proc=subprocess.Popen(
            [self._exe,"-m",self._model,"--port",str(self.port),
             "--ctx-size","1024","--parallel","8","--cont-batching","--no-warmup"],
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        # fix: extract port from base url
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            try:
                if requests.get(self.base+"/health",timeout=2).status_code==200: return
            except: pass
            time.sleep(0.3)
        raise RuntimeError("server not healthy")
    def stop(self):
        if self._proc: self._proc.terminate()
        try: self._proc.wait(timeout=5)
        except: self._proc.kill()
    def top_n(self, prompt, n):
        r=requests.post(self.base+"/completion", json={
            "prompt":prompt,"n_predict":1,"n_probs":n,
            "temperature":1.0,"top_k":0,"top_p":1.0,"min_p":0.0,"cache_prompt":True
        }, timeout=REQ_TIMEOUT)
        r.raise_for_status()
        lp=r.json().get("completion_probabilities",[{}])[0].get("top_logprobs",[])
        return [{"tok":i.get("token",""),"p":math.exp(i.get("logprob",-999)),"id":i.get("id",0)} for i in lp]

# ---- RL update (zero-cost, uses cached hidden) ----
def rl_update(model, optimizer, hidden, target_id, reward):
    """hidden: (1,1,1024) cached from tree build. Zero backbone cost."""
    model.train()
    # lm_head forward using cached hidden (no backbone re-forward)
    logits = model.lm_head(hidden)  # (1,1,vocab)
    logits = logits[0,-1,:]  # (vocab,)
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -reward * log_probs[target_id]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.eval()
    return loss.item()

def get_hidden(model, input_ids):
    """Run backbone only (no lm_head), return hidden state at last position."""
    with torch.no_grad():
        out = model.model(input_ids=input_ids)  # backbone only
        hidden = out.last_hidden_state[:, -1, :].unsqueeze(1)  # (1,1,hidden_dim)
    return hidden

def get_top_k_pytorch(model, tokenizer, input_ids, k=5):
    with torch.no_grad():
        out = model(input_ids)
        logits = out.logits[0,-1,:]
        probs = F.softmax(logits, dim=-1)
        top_p, top_id = torch.topk(probs, k)
    return [{"tok":tokenizer.decode(top_id[i].item()),"p":top_p[i].item(),
             "id":top_id[i].item()} for i in range(top_id.shape[0])]

# ---- main ----
def main():
    ap=argparse.ArgumentParser(description="cli_emojiless_exp_v0.2")
    ap.add_argument("--log-file",required=True)
    ap.add_argument("-n",type=int,default=WIDTH)
    ap.add_argument("-d",type=int,default=DEPTH)
    ap.add_argument("--top-n",type=int,default=TOP_N)
    ap.add_argument("--model",default=MODEL_PATH)
    ap.add_argument("--server",default=r"E:\llama.cpp\llama-server.exe")
    ap.add_argument("--ctx-chars",type=int,default=CTX_CHARS)
    ap.add_argument("--rl-lr",type=float,default=LR)
    ap.add_argument("--llama-port",type=int,default=0,help="0=auto")
    args=ap.parse_args()

    log_file=os.path.abspath(args.log_file)
    if not os.path.exists(log_file):
        print(f"[v0.2] log not found: {log_file}",file=sys.stderr); return 1
    last_pos=os.path.getsize(log_file)
    print(f"[v0.2] watching {log_file} (offset {last_pos})",flush=True)

    # load PyTorch model (for RL + candidate generation via PyTorch)
    print(f"[v0.2] loading model {args.model}",flush=True)
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForCausalLM.from_pretrained(args.model).to(DEVICE)

    # untie lm_head
    if model.lm_head.weight is model.model.embed_tokens.weight:
        print("[v0.2] untying lm_head from embed_tokens",flush=True)
        import torch.nn as nn
        model.lm_head.weight=nn.Parameter(model.model.embed_tokens.weight.data.clone())

    for p in model.parameters(): p.requires_grad=False
    model.lm_head.weight.requires_grad=True
    optimizer=torch.optim.SGD([model.lm_head.weight],lr=args.rl_lr)
    print(f"[v0.2] model ready on {DEVICE}, lr={args.rl_lr}",flush=True)

    # start llama-server for tree API (separate from PyTorch model)
    port=args.llama_port if args.llama_port else free_port()
    print(f"[v0.2] starting llama-server on port {port}",flush=True)
    gguf = args.model if '.gguf' in args.model else r"E:\llama.cpp\models\Qwen3-0.6B-Base-Q8_0.gguf"
    tsrv=TreeServer(args.server,gguf,port)
    try: tsrv.start()
    except Exception as e:
        print(f"[v0.2] [WARN] llama-server failed: {e}, using PyTorch only",flush=True)
        tsrv=None

    print(f"[v0.2] ready. type Chinese in any app, Ctrl+C to stop\n",flush=True)

    running=True; commits=0
    import signal as sig_mod
    def handler(sig,frame):
        nonlocal running; running=False
    sig_mod.signal(sig_mod.SIGINT,handler)

    # hidden state cache: context_text → (hidden_tensor, input_ids)
    hidden_cache={}

    try:
        while running:
            time.sleep(LOG_POLL)
            if not os.path.exists(log_file): continue
            cur=os.path.getsize(log_file)
            if cur<=last_pos: continue
            with open(log_file,"r",encoding="utf-8",errors="replace") as f:
                f.seek(last_pos); new=f.read()
            last_pos=os.path.getsize(log_file)

            for line in new.splitlines():
                m=re.search(r"ctx\(\d+/\d+\):\s(.+)",line)
                if not m: continue
                ctx=m.group(1).strip()
                if not ctx: continue
                if len(ctx)>args.ctx_chars: ctx=ctx[-args.ctx_chars:]
                commits+=1

                print(f"\n{'='*60}",flush=True)
                print(f"[commit #{commits}] ctx: {ctx!r}",flush=True)
                print(f"{'='*60}",flush=True)

                t0=time.monotonic()

                # build candidate tree via llama-server
                try:
                    cands=tsrv.top_n(ctx,args.n) if tsrv else []
                except Exception as e:
                    print(f"[v0.2] [WARN] tree API error: {e}",flush=True); cands=[]

                # PyTorch forward: get hidden state for RL
                input_ids=tokenizer.encode(ctx,return_tensors="pt").to(DEVICE)
                hidden=get_hidden(model,input_ids)
                hidden_cache[ctx]=(hidden,input_ids)

                # show tree-level candidates from llama-server
                if cands:
                    for i,c in enumerate(cands):
                        print(f"  tree[{i+1}] {c['tok']!r} p={c['p']:.4f}",flush=True)

                # PyTorch top-k (for RL target lookup)
                pt_cands=get_top_k_pytorch(model,tokenizer,input_ids,args.n)

                elapsed=time.monotonic()-t0
                print(f"[v0.2] tree built {elapsed:.2f}s | "
                      f"llama={len(cands)} pytorch={len(pt_cands)} cands",flush=True)

                # simulate: show PyTorch top candidates for RL
                for i,c in enumerate(pt_cands[:args.top_n]):
                    print(f"  pt[{i+1}] p={c['p']:.4f} {c['tok']!r}",flush=True)

                # simulate user typing: pick the top candidate for demo
                # (real version: user types via IME, we match)
                if pt_cands:
                    user_tok=pt_cands[0]
                    reward=user_tok["p"]  # full match
                    target_id=user_tok["id"]
                    print(f"[v0.2] [RL] user picked {user_tok['tok']!r} "
                          f"reward={reward:.4f}",flush=True)

                    # zero-cost RL update using cached hidden
                    loss=rl_update(model,optimizer,hidden,target_id,reward)
                    print(f"[v0.2] [RL] updated lm_head, loss={loss:.6f}",flush=True)

                    # re-query to show change
                    new_cands=get_top_k_pytorch(model,tokenizer,input_ids,3)
                    for i,c in enumerate(new_cands):
                        marker=" ←" if c["id"]==target_id else ""
                        print(f"  after[{i+1}] p={c['p']:.4f} {c['tok']!r}{marker}",flush=True)

    except Exception as e:
        print(f"[v0.2] [ERROR] {e}",file=sys.stderr)
    finally:
        if tsrv: tsrv.stop()
        print(f"\n[v0.2] stopped. commits={commits}",flush=True)
    return 0

if __name__=="__main__":
    sys.exit(main())
