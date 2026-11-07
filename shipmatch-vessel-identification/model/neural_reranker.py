"""Offline deterministic Qwen name-pair scoring. No test aggregates or generation."""
from pathlib import Path
import json,os
import numpy as np

INSTRUCTION=("Match a vessel name to the same name in the catalog. The query may be a Korean "
 "phonetic transliteration of an English name. Compare pronunciation and spelling across "
 "languages. Preserve distinguishing words and numbers; ignore punctuation and the suffix Ho. "
 "A shared fleet prefix alone does not make two names the same. Judge names only.")
PREFIX=('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
 'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUFFIX='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

class NameReranker:
 def __init__(self,path,device=None,batch_size=16,max_length=256):
  os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
  import torch
  from transformers import AutoTokenizer,AutoModelForCausalLM
  torch.set_num_threads(6)
  torch.use_deterministic_algorithms(True)
  self.torch=torch
  self.device=device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
  # Fixed lengths remove padding shape dependence between singleton and batches.
  self.batch_size=batch_size;self.max_length=max_length;self.cache={}
  self.tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True,padding_side='left')
  dtype=torch.float32 if self.device=='cpu' else torch.float16
  self.model=AutoModelForCausalLM.from_pretrained(str(path),local_files_only=True,
       torch_dtype=dtype,attn_implementation='eager').to(self.device).eval()
  self.no=self.tokenizer.convert_tokens_to_ids('no');self.yes=self.tokenizer.convert_tokens_to_ids('yes')
  self.prefix=self.tokenizer.encode(PREFIX,add_special_tokens=False)
  self.suffix=self.tokenizer.encode(SUFFIX,add_special_tokens=False)

 def apply_adapter(self,path):
  """Merge our q/v-only rank-8 adapter without an extra inference dependency."""
  from safetensors.torch import load_file
  path=Path(path);config=json.loads((path/'adapter_config.json').read_text())
  if set(config['target_modules'])!={'q_proj','v_proj'} or config.get('fan_in_fan_out') or config.get('bias')!='none':
   raise ValueError('Unsupported adapter architecture')
  torch=self.torch;torch.backends.cuda.matmul.allow_tf32=False
  weights=load_file(str(path/'adapter_model.safetensors'),device=self.device)
  scale=float(config['lora_alpha'])/int(config['r']);base=dict(self.model.named_parameters())
  self.model.to(torch.bfloat16)
  applied=0
  with torch.no_grad():
   for key,a in weights.items():
    if not key.endswith('.lora_A.weight'):continue
    b=weights[key.replace('.lora_A.weight','.lora_B.weight')]
    target=key.removeprefix('base_model.model.').replace('.lora_A.weight','.weight')
    if target not in base:raise ValueError('Adapter target mismatch')
    delta=(b.float()@a.float())*scale
    base[target].add_(delta);applied+=1
  if applied==0 or applied*2!=len(weights):raise ValueError('Incomplete adapter merge')
  self.model.to(torch.float16 if self.device!='cpu' else torch.float32).eval();self.cache.clear()

 def inputs(self,pairs):
  texts=[f'<Instruct>: {INSTRUCTION}\n<Query>: {q}\n<Document>: {c}' for q,c in pairs]
  ids=self.tokenizer(texts,padding=False,truncation=True,add_special_tokens=False,
      max_length=self.max_length-len(self.prefix)-len(self.suffix))['input_ids']
  ids=[self.prefix+x+self.suffix for x in ids]
  batch=self.tokenizer.pad({'input_ids':ids},padding='max_length',max_length=self.max_length,return_tensors='pt')
  # Explicit position ids make left padding equivalent to an unpadded row.
  batch['position_ids']=(batch['attention_mask'].long().cumsum(-1)-1).clamp_min(0)
  return {k:v.to(self.device) for k,v in batch.items()}

 def score(self,pairs):
  pairs=[(str(q),str(c)) for q,c in pairs]
  pending=list(dict.fromkeys(p for p in pairs if p not in self.cache))
  torch=self.torch
  with torch.inference_mode():
   for s in range(0,len(pending),self.batch_size):
    ps=pending[s:s+self.batch_size]
    # A fixed batch shape also makes a singleton use the same GEMM geometry.
    padded=ps+[('', '')]*(self.batch_size-len(ps));inp=self.inputs(padded)
    # Only project the final hidden state, avoiding B x L x 151936 vocabulary logits.
    hidden=self.model.model(**inp,use_cache=False).last_hidden_state[:,-1,:]
    weights=self.model.lm_head.weight[[self.no,self.yes]]
    logits=torch.nn.functional.linear(hidden,weights).float()
    values=(logits[:,1]-logits[:,0]).cpu().numpy()[:len(ps)]
    for p,v in zip(ps,values):self.cache[p]=float(v)
  return np.asarray([self.cache[p] for p in pairs],dtype=np.float32)
