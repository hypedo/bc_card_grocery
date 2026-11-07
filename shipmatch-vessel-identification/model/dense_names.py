"""Generic pretrained name embeddings; deterministic independent-query retrieval."""
from pathlib import Path
import os,numpy as np

TASK='Retrieve the catalog vessel name with the same spelling or pronunciation as this name, allowing Korean transliteration.'

class NameEncoder:
 def __init__(self,path,device=None,batch_size=32):
  os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
  import torch
  from transformers import AutoTokenizer,AutoModel
  torch.set_num_threads(6);torch.use_deterministic_algorithms(True)
  self.torch=torch;self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu');self.batch_size=batch_size;self.cache={}
  self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,padding_side='left')
  self.model=AutoModel.from_pretrained(path,local_files_only=True,torch_dtype=torch.float16 if self.device=='cuda' else torch.float32,attn_implementation='eager').to(self.device).eval()

 def encode(self,names,query=True):
  names=list(map(str,names));keys=[(query,n) for n in names];pending=list(dict.fromkeys(k for k in keys if k not in self.cache));torch=self.torch
  length=128 if query else 64
  with torch.inference_mode():
   for s in range(0,len(pending),self.batch_size):
    keys_now=pending[s:s+self.batch_size];texts=[f'Instruct: {TASK}\nQuery: {n}' if q else n for q,n in keys_now]
    texts+=['']*(self.batch_size-len(texts))
    batch=self.tokenizer(texts,padding='max_length',truncation=True,max_length=length,return_tensors='pt')
    # All real names end in a nonpadding token under left padding. Empty fillers are ignored.
    batch['position_ids']=(batch['attention_mask'].long().cumsum(-1)-1).clamp_min(0)
    batch={k:v.to(self.device) for k,v in batch.items()}
    hidden=self.model(**batch,use_cache=False).last_hidden_state[:,-1,:].float()
    vec=torch.nn.functional.normalize(hidden,p=2,dim=1).cpu().numpy()[:len(keys_now)]
    for k,v in zip(keys_now,vec):self.cache[k]=v.astype(np.float32)
  return np.stack([self.cache[k] for k in keys]) if keys else np.empty((0,1024),np.float32)

class DenseNames:
 def __init__(self,path,catalog,encoder=None):
  data=np.load(path);self.ids=data['ids'].astype(str);self.vectors=data['vectors'].astype(np.float32)
  self.id_to_row={s:i for i,s in enumerate(self.ids)}
  catalog_lookup=dict(zip(catalog.vessel_id.astype(str),range(len(catalog))))
  self.row_to_catalog=np.asarray([catalog_lookup[s] for s in self.ids],dtype=np.int32)
  self.catalog_to_row=np.asarray([self.id_to_row[s] for s in catalog.vessel_id.astype(str)],dtype=np.int32)
  self.encoder=encoder;self.precomputed={};self.cache={}

 def add_queries(self,path):
  data=np.load(path)
  self.precomputed.update(zip(data['names'].astype(str),data['vectors'].astype(np.float32)))

 def vector(self,name):
  name=str(name)
  if name not in self.cache:
   if name in self.precomputed:self.cache[name]=self.precomputed[name]
   else:self.cache[name]=self.encoder.encode([name],query=True)[0]
  return self.cache[name]

 def prepare(self,names):
  names=list(dict.fromkeys(str(n) for n in names if str(n) not in self.cache and str(n) not in self.precomputed))
  if names:
   self.cache.update(zip(names,self.encoder.encode(names,query=True)))

 def retrieve(self,name,k=64):
  scores=self.vectors@self.vector(name)
  # Stable catalog ID tie break independent of other evaluation rows.
  ids=np.argpartition(-scores,min(k,len(scores)-1))[:k]
  ids=ids[np.lexsort((self.ids[ids],-scores[ids]))]
  return self.row_to_catalog[ids].tolist()

 def similarities(self,name,indices):
  rows=self.catalog_to_row[np.asarray(indices,dtype=int)]
  return self.vectors[rows]@self.vector(name)
