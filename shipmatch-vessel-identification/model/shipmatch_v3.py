"""Multiview official-catalog retrieval and corruption-robust pair features."""
from pathlib import Path
import re,numpy as np,joblib
from rapidfuzz import fuzz,process
from sklearn.feature_extraction.text import TfidfVectorizer
from shipmatch import *

V3_FEATURE_NAMES=ENHANCED_FEATURE_NAMES+['dense_name','char_tfidf','flag_exact','type_exact','igt_dwt_gap','name_containment','translated_containment']

class RobustIndex(EnhancedIndex):
 def __init__(self,catalog,knowledge=None,dense=None,sparse_path=None,auxiliary=False):
  super().__init__(catalog,knowledge);self.dense=dense;self._views={};self.auxiliary=auxiliary
  self.aux_phons=[auxiliary_phon(n) for n in self.raw] if auxiliary else []
  self.weak_phons=[weak_phon(n) for n in self.raw] if auxiliary else []
  self._aux_cache={}
  if sparse_path and Path(sparse_path).exists():
   saved=joblib.load(sparse_path);self.tfidf=saved['vectorizer'];self.matrix=saved['matrix'];saved_ids=saved['ids']
   lookup={v:i for i,v in enumerate(saved_ids)};self.matrix=self.matrix[[lookup[v] for v in self.ids]]
  else:
   self.tfidf=TfidfVectorizer(analyzer='char',ngram_range=(2,4),lowercase=False,dtype=np.float32,norm='l2',sublinear_tf=True)
   self.matrix=self.tfidf.fit_transform(self.softphons)
 def save_sparse(self,path):joblib.dump({'vectorizer':self.tfidf,'matrix':self.matrix,'ids':self.ids},path,compress=3)
 def views(self,name):
  if name not in self._views:
   vec=self.tfidf.transform([softphon(name)]);scores=(self.matrix@vec.T).toarray().ravel()
   ids=np.argpartition(-scores,47)[:48];ids=ids[np.lexsort((self.ids[ids],-scores[ids]))]
   dense=self.dense.retrieve(name,64) if self.dense is not None and hangul(name) else []
   self._views[name]=(ids.tolist(),scores,dense)
  return self._views[name]
 def candidates(self,row,limit=120):
  name=text(row.get('vessel_name'));old=super().candidates(row,limit=70)
  if not hangul(name):return old
  sparse,_,dense=self.views(name)
  # Preserve candidates from independent retrieval channels despite noisy numeric metadata.
  extra=[]
  if self.auxiliary:
   if name not in self._aux_cache:
    q=softphon(name).replace('J','S')
    weak=process.extract(q,self.weak_phons,scorer=fuzz.ratio,limit=256)
    weak=sorted(weak,key=lambda z:(-(z[1]+.2*fuzz.ratio(roman(name),self.romans[z[2]])),self.ids[z[2]]))[:16]
    acronym=process.extract(q,self.aux_phons,scorer=fuzz.ratio,limit=16)
    self._aux_cache[name]=list(dict.fromkeys([z[2] for z in weak]+[z[2] for z in acronym]))
   extra=self._aux_cache[name]
  out=list(dict.fromkeys(old[:60]+extra+dense[:48]+sparse[:32]))
  return out[:limit]
 def features(self,row,candidates):
  f=super().features(row,candidates);name=text(row.get('vessel_name'));ko=hangul(name)
  dense=self.dense.similarities(name,candidates) if self.dense is not None and ko else np.zeros(len(candidates),np.float32)
  scores=self.views(name)[1] if ko else np.zeros(len(self.ids),np.float32)
  ig=pos(row.get('international_gross_tonnage'));q=norm(clean(name));t=self.knowledge.translate(name)
  extra=[]
  for i,ds in zip(candidates,dense):
   c=self.names[i];d=self.dwt[i]
   extra.append([ds,scores[i],norm(row.get('flag_name'))==norm(self.flags[i]) and bool(norm(self.flags[i])),
    norm(row.get('vessel_type'))==norm(self.types[i]) and bool(norm(self.types[i])),
    abs(ig-d)/max(ig,d) if np.isfinite(ig*d) else 1.,bool(q) and (q in c or c in q),bool(t) and (t in c or c in t)])
  return np.column_stack([f,np.asarray(extra,np.float32)])
 def direct(self,row):
  v=super().direct(row)
  if not v:return None
  i=self.id_to_index[v];f=EnhancedIndex.features(self,row,[i])[0]
  # A known call remains useful, but cannot override contradictory name AND numeric evidence.
  name=max(f[2],f[6],f[10],f[42]);numeric=f[22]
  return v if name>=.64 or numeric<=.065 else None

def shortlist(row,indices,features,scores,index,k=8):
 order=np.argsort(-scores,kind='stable').tolist()
 if not hangul(row.get('vessel_name')):return order[:1]
 chosen=order[:k]
 # Reserve one dense candidate and one purely phonetic candidate outside the learned top ranks.
 dense=np.argsort(-features[:,45],kind='stable')[:2]
 phon=np.argsort(-np.maximum(features[:,42],features[:,10]),kind='stable')[:2]
 chosen=list(dict.fromkeys(chosen+dense.tolist()+phon.tolist()))[:k+4]
 if getattr(index,'auxiliary',False):
  aux=index._aux_cache.get(text(row.get('vessel_name')),[])
  reserve=aux[:16] if len(softphon(text(row.get('vessel_name'))))<=4 else aux[16:18]
  lookup={int(v):j for j,v in enumerate(indices)}
  chosen=list(dict.fromkeys(chosen+[lookup[i] for i in reserve if i in lookup]))
 return chosen

# Generic spoken-letter and weak-final-R views, not a vessel alias dictionary.
LETTER_NAMES=dict(zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ',['에이','비','씨','디','이','에프','지','에이치','아이','제이','케이','엘','엠','엔','오','피','큐','알','에스','티','유','브이','더블유','엑스','와이','제트']))
def auxiliary_phon(name):
 words=clean(name).split();raw_words=text(name).split();parts=[]
 for j,w in enumerate(words):
  raw=raw_words[j] if j<len(raw_words) else w
  if raw.isupper() and w.isalpha() and 1<=len(w)<=4:parts.append(''.join(LETTER_NAMES.get(c,c) for c in w))
  else:parts.append(w)
 return softphon(' '.join(parts)).replace('J','S')
def weak_phon(name):
 p=softphon(name).replace('J','S')
 if norm(name).endswith('ER') and p.endswith('L'):p=p[:-1]
 return p
