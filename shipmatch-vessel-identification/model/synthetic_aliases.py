"""Phonetic aliases generated exclusively from released catalog strings.

Only a fixed catalog index is shared. Query caches hold pure function values;
scores never use other evaluation rows or evaluation-population statistics.
"""
from pathlib import Path
from collections import OrderedDict
import json,re,unicodedata
import numpy as np,joblib
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from shipmatch import clean,hangul,norm

ALIAS_FEATURE_NAMES=['alias_cosine','alias_syllable_ratio','alias_jamo_ratio','alias_exact',
                     'alias_length_ratio','alias_containment','alias_common_suffix','alias_margin']

def ko_norm(name):
    s=norm(clean(unicodedata.normalize('NFKC',str(name))))
    for x,y in [('씨','시'),('쓰','스'),('쎄','세'),('앤','엔')]:s=s.replace(x,y)
    return s

def ko_jamo(name):return unicodedata.normalize('NFD',ko_norm(name))

def build_alias_index(catalog,jsonl_paths,path):
    lookup={str(n):[] for n in catalog.vessel_name}
    for i,n in enumerate(catalog.vessel_name.astype(str)):lookup[n].append(i)
    aliases=[];rows=[];seen=set();accepted=0;records=[];targets={}
    for file in jsonl_paths:
        for line in Path(file).read_text().splitlines():
            r=json.loads(line)
            if not r['accepted'] or r['english'] not in lookup:continue
            alias=ko_norm(r['korean'])
            if not alias or not hangul(alias):continue
            if re.search('설명|선박은|입니다|사용자|요청|어시스턴트|조정해야|음역하|옮기면|변환한|제공된|죄송',alias):continue
            if sorted(int(n) for n in re.findall(r'\d+',r['english']))!=sorted(int(n) for n in re.findall(r'\d+',r['korean'])):continue
            records.append((r['english'],alias));targets.setdefault(alias,set()).add(norm(r['english']))
    for english,alias in records:
            # Fixed catalog-wide generation-quality audit, never Test statistics.
            if len(targets[alias])>12:continue
            accepted+=1
            for i in lookup[english]:
                if (i,alias) in seen:continue
                seen.add((i,alias));rows.append(i);aliases.append(alias)
    if not aliases:raise ValueError('No accepted catalog-derived aliases')
    tf=TfidfVectorizer(analyzer='char',ngram_range=(2,4),lowercase=False,dtype=np.float32,sublinear_tf=True,norm='l2',max_features=160000)
    matrix=tf.fit_transform([ko_jamo(s) for s in aliases]).tocsr()
    payload=dict(vectorizer=tf,matrix=matrix,aliases=aliases,rows=np.asarray(rows,np.int32),ids=catalog.vessel_id.astype(str).to_numpy(),source='Official catalog strings transformed by public generic language models; no Train Korean aliases or evaluation input supplied to generator')
    Path(path).parent.mkdir(parents=True,exist_ok=True);joblib.dump(payload,path,compress=3)
    return dict(accepted_outputs=accepted,indexed_aliases=len(aliases),covered_catalog_vessels=len(set(rows)),catalog_vessels=len(catalog),vocabulary=len(tf.vocabulary_),matrix_nonzero=matrix.nnz)

class SyntheticAliases:
    def __init__(self,path,catalog):
        p=joblib.load(path);self.tf=p['vectorizer'];self.matrix=p['matrix'];self.aliases=p['aliases'];self.rows=p['rows'];self.ids=catalog.vessel_id.astype(str).to_numpy()
        if not np.array_equal(self.ids,p['ids']):raise ValueError('Synthetic alias catalog order mismatch')
        self.alias_jamo=[ko_jamo(a) for a in self.aliases];self.by_row={}
        for j,i in enumerate(self.rows):self.by_row.setdefault(int(i),[]).append(j)
        self.cache=OrderedDict()
    def view(self,name):
        name=str(name)
        if name not in self.cache:
            raw=(self.matrix@self.tf.transform([ko_jamo(name)]).T).toarray().ravel()
            scores=np.zeros(len(self.ids),np.float32);np.maximum.at(scores,self.rows,raw)
            k=min(32,len(scores));order=np.argpartition(-scores,k-1)[:k];order=order[np.lexsort((self.ids[order],-scores[order]))]
            self.cache[name]=(scores,order)
            if len(self.cache)>256:self.cache.popitem(last=False)
        self.cache.move_to_end(name);return self.cache[name]
    def features(self,name,candidates):
        if not hangul(name):return np.zeros((len(candidates),len(ALIAS_FEATURE_NAMES)),np.float32)
        scores,order=self.view(name);q=ko_norm(name);j=ko_jamo(name);out=[]
        best=float(scores[order[0]])
        for i in candidates:
            values=[]
            for t in self.by_row.get(int(i),[]):
                a=self.aliases[t];aj=self.alias_jamo[t]
                suffix=0
                for x,y in zip(reversed(q),reversed(a)):
                    if x!=y:break
                    suffix+=1
                values.append([fuzz.ratio(q,a)/100.,fuzz.ratio(j,aj)/100.,float(q==a),min(len(q),len(a))/max(len(q),len(a),1),float(bool(q) and(q in a or a in q)),suffix/max(len(q),len(a),1)])
            v=np.max(values,axis=0).tolist() if values else [0.]*6
            out.append([scores[i]]+v+[float(scores[i])-best])
        return np.asarray(out,np.float32)

    def best_alias(self,name,i):
        positions=self.by_row.get(int(i),[])
        if not positions:return ''
        q=ko_jamo(name)
        t=max(positions,key=lambda t:(fuzz.ratio(q,self.alias_jamo[t]),-t))
        return self.aliases[t]
