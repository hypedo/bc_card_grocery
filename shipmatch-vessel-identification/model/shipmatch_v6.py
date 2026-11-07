"""Independent full-record and name-only candidate views, official references only."""
import numpy as np
from functools import lru_cache
from shipmatch import hangul,norm,clean
from shipmatch_v4 import canonical_row,query_text
from shipmatch_v5 import V5Index,shortlist_v5,FastKnowledge,Candidates

# Generic letter pronunciations, not a vessel-specific alias table.
LETTERS={'에이치':'H','더블유':'W','더블류':'W','에이':'A','비':'B','씨':'C','시':'C','디':'D','이':'E',
 '에프':'F','지':'G','아이':'I','제이':'J','케이':'K','엘':'L','엠':'M','엔':'N','앤':'N','오':'O',
 '피':'P','큐':'Q','알':'R','에스':'S','티':'T','유':'U','브이':'V','엑스':'X','와이':'Y','제트':'Z','제드':'Z'}

@lru_cache(maxsize=16384)
def acronym_name(name):
    parts=[]
    for word in clean(name).split():
        at=0;letters=[]
        while at<len(word):
            hits=[k for k in LETTERS if word.startswith(k,at)]
            if not hits:break
            token=max(hits,key=len);letters.append(LETTERS[token]);at+=len(token)
        parts.append(''.join(letters)+' '+word[at:] if len(letters)>=2 and at>=3 else word)
    return ' '.join(parts)

class AcronymKnowledge(FastKnowledge):
    @lru_cache(maxsize=16384)
    def translate(self,name):return super().translate(acronym_name(name))

def name_row(row):
    r=dict(row);r.update(gross_tonnage=np.nan,international_gross_tonnage=np.nan,flag_name='',vessel_type='')
    return r

def name_document(document):return document.split('\n',1)[0]

def shortlist_v6(row,candidates,features,scores,index):
    original=getattr(candidates,'original',None)
    if original is not None:
        positions=[j for j,c in enumerate(candidates) if c in original]
        subset=Candidates([candidates[j] for j in positions]);subset.base=candidates.base
        old=[positions[j] for j in shortlist_v5(row,subset,features[positions],scores[positions],index=index)]
    else:old=shortlist_v5(row,candidates,features,scores,index=index)
    if getattr(index,'rescue',False) and hangul(row['vessel_name']):
        d=index.auxiliary_scores(row,candidates)
        common=.06*features[:,33]+.1*features[:,34]-.5*features[:,18]
        extra=[int(np.argmax(d-lam*features[:,22]+common)) for lam in [.4,1.2]]
        return list(dict.fromkeys(old+extra))
    if not hangul(row['vessel_name']) or not getattr(index,'expand_names',False):return old
    # Same row's name is judged without its potentially corrupted metadata.
    nf=index.features(name_row(row),candidates)
    ns=index.name_ranker.predict(nf,thread_count=1)
    extra=np.argsort(-ns,kind='stable')[:4].tolist()
    return list(dict.fromkeys(old+extra))

class V6Index(V5Index):
    name_ranker=None
    rescue=False
    def candidates(self,row,limit=480):
        values=super().candidates(row)
        if (self.rescue or getattr(self,'wide',False)) and self.aux_dense is not None and hangul(canonical_row(row)['vessel_name']):
            _,d=self.dense_view(canonical_row(row)['vessel_name'])
            top=np.argpartition(-d,255)[:256];top=top[np.lexsort((self.ids[top],-d[top]))].tolist()
            result=Candidates(list(dict.fromkeys(list(values)+top))[:limit]);result.base=values.base;result.original=frozenset(values)
            return result
        return values
    def direct(self,row):
        r=canonical_row(row)
        aliases=self.knowledge.aliases.get(norm(r['vessel_name']),set())
        if hangul(r['vessel_name']) and len(norm(r['vessel_name']))>=4 and len(aliases)==1:
            vid=next(iter(aliases));i=self.id_to_index.get(vid)
            if i is not None:
                v=self.gt[i];g=r['gross_tonnage'];ig=r['international_gross_tonnage']
                gaps=[abs(x-v)/max(x,v) for x in [g,ig] if np.isfinite(x*v)]
                # Exact official alias plus independent numerical corroboration.
                # Ambiguous aliases and contradictory tonnage use normal ranking.
                if gaps and min(gaps)<=.055:return vid
        return super().direct(r)

def wide_proposals(row,candidates,features,scores,original,index,cap=16):
    used=set(original);out=[]
    d=index.auxiliary_scores(row,candidates)
    channels=[np.argsort(-scores,kind='stable'),np.argsort(-d,kind='stable'),
      np.argsort(-np.maximum(features[:,42],features[:,10]),kind='stable'),np.argsort(-features[:,52],kind='stable')]
    streams=[[int(j) for j in order if int(j) not in used] for order in channels]
    for step in range(cap):
        for stream in streams:
            if step<len(stream):
                j=stream[step]
                if j not in used:used.add(j);out.append(j)
                if len(out)>=cap:return out
    return out
