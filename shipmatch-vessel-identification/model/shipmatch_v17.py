"""Fixed compositional phonetic proposals; no evaluation-population state."""
from collections import OrderedDict
import numpy as np
from shipmatch import hangul
from shipmatch_v4 import canonical_row
from shipmatch_v5 import Candidates
from shipmatch_v7 import V7Index,shortlist_v7
from learned_translit import TranslitIndex

class V17Index(V7Index):
    def __init__(self,*args,translit_aug_path,**kwargs):
        super().__init__(*args,**kwargs)
        self.translit_aug=TranslitIndex(translit_aug_path,self.catalog)
        self.phonetic_cache=OrderedDict()
    def phonetic_view(self,name):
        name=str(name)
        if name not in self.phonetic_cache:
            scores=np.asarray(self.translit_aug.scores(name),dtype=np.float32)
            k=min(64,len(scores));order=np.argpartition(-scores,k-1)[:k]
            order=order[np.lexsort((self.ids[order],-scores[order]))]
            self.phonetic_cache[name]=(scores,order)
            if len(self.phonetic_cache)>256:self.phonetic_cache.popitem(last=False)
        self.phonetic_cache.move_to_end(name)
        return self.phonetic_cache[name]
    def phonetic_scores(self,row,candidates):
        r=canonical_row(row)
        if not hangul(r['vessel_name']):return np.zeros(len(candidates),np.float32)
        return self.phonetic_view(r['vessel_name'])[0][candidates]
    def candidates(self,row,limit=420):
        old=super().candidates(row);r=canonical_row(row)
        if not hangul(r['vessel_name']):return old
        scores,order=self.phonetic_view(r['vessel_name'])
        extra=[int(i) for i in order[:24] if scores[i]>.2]
        result=Candidates(list(dict.fromkeys(list(old)+extra))[:limit])
        result.base=old.base|frozenset(extra)
        result.v7_original=frozenset(old)
        result.v7_base=old.base
        return result

def shortlist_v17(row,candidates,features,scores,index):
    original=getattr(candidates,'v7_original',None)
    if original is not None:
        positions=[j for j,i in enumerate(candidates) if i in original]
        subset=Candidates([candidates[j] for j in positions]);subset.base=candidates.v7_base
        old=[positions[j] for j in shortlist_v7(row,subset,features[positions],scores[positions],index)]
    else:
        old=shortlist_v7(row,candidates,features,scores,index)
    if not hangul(canonical_row(row)['vessel_name']):return old
    phonetic=index.phonetic_scores(row,candidates)
    order=np.argsort(-phonetic,kind='stable')[:2]
    return list(dict.fromkeys(old+[int(j) for j in order if phonetic[j]>=.5]))
