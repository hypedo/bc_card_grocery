"""Score-preserving CPU optimizations of the v4 official-data matcher."""
from functools import lru_cache
import numpy as np
from shipmatch import Knowledge, text, hangul
from shipmatch_v4 import V4Index, canonical_row, shortlist_v4

class FastKnowledge(Knowledge):
    @lru_cache(maxsize=16384)
    def translate(self, name):
        return super().translate(name)

    def probability(self, field, q, c):
        return self._probability(field, text(q), text(c))

    @lru_cache(maxsize=16384)
    def _probability(self, field, q, c):
        counts = getattr(self, field + '_counts').get(q, {})
        return counts.get(c, 0) / max(sum(counts.values()), 1)

class Candidates(list):
    pass

class V5Index(V4Index):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metadata_codes={}
        for field, source in [('flag', self.flags), ('type', self.types)]:
            labels, codes=np.unique(np.asarray([text(v) for v in source]), return_inverse=True)
            self._metadata_codes[field]=(labels, codes)

    @lru_cache(maxsize=96)
    def metadata_probability(self, field, value):
        counts=getattr(self.knowledge, field+'_counts').get(text(value), {})
        total=max(sum(counts.values()),1)
        labels,codes=self._metadata_codes[field]
        return np.asarray([counts.get(v,0)/total for v in labels],np.float32)[codes]

    def numeric_candidates(self, row):
        g,ig=row['gross_tonnage'],row['international_gross_tonnage']
        if not np.isfinite(g) and not np.isfinite(ig):return []
        def gap(v):return np.where(np.isfinite(v*self.gt),np.abs(v-self.gt)/np.maximum(v,self.gt),1.)
        distance=np.minimum(gap(g),gap(ig))
        score=-6*distance+.35*self.metadata_probability('type',row['vessel_type'])+.2*self.metadata_probability('flag',row['flag_name'])
        out=np.argpartition(-score,47)[:48]
        out=out[np.lexsort((self.ids[out],-score[out]))].tolist()
        # Preserve stable full-sort ties, including large equal-distance blocks.
        threshold=np.partition(distance,7)[7]
        lower=np.flatnonzero(distance<threshold)
        equal=np.flatnonzero(distance==threshold)[:max(0,8-len(lower))]
        pure=np.r_[lower,equal]
        pure=pure[np.lexsort((pure,distance[pure]))][:8].tolist()
        return list(dict.fromkeys(pure+out))

    def candidates(self, row, limit=340):
        row=canonical_row(row);base=self.base_candidates(row)
        values=base
        if self.aux_dense is not None and hangul(row['vessel_name']):
            dense,_=self.dense_view(row['vessel_name']);values=list(dict.fromkeys(base+dense))[:limit]
        result=Candidates(values);result.base=frozenset(base)
        return result


def shortlist_v5(row,candidates,features,scores,k=6,index=None):
    if index is None or index.aux_dense is None or not hangul(row['vessel_name']):
        return shortlist_v4(row,candidates,features,scores,k=k)
    base=getattr(candidates,'base',None)
    if base is None:base=set(index.base_candidates(row))
    positions=[j for j,c in enumerate(candidates) if c in base]
    chosen=[positions[j] for j in shortlist_v4(row,[candidates[j] for j in positions],features[positions],scores[positions],k=k)]
    order=np.argsort(-scores,kind='stable').tolist()
    dense=np.argsort(-index.auxiliary_scores(row,candidates),kind='stable')[:4].tolist()
    return list(dict.fromkeys(chosen+dense+order[:2]))[:18]
