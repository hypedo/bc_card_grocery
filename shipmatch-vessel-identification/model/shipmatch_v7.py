"""Catalog-derived Korean alias evidence with a Korean-only augmented ranker."""
from pathlib import Path
import numpy as np
from catboost import CatBoostRanker
from shipmatch import hangul
from shipmatch_v4 import V4_FEATURE_NAMES,canonical_row
from shipmatch_v5 import Candidates
from shipmatch_v6 import V6Index,shortlist_v6
from synthetic_aliases import SyntheticAliases,ALIAS_FEATURE_NAMES

V7_FEATURE_NAMES=V4_FEATURE_NAMES+ALIAS_FEATURE_NAMES

class HybridRough:
    feature_names_=V7_FEATURE_NAMES
    def __init__(self,old_path,new_path,blend=1.):
        self.blend=blend
        self.old=CatBoostRanker();self.old.load_model(str(old_path))
        self.new=CatBoostRanker();self.new.load_model(str(new_path))
        if self.old.feature_names_!=V4_FEATURE_NAMES or self.new.feature_names_!=V7_FEATURE_NAMES:raise ValueError('Ranker feature schema mismatch')
    def predict(self,X,thread_count=6):
        X=np.asarray(X);k=X[:,0]>.5;out=np.empty(len(X),np.float64)
        if k.any():out[k]=self.blend*self.new.predict(X[k],thread_count=thread_count)+(1-self.blend)*self.old.predict(X[k,:len(V4_FEATURE_NAMES)],thread_count=thread_count)
        if (~k).any():out[~k]=self.old.predict(X[~k,:len(V4_FEATURE_NAMES)],thread_count=thread_count)
        return out

class V7Index(V6Index):
    def __init__(self,*args,alias_path,**kwargs):
        super().__init__(*args,**kwargs);self.synthetic=SyntheticAliases(alias_path,self.catalog)
    def candidates(self,row,limit=370):
        old=super().candidates(row)
        r=canonical_row(row)
        if not hangul(r['vessel_name']):return old
        scores,order=self.synthetic.view(r['vessel_name']);extra=[int(i) for i in order[:24] if scores[i]>.2]
        out=Candidates(list(dict.fromkeys(list(old)+extra))[:limit]);out.base=old.base|frozenset(extra)
        return out
    def features(self,row,candidates):
        base=super().features(row,candidates);r=canonical_row(row)
        return np.column_stack([base,self.synthetic.features(r['vessel_name'],candidates)]).astype(np.float32)

    def document_for(self,row,i):
        document=self.documents[i]
        if not getattr(self,'alias_documents',False) or not hangul(row['vessel_name']):return document
        alias=self.synthetic.best_alias(row['vessel_name'],i)
        if not alias:return document
        first,rest=document.split('\n',1)
        return first+'; Korean phonetic: '+alias+'\n'+rest

def shortlist_v7(row,candidates,features,scores,index):
    old=shortlist_v6(row,candidates,features,scores,index)
    if not hangul(canonical_row(row)['vessel_name']):return old
    alias=np.argsort(-features[:,58],kind='stable')[:2]
    extra=[int(j) for j in alias if features[j,58]>=.5]
    return list(dict.fromkeys(old+extra))
