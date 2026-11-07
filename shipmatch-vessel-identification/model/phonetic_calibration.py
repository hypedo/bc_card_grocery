"""Compositional phonetic confidence computed against the fixed catalog."""
from collections import OrderedDict
import numpy as np
from shipmatch import hangul
from learned_translit import TranslitIndex
PHONETIC_FEATURE_NAMES=['compositional_cosine','compositional_margin','compositional_reciprocal_rank']
class PhoneticEvidence:
    def __init__(self,path,catalog,index=None):
        self.index=index if index is not None else TranslitIndex(path,catalog);self.cache=OrderedDict()
    def features(self,name,candidates,scores=None):
        if not hangul(name):return np.zeros((len(candidates),3),np.float32)
        name=str(name)
        if name not in self.cache:
            values=self.index.scores(name) if scores is None else scores;k=min(128,len(values));order=np.argpartition(-values,k-1)[:k];order=order[np.lexsort((self.index.ids[order],-values[order]))]
            self.cache[name]=(values,float(values[order[0]]),{int(i):1./(r+1) for r,i in enumerate(order)})
            if len(self.cache)>256:self.cache.popitem(last=False)
        self.cache.move_to_end(name);values,best,ranks=self.cache[name]
        return np.column_stack([values[candidates],values[candidates]-best,[ranks.get(int(i),0.) for i in candidates]]).astype(np.float32)

class CalibratedRough:
    """Change final Korean scores after the preserved shortlist is selected."""
    def __init__(self,path,index,weight=1.,blend=.5,ipa=None):
        from catboost import CatBoostRanker
        from shipmatch_v7 import V7_FEATURE_NAMES
        self.model=CatBoostRanker();self.model.load_model(str(path))
        names=V7_FEATURE_NAMES+PHONETIC_FEATURE_NAMES
        if ipa is not None:
            from ipa_features import IPA_FEATURE_NAMES
            names=names+IPA_FEATURE_NAMES
        if self.model.feature_names_!=names:raise ValueError('Phonetic ranker schema mismatch')
        self.ipa=ipa;self.ipa_enabled=ipa is not None
        self.index=index;self.evidence=PhoneticEvidence(None,index.catalog,index=index.translit_aug);self.weight=weight;self.blend=blend
    def feature_matrix(self,row,candidates,features,phonetic=None):
        if phonetic is None:
            phonetic=self.evidence.features(row['vessel_name'],candidates,scores=self.index.phonetic_view(row['vessel_name'])[0])
        parts=[features,phonetic]
        if self.ipa is not None:parts.append(self.ipa.features(row,candidates,features,phonetic))
        return np.column_stack(parts).astype(np.float32)
    def predict_features(self,row,candidates,features,phonetic=None,thread_count=6):
        return self.model.predict(self.feature_matrix(row,candidates,features,phonetic),thread_count=thread_count)
    def apply(self,row,candidates,features,scores,order,rough):
        if not hangul(row['vessel_name']):return scores
        chosen=np.asarray(order,dtype=np.int64);ids=np.asarray(candidates)[chosen];x=features[chosen]
        phonetic=self.evidence.features(row['vessel_name'],ids,scores=self.index.phonetic_view(row['vessel_name'])[0])
        legacy=rough.old.predict(x[:,:58],thread_count=6)
        proposal=(1-self.blend)*legacy+self.blend*self.predict_features(row,ids,x,phonetic,thread_count=6)
        result=scores.copy();result[chosen]=(1-self.weight)*result[chosen]+self.weight*proposal
        return result
