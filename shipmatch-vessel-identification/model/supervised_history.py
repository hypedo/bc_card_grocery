"""Corroborated identity evidence from official Train and catalog phonetics."""
from collections import defaultdict
import re
import numpy as np
from shipmatch import hangul,norm
from shipmatch_v4 import canonical_row
from shipmatch_v17 import V17Index
from synthetic_aliases import SyntheticAliases

class CallEvidence:
    def __init__(self,train,catalog,synthetic=None,alias_path=None):
        self.synthetic=synthetic if synthetic is not None else SyntheticAliases(alias_path,catalog)
        self.rows={str(v):i for i,v in enumerate(catalog.vessel_id)}
        self.names=catalog.vessel_name.astype(str).tolist();self.gt=catalog.gross_tonnage.to_numpy(float)
        calls=defaultdict(set);self.aliases=defaultdict(set);self.tonnages=defaultdict(list)
        for r in train.to_dict('records'):
            r=canonical_row(r);vid=str(r['target_vessel_id'])
            if vid not in self.rows:continue
            if r['call_sign']:calls[r['call_sign']].add(vid)
            if hangul(r['vessel_name']):self.aliases[norm(r['vessel_name'])].add(vid)
            for field in ['gross_tonnage','international_gross_tonnage']:
                if np.isfinite(r[field]):self.tonnages[vid].append(r[field])
        self.calls={k:next(iter(v)) for k,v in calls.items() if len(v)==1}
        self.tonnages={k:np.unique(v) for k,v in self.tonnages.items()}

    def predict(self,row):
        r=canonical_row(row);name=r['vessel_name']
        if not hangul(name) or len(norm(name))<4:return None
        vid=self.calls.get(r['call_sign'])
        if vid is None:return None
        # Two independently represented identity fields agree in official Train.
        known=self.aliases.get(norm(name),set())
        if known:
            return vid if known=={vid} else None
        i=self.rows[vid]
        qn=[int(v) for v in re.findall(r'\d+',name)];cn=[int(v) for v in re.findall(r'\d+',self.names[i])]
        if (qn or cn) and qn!=cn:return None
        evidence=self.synthetic.features(name,[i])[0];cosine,jamo=float(evidence[0]),float(evidence[2])
        values=[r[f] for f in ['gross_tonnage','international_gross_tonnage'] if np.isfinite(r[f])]
        if not values:return vid if cosine>=.85 and jamo>=.92 else None
        references=np.r_[self.gt[i],self.tonnages.get(vid,[])];references=references[np.isfinite(references)&(references>0)]
        if not len(references):return None
        gap=min(float(np.min(np.abs(references-v)/np.maximum(references,v))) for v in values)
        return vid if gap<=.055 and cosine>=.72 and jamo>=.88 else None

class HistoryIndex(V17Index):
    def __init__(self,*args,train_reference,**kwargs):
        super().__init__(*args,**kwargs)
        self.history=CallEvidence(train_reference,self.catalog,synthetic=self.synthetic)
    def direct(self,row):
        value=self.history.predict(row)
        return value if value is not None else super().direct(row)
