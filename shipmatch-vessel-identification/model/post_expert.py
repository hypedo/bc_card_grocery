"""Conservative4B expert residual, with independent target-specific caches."""
import numpy as np
from shipmatch import hangul
from shipmatch_v4 import canonical_row
from information_veto import InformationVeto

class PostExpert:
    def __init__(self,catalog,weight=.5,gate=8.,cap=4,tolerance=.05):
        self.weight=float(weight);self.gate=float(gate);self.cap=int(cap);self.veto=InformationVeto(catalog,tolerance=tolerance)
    def apply(self,test,prediction,states,neural):
        result=list(prediction);pairs=[];jobs=[]
        for state in states:
            position,ids,scores=state[:3];order=np.argsort(-scores,kind='stable')[:self.cap]
            if len(order)<2 or scores[order[0]]-scores[order[1]]>self.gate:continue
            row=canonical_row(test.iloc[position].to_dict())
            if not hangul(row['vessel_name']) or not any(np.isfinite(row[k]) for k in ['gross_tonnage','international_gross_tonnage']):continue
            selected=[ids[t] for t in order];baseline=result[position]
            # A whole-job skip is valid only when every possible proposal is vetoed.
            if not any(self.veto.choose(row,baseline,v)!=baseline for v in selected):continue
            jobs.append((position,row,selected,np.asarray(scores)[order],len(pairs)))
            pairs.extend((state[6],state[4][t]) for t in order)
        if not pairs:return result
        primary=neural.score(pairs);expert=neural.expert_score(pairs)
        if not np.isfinite(primary).all() or not np.isfinite(expert).all():raise RuntimeError('Nonfinite expert residual')
        delta=.5*(np.asarray(expert,np.float32)-np.asarray(primary,np.float32))
        for position,row,ids,scores,at in jobs:
            proposal=ids[int(np.argmax(scores+self.weight*delta[at:at+len(ids)]))]
            result[position]=self.veto.choose(row,result[position],proposal)
        return result
