"""Row-local veto when every independent phonetic view favors the old match."""
import numpy as np
from information_veto import InformationVeto
from shipmatch_v4 import canonical_row

class EvidenceVeto(InformationVeto):
    def __init__(self,catalog,index,tolerance=.05):
        super().__init__(catalog,tolerance=tolerance)
        self.index=index

    def choose(self,row,baseline,proposal):
        proposal=super().choose(row,baseline,proposal)
        if baseline==proposal:return baseline
        row=canonical_row(row)
        indices=[self.index.id_to_index[baseline],self.index.id_to_index[proposal]]
        features=self.index.features(row,indices)
        compositional=self.index.phonetic_view(row['vessel_name'])[0][indices]
        evidence=np.column_stack([features[:,52],features[:,58],features[:,60],compositional])
        if not np.isfinite(evidence).all():raise RuntimeError('Nonfinite phonetic consensus evidence')
        dominated=np.all(evidence[1]<=evidence[0]+1e-6) and np.any(evidence[1]<evidence[0]-1e-6)
        values=[row[k] for k in ['gross_tonnage','international_gross_tonnage'] if np.isfinite(row[k])]
        def gap(vid):
            gt=self.gt.get(str(vid),np.nan)
            return min(abs(v-gt)/max(v,gt) for v in values) if values and np.isfinite(gt) else 1.
        numeric_better=gap(proposal)<gap(baseline)-.01
        return baseline if dominated and not numeric_better else proposal
