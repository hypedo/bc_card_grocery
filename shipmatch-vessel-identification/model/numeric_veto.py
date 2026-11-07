"""Conservative per-row veto for phonetic changes that worsen tonnage evidence."""
import numpy as np
from shipmatch_v4 import canonical_row,numeric

class NumericVeto:
    def __init__(self,catalog,tolerance=.05):
        self.gt={str(r.vessel_id):numeric(r.gross_tonnage) for r in catalog.itertuples()};self.tolerance=tolerance
    def choose(self,row,baseline,proposal):
        if baseline==proposal:return baseline
        r=canonical_row(row);values=[r[k] for k in ['gross_tonnage','international_gross_tonnage'] if np.isfinite(r[k])]
        if not values:return baseline
        old=self.gt.get(str(baseline),np.nan);new=self.gt.get(str(proposal),np.nan)
        if not np.isfinite(new):return baseline
        def gap(gt):return min(abs(v-gt)/max(v,gt) for v in values) if np.isfinite(gt) else 1.
        return proposal if gap(new)<=gap(old)+self.tolerance else baseline
