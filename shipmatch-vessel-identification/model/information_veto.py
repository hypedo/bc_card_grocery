"""Do not resolve an omitted distinguishing vessel number with phonetics."""
import re
from shipmatch import norm
from shipmatch_v4 import canonical_row
from numeric_veto import NumericVeto

def digits(value):return tuple(str(int(n)) for n in re.findall(r'\d+',str(value)))
def stem(value):return norm(re.sub(r'\d+','',str(value)))

class InformationVeto(NumericVeto):
    def __init__(self,catalog,tolerance=.05):
        super().__init__(catalog,tolerance=tolerance)
        self.names={str(r.vessel_id):str(r.vessel_name) for r in catalog.itertuples()}
    def choose(self,row,baseline,proposal):
        if baseline==proposal:return baseline
        old=self.names.get(str(baseline),'');new=self.names.get(str(proposal),'')
        if old and new and stem(old)==stem(new) and digits(old)!=digits(new):
            if digits(canonical_row(row)['vessel_name'])!=digits(new):return baseline
        return super().choose(row,baseline,proposal)
