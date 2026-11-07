"""Preserve V32 scoring and cheaply check only its actual changed decisions."""
from post_expert import PostExpert as PrimaryPostExpert
from evidence_veto import EvidenceVeto

class PostExpert(PrimaryPostExpert):
    def __init__(self,catalog,index,**kwargs):
        super().__init__(catalog,**kwargs)
        self.evidence=EvidenceVeto(catalog,index,tolerance=kwargs.get('tolerance',.05))

    def apply(self,test,prediction,states,neural):
        result=super().apply(test,prediction,states,neural)
        for position,(baseline,proposal) in enumerate(zip(prediction,list(result))):
            if baseline!=proposal:
                result[position]=self.evidence.choose(test.iloc[position].to_dict(),baseline,proposal)
        return result
