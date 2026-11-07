"""Preserve consensus decisions and add at most two row-local catalog proposals."""
import numpy as np
from post_expert_consensus import PostExpert as ConsensusPostExpert
from shipmatch import hangul
from shipmatch_v4 import canonical_row,normalized_text

class PostExpert(ConsensusPostExpert):
    def __init__(self,catalog,index,rough,calibration,complement,dense_weight=2.,retrieval_topk=64,proposals=2,coarse_floor=4.,rescue_advantage=.5,rescue_gate=8.,**kwargs):
        super().__init__(catalog,index,**kwargs)
        self.index=index;self.rough=rough;self.calibration=calibration;self.complement=complement
        self.dense_weight=float(dense_weight);self.retrieval_topk=int(retrieval_topk);self.proposals=int(proposals);self.coarse_floor=float(coarse_floor);self.rescue_advantage=float(rescue_advantage);self.rescue_gate=float(rescue_gate)
        if calibration.numeric.weight!=1.:raise ValueError('Rescue requires the frozen calibrated score')

    def prepare_jobs(self,test,prediction,states):
        eligible=[]
        for state in states:
            position,ids,scores=state[:3];order=np.argsort(-scores,kind='stable')
            if len(order)<2 or scores[order[0]]-scores[order[1]]>self.rescue_gate:continue
            row=canonical_row(test.iloc[position].to_dict())
            if not hangul(row['vessel_name']) or not any(np.isfinite(row[k]) for k in ['gross_tonnage','international_gross_tonnage']):continue
            eligible.append((state,row))
        if not eligible:return []
        self.complement.prepare([normalized_text(row['vessel_name']) for _,row in eligible]);jobs=[]
        for state,row in eligible:
            position,old_ids,scores=state[:3];name=normalized_text(row['vessel_name']);similarity=self.complement.vectors@self.complement.vector(name);k=min(self.retrieval_topk,len(similarity));top=np.argpartition(-similarity,k-1)[:k];top=top[np.lexsort((self.complement.ids[top],-similarity[top]))]
            chosen=[str(self.complement.ids[j]) for j in top if str(self.complement.ids[j]) not in old_ids and self.veto.choose(row,prediction[position],str(self.complement.ids[j]))!=prediction[position]]
            if not chosen:continue
            indices=np.asarray([self.index.id_to_index[v] for v in chosen]);features=self.index.features(row,indices);phonetic=self.calibration.evidence.features(row['vessel_name'],indices,scores=self.index.phonetic_view(row['vessel_name'])[0]);blend=self.calibration.numeric.blend
            base=(1-blend)*self.rough.old.predict(features[:,:58],thread_count=6)+blend*self.calibration.numeric.predict_features(row,indices,features,phonetic,thread_count=6)+self.dense_weight*self.index.aux_dense.similarities(name,indices)
            rank=np.lexsort((np.asarray(chosen),-base));selected=[int(j) for j in rank[:self.proposals] if base[j]>=max(state[5])-self.coarse_floor]
            if not selected:continue
            ids=[chosen[j] for j in selected];pairs=[(state[6],self.index.document_for(row,self.index.id_to_index[v])) for v in ids];order=np.argsort(-scores,kind='stable')[:self.cap];old_pairs=[(state[6],state[4][t]) for t in order]
            jobs.append({'position':position,'row':row,'ids':ids,'pairs':pairs,'base_scores':base[selected],'old_pairs':old_pairs,'old_scores':np.asarray(scores)[order]})
        return jobs

    def apply(self,test,prediction,states,neural):
        result=super().apply(test,prediction,states,neural);jobs=self.prepare_jobs(test,prediction,states)
        if not jobs:return result
        pairs=[]
        for job in jobs:
            job['at']=len(pairs);pairs.extend(job['pairs']);job['old_at']=len(pairs);pairs.extend(job['old_pairs'])
        primary=neural.score(pairs);expert=neural.expert_score(pairs)
        if not np.isfinite(primary).all() or not np.isfinite(expert).all():raise RuntimeError('Nonfinite complementary candidate score')
        for job in jobs:
            at=job['at'];n=len(job['pairs']);old_at=job['old_at'];old_n=len(job['old_pairs'])
            new=job['base_scores']+.5*primary[at:at+n]+.5*self.weight*(expert[at:at+n]-primary[at:at+n]);old=job['old_scores']+.5*self.weight*(expert[old_at:old_at+old_n]-primary[old_at:old_at+old_n]);winner=int(np.argmax(new));proposal=job['ids'][winner];position=job['position']
            if new[winner]-max(old)>self.rescue_advantage and self.evidence.choose(job['row'],prediction[position],proposal)==proposal:result[position]=proposal
        return result
