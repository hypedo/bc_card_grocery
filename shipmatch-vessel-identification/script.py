"""Offline expanded-phonetic-evidence row-independent 4B vessel matching entry."""
import os
for _key in ["OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]:os.environ[_key]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"
from pathlib import Path
import sys,argparse,time,json
import numpy as np,pandas as pd
from catboost import CatBoostRanker
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE/'model'))
from shipmatch import hangul
from shipmatch_v5 import FastKnowledge
from shipmatch_v6 import AcronymKnowledge as Knowledge, name_row,wide_proposals,name_document
from shipmatch_v7 import HybridRough, V7_FEATURE_NAMES
from supervised_history import HistoryIndex as V4Index
from shipmatch_v17 import shortlist_v17 as shortlist_v6
from shipmatch_v4 import V4_FEATURE_NAMES,canonical_row,query_text,joint_route,normalized_text
from threadpoolctl import threadpool_limits
_BLAS_LIMIT=threadpool_limits(limits=1,user_api="blas")
from prefix_v6 import PrefixV6 as StructuredReranker
from dense_names import DenseNames
from adapted_dense import AdaptedNameEncoder
from phonetic_calibration import CalibratedRough
from numeric_calibration import NumericCalibratedRough,build_calibration
from post_expert_rescue import PostExpert


class LazyJoint:
    def __init__(self,path,adapter,name_adapter=None,expert_adapter=None):
        self.path=path;self.adapter=adapter;self.name_adapter=name_adapter;self.expert_adapter=expert_adapter;self.loaded=None
        self.active=None;self.caches={};self.full_caches={};self.base_qv=None
    def switch(self,target):
        if self.loaded is None:
            self.loaded=StructuredReranker(self.path,batch_size=32,max_length=256)
            if self.name_adapter is not None or self.expert_adapter is not None:
                self.base_qv={n:p.detach().cpu().clone() for n,p in self.loaded.model.named_parameters() if n.endswith(('q_proj.weight','v_proj.weight'))}
        if self.active==target:return
        if self.active is not None:
            self.caches[self.active]=self.loaded.cache;self.full_caches[self.active]=getattr(self.loaded,'full_cache',{})
            self.loaded.cache={};self.loaded.full_cache={}
            if self.base_qv is not None:
                with self.loaded.torch.no_grad():
                    for n,p in self.loaded.model.named_parameters():
                        if n in self.base_qv:p.copy_(self.base_qv[n])
        self.loaded.apply_adapter(self.expert_adapter if target=='expert' else self.name_adapter if target=='name' else self.adapter)
        self.loaded.tail_length=32 if target=='name' else 64
        self.loaded.cache=self.caches.setdefault(target,{})
        self.loaded.full_cache=self.full_caches.setdefault(target,{})
        self.active=target
    def score(self,pairs):
        self.switch('joint')
        return self.loaded.score(pairs)
    def full_score(self,pairs):
        self.switch('joint')
        return self.loaded.full_score(pairs)
    def expert_score(self,pairs):
        if self.expert_adapter is None:raise RuntimeError('Missing expert adapter')
        self.switch('expert');return self.loaded.score(pairs)
    def expert_full_score(self,pairs):
        if self.expert_adapter is None:raise RuntimeError('Missing expert adapter')
        self.switch('expert');return self.loaded.full_score(pairs)
    def name_score(self,pairs):
        if self.name_adapter is None:return self.score(pairs)
        self.switch('name');return self.loaded.score(pairs)
    def name_full_score(self,pairs):
        if self.name_adapter is None:return self.full_score(pairs)
        self.switch('name');return self.loaded.full_score(pairs)
    def clear_cache(self):
        for v in self.caches.values():v.clear()
        for v in self.full_caches.values():v.clear()
        if self.loaded is not None:self.loaded._instruction_states=None


class LazyEncoder:
    def __init__(self,path,adapter):self.path=path;self.adapter=adapter;self.loaded=None
    def encode(self,names,query=True):
        if self.loaded is None:self.loaded=AdaptedNameEncoder(self.path,self.adapter,batch_size=32)
        return self.loaded.encode(names,query=query)


def predict_frame(test,index,rough,neural,alpha=.5,batch_size=128,english_gate=4.,beta=4.,routing='old',gamma=.5,name_cap=8,name_gate=6.,cascade_cap=0,cascade_gate=4.,cascade_advantage=1.,alias_beta=0.,phonetic_beta=0.,calibration=None,post_expert=None):
    if cascade_cap and gamma:raise ValueError('Cascade and additive expert are separate methods')
    if post_expert is not None and (cascade_cap or gamma):raise ValueError('Post-expert requires the frozen primary decision')
    result=[];pairs=[];jobs=[];precise_pairs=[];precise_jobs=[];states=[]
    index.name_ranker=rough;index.wide=bool(cascade_cap)
    def flush():
        if not pairs:return
        llm=neural.score(pairs)
        for position,ids,s,at,nquery,rescue in jobs:
            combined=s+alpha*llm[at:at+len(ids)]
            state=[position,ids,combined,nquery,[pairs[at+t][1] for t in range(len(ids))],s.copy(),pairs[at][0],rescue]
            contenders=np.flatnonzero(combined>=combined.max()-.5)
            if len(contenders)>1 and hasattr(neural,'full_score'):
                precise_jobs.append((state,contenders,s[contenders].copy(),len(precise_pairs)))
                precise_pairs.extend(pairs[at+int(t)] for t in contenders)
            states.append(state)
        pairs.clear();jobs.clear()
    if index.aux_dense is not None:
        names=[normalized_text(n) for n in test.vessel_name.tolist()]
        index.aux_dense.prepare([n for n in names if hangul(n)])
    from queue import Queue,Full
    from threading import Thread,Event
    queue=Queue(maxsize=16);stop=Event()
    def put(value):
        while not stop.is_set():
            try:queue.put(value,timeout=.1);return
            except Full:continue
    def prepare():
        try:
            for start in range(0,len(test),batch_size):
                if stop.is_set():break
                rows=[canonical_row(r) for r in test.iloc[start:start+batch_size].to_dict('records')]
                output=[None]*len(rows);pending=[];local_jobs=[]
                for j,r in enumerate(rows):
                    v=index.direct(r)
                    if v:output[j]=v
                    else:pending.append(j)
                prepared=[]
                for j in pending:
                    r=rows[j];c=index.candidates(r);f=index.features(r,c);prepared.append((j,c,f))
                flat=rough.predict(np.concatenate([f for _,_,f in prepared]),thread_count=6) if prepared else []
                at=0
                for j,c,f in prepared:
                    r=rows[j];s=flat[at:at+len(c)];rough_s=s;at+=len(c)
                    order=shortlist_v6(r,c,f,s,index)
                    if calibration is not None:s=calibration.apply(r,c,f,s,order,rough)
                    s=s+beta*index.auxiliary_scores(r,c)+alias_beta*(.5*f[:,58]+.5*f[:,60])+phonetic_beta*index.phonetic_scores(r,c)
                    if getattr(calibration,'ipa_enabled',False):order=sorted(order,key=lambda t:(-s[t],int(t)))
                    gap=s[order[0]]-s[order[1]] if len(order)>1 else np.inf
                    if alpha==0 or not joint_route(r,f[order[0]],gap,english_gate,routing):
                        output[j]=index.ids[c[order[0]]];continue
                    query=query_text(r);nquery=query_text(name_row(r)) if hangul(r['vessel_name']) and (gamma or cascade_cap) else None
                    rescue=None
                    if cascade_cap and nquery is not None:
                        wide=wide_proposals(r,c,f,rough_s,order,index,cascade_cap)
                        rescue=([index.ids[c[t]] for t in wide],s[wide].copy(),[(query,index.document_for(r,c[t])) for t in wide])
                    local_jobs.append((j,[index.ids[c[t]] for t in order],s[order].copy(),nquery,[(query,index.document_for(r,c[t])) for t in order],rescue))
                put((output,local_jobs))
            put(None)
        except BaseException as error:put(error)
    worker=Thread(target=prepare,name='vessel-cpu-pipeline',daemon=True)
    previous_interval=sys.getswitchinterval();sys.setswitchinterval(.001)
    try:
        worker.start()
        while True:
            payload=queue.get()
            if payload is None:break
            if isinstance(payload,BaseException):raise payload
            output,local_jobs=payload;offset=len(result);result.extend(output)
            for j,ids,s,nquery,local_pairs,rescue in local_jobs:
                jobs.append((offset+j,ids,s,len(pairs),nquery,rescue));pairs.extend(local_pairs)
            if len(jobs)>=128:flush()
    finally:
        stop.set()
        try:
            if worker.ident is not None:worker.join()
        finally:sys.setswitchinterval(previous_interval)
    flush()
    if precise_pairs:
        values=neural.full_score(precise_pairs)
        for state,contenders,s,at in precise_jobs:
            state[2][contenders]=s+alpha*values[at:at+len(contenders)]
    if cascade_cap:
        proposals=[];proposal_jobs=[]
        for position,ids,combined,nquery,docs,raw_base,original_query,rescue in states:
            order=np.argsort(-combined,kind='stable');winner=int(order[0]);result[position]=ids[winner]
            if nquery is None or rescue is None or len(order)<2 or combined[order[0]]-combined[order[1]]>cascade_gate:continue
            rid,rs,rpairs=rescue
            proposal_jobs.append((position,ids[winner],float(combined[winner]),float(raw_base[winner]),(original_query,docs[winner]),rid,rs,rpairs,len(proposals)))
            proposals.append((nquery,name_document(docs[winner])));proposals.extend((nquery,name_document(p[1])) for p in rpairs)
        if proposals:
            values=neural.name_score(proposals);joint_pairs=[];joint_jobs=[]
            for position,old_id,old_score,old_base,old_pair,ids,s,ps,at in proposal_jobs:
                scores=values[at+1:at+1+len(ids)];order=np.argsort(-scores,kind='stable')[:2]
                order=[int(t) for t in order if scores[t]>=values[at]+cascade_advantage]
                if not order:continue
                joint_jobs.append((position,[old_id]+[ids[t] for t in order],np.r_[old_base,s[order]],old_score,old_pair,len(joint_pairs)))
                joint_pairs.extend(ps[t] for t in order)
            if joint_pairs:
                scored=neural.score(joint_pairs);close=[];close_jobs=[]
                for position,ids,base,old_score,old_pair,at in joint_jobs:
                    combined=np.r_[old_score,base[1:]+alpha*scored[at:at+len(ids)-1]]
                    order=np.argsort(-combined,kind='stable');result[position]=ids[int(order[0])]
                    if len(order)>1 and combined[order[0]]-combined[order[1]]<=.5 and hasattr(neural,'full_score'):
                        close_jobs.append((position,ids,base,len(close)));close.append(old_pair);close.extend(joint_pairs[at:at+len(ids)-1])
                if close:
                    full=neural.full_score(close)
                    for position,ids,base,at in close_jobs:result[position]=ids[int(np.argmax(base+alpha*full[at:at+len(ids)]))]
        return result
    name_pairs=[];name_jobs=[]
    for position,ids,combined,nquery,docs,raw_base,original_query,rescue in states:
        result[position]=ids[int(np.argmax(combined))]
        if nquery is None:continue
        order=np.argsort(-combined,kind='stable')[:name_cap]
        if len(order)<2 or combined[order[0]]-combined[order[1]]>name_gate:continue
        name_jobs.append((position,[ids[t] for t in order],combined[order].copy(),len(name_pairs),raw_base[order].copy(),[(original_query,docs[t]) for t in order]))
        name_pairs.extend((nquery,docs[t]) for t in order)
    if name_pairs:
        values=getattr(neural,'name_score',neural.score)(name_pairs);close_pairs=[];close_original=[];close_jobs=[]
        for position,ids,s,at,raw_base,original_pairs in name_jobs:
            combined=s+gamma*values[at:at+len(ids)];contenders=np.flatnonzero(combined>=combined.max()-.5)
            if len(contenders)>1 and hasattr(neural,'full_score'):
                close_jobs.append((position,[ids[t] for t in contenders],raw_base[contenders].copy(),len(close_pairs)))
                close_pairs.extend(name_pairs[at+t] for t in contenders)
                close_original.extend(original_pairs[t] for t in contenders)
            else:result[position]=ids[int(np.argmax(combined))]
        if close_pairs:
            original_full=neural.full_score(close_original)
            full=getattr(neural,'name_full_score',neural.full_score)(close_pairs)
            for position,ids,s,at in close_jobs:result[position]=ids[int(np.argmax(s+alpha*original_full[at:at+len(ids)]+gamma*full[at:at+len(ids)]))]
    if post_expert is not None:result=post_expert.apply(test,result,states,neural)
    return result


def run(data_dir=None,model_dir=None,output_path=None,ipa_model_dir=None,use_packaged_ipa=True):
    start=time.perf_counter();data_dir=Path(data_dir) if data_dir else BASE/'data';model_dir=Path(model_dir) if model_dir else BASE/'model'
    output_path=Path(output_path) if output_path else BASE/'output/submission.csv'
    from weight_storage import ensure_local_weights
    ensure_local_weights(model_dir)
    test=pd.read_csv(data_dir/'test.csv',encoding='utf-8-sig',dtype={c:str for c in ['sample_id','vessel_name','call_sign','flag_name','vessel_type']})
    required=['sample_id','vessel_name','call_sign','gross_tonnage','international_gross_tonnage','flag_name','vessel_type','arrival_year']
    if not set(required).issubset(test.columns) or test.sample_id.isna().any() or not test.sample_id.is_unique:raise ValueError('Invalid input schema')
    ca=pd.read_csv(data_dir/'vessel_catalog.csv');fit=pd.read_csv(model_dir/'train_reference.csv')
    config=json.loads((model_dir/'config.json').read_text())
    if ipa_model_dir is None and use_packaged_ipa and config.get('ipa_catboost_bundle'):
        ipa_model_dir=model_dir/config['ipa_catboost_bundle']
    rough=HybridRough(model_dir/'rough.cbm',model_dir/'rough_augmented.cbm',blend=config.get('rough_blend',1.))
    if rough.feature_names_!=V7_FEATURE_NAMES:raise ValueError('Feature schema mismatch')
    aux=DenseNames(model_dir/'catalog_embeddings.npz',ca,LazyEncoder(model_dir/'embed06',model_dir/'embed_lora')) if config.get('aux_dense',False) else None
    index=V4Index(ca,Knowledge(fit,ca),model_dir/'translit.joblib',model_dir/'char_index.joblib',aux_dense=aux,alias_path=model_dir/'alias_index.joblib',translit_aug_path=model_dir/'translit_aug.joblib',train_reference=fit)
    index.alias_documents=config.get('alias_documents',False)
    calibration=build_calibration(model_dir,index,config,ipa_model_dir=ipa_model_dir)
    expert=model_dir/'name_lora' if config.get('name_expert_adapter',False) else None
    complement=DenseNames(model_dir/'catalog_embeddings_complement.npz',ca,LazyEncoder(model_dir/'embed06',model_dir/'embed_complement_lora'))
    post=PostExpert(ca,index=index,rough=rough,calibration=calibration,complement=complement,dense_weight=config['beta'],retrieval_topk=config['complementary_retrieval_topk'],proposals=config['complementary_proposals'],coarse_floor=config['rescue_coarse_floor'],rescue_advantage=config['rescue_advantage'],rescue_gate=config['rescue_gate'],weight=config['post_expert_weight'],gate=config['post_expert_gate'],cap=config['post_expert_cap'],tolerance=config['numeric_veto_tolerance'])
    pred=predict_frame(test,index,rough,LazyJoint(model_dir/'qwen4',model_dir/'joint_lora',expert,expert_adapter=model_dir/'phonetic_expert_lora'),alpha=config['alpha'],english_gate=config['english_gate'],beta=config.get('beta',0.),routing=config.get('routing','old'),gamma=config.get('gamma',0.),name_cap=config.get('name_cap',8),name_gate=config.get('name_gate',6.),cascade_cap=config.get('cascade_cap',0),cascade_gate=config.get('cascade_gate',4.),cascade_advantage=config.get('cascade_advantage',1.),alias_beta=config.get('alias_beta',0.),phonetic_beta=config.get('phonetic_beta',0.),calibration=calibration,post_expert=post)
    if len(pred)!=len(test) or not set(pred).issubset(index.id_to_index):raise ValueError('Invalid predictions')
    output_path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'sample_id':test.sample_id,'target_vessel_id':pred}).to_csv(output_path,index=False)
    print('Saved submission.csv in %.2f seconds'%(time.perf_counter()-start),flush=True)
    return output_path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir');p.add_argument('--model-dir');p.add_argument('--output-path');p.add_argument('--ipa-model-dir',help='Trained IPA CatBoost bundle; packaged config can supply the default');p.add_argument('--original-v40',action='store_true',help='Disable the packaged IPA default for comparison');a=p.parse_args();run(a.data_dir,a.model_dir,a.output_path,a.ipa_model_dir,not a.original_v40)
