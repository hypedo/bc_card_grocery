"""Fixed-instruction and exact-query KV reuse for the offline 4B vessel scorer.

Only fixed model/instruction constants and immutable same-query states are
shared. No evaluation-population features or statistics are computed. Oversized
prompts retain the original full scorer and original truncation policy.
"""
import numpy as np
from structured_reranker import StructuredReranker
from prefix_reranker import PrefixReranker
from shipmatch_v4 import JOINT_INSTRUCTION

class PrefixV6(PrefixReranker):
    constant_length=104

    def apply_adapter(self,path):
        self._instruction_states=None
        return super().apply_adapter(path)

    prefix_length=192
    tail_length=64
    prefill_batch_size=32

    def _tokens(self, pairs):
        texts=[f'<Instruct>: {JOINT_INSTRUCTION}\n<Query>: {q}\n<Document>: {c}' for q,c in pairs]
        encoded=self.tokenizer(texts,padding=False,truncation=True,add_special_tokens=False,
            max_length=self.max_length-len(self.prefix)-len(self.suffix))['input_ids']
        return [self.prefix+x+self.suffix for x in encoded]

    def _head(self, query):
        text=f'<Instruct>: {JOINT_INSTRUCTION}\n<Query>: {query}\n<Document>:'
        return self.prefix+self.tokenizer.encode(text,add_special_tokens=False)

    def _pad(self, ids, length):
        batch=self.tokenizer.pad({'input_ids':ids},padding='max_length',max_length=length,return_tensors='pt')
        return {k:v.to(self.device) for k,v in batch.items()}

    def full_score(self,pairs):
        # A separate pure-function cache keeps close decisions on the original
        # full-prompt, fixed-32 geometry rather than mixing numerical variants.
        if not hasattr(self,'full_cache'):self.full_cache={}
        fast_cache,batch=self.cache,self.batch_size
        try:
            self.cache=self.full_cache;self.batch_size=32
            return StructuredReranker.score(self,pairs)
        finally:self.cache=fast_cache;self.batch_size=batch

    def score(self,pairs):
        pairs=[(str(q),str(c)) for q,c in pairs]
        pending=list(dict.fromkeys(p for p in pairs if p not in self.cache))
        if not pending:return np.asarray([self.cache[p] for p in pairs],np.float32)
        from transformers.cache_utils import DynamicCache
        torch=self.torch
        tokens=self._tokens(pending)
        heads={q:self._head(q) for q,_ in pending}
        filler=self._head('?')
        grouped={};fallback=[]
        for pair,ids in zip(pending,tokens):
            head=heads[pair[0]];tail=ids[len(head):]
            if len(head)>self.prefix_length or len(tail)>self.tail_length or not tail or ids[:len(head)]!=head or head[:self.constant_length]!=filler[:self.constant_length]:
                fallback.append(pair)
            else:grouped.setdefault(pair[0],[]).append((pair,tail))
        queries=list(grouped)
        with torch.inference_mode():
            if getattr(self,'_instruction_states',None) is None and queries:
                constant=filler[:self.constant_length]
                inp=self._pad([constant],self.constant_length)
                inp['position_ids']=(inp['attention_mask'].long().cumsum(-1)-1).clamp_min(0)
                self._instruction_states=self.model.model(**inp,use_cache=True).past_key_values
            for start in range(0,len(queries),self.prefill_batch_size):
                qs=queries[start:start+self.prefill_batch_size]
                prefix=self._pad([heads[q][self.constant_length:] for q in qs]+[filler[self.constant_length:]]*(self.prefill_batch_size-len(qs)),self.prefix_length-self.constant_length)
                local_mask=prefix['attention_mask']
                constant_mask=torch.ones((self.prefill_batch_size,self.constant_length),device=self.device,dtype=local_mask.dtype)
                mask=torch.cat([constant_mask,local_mask],dim=1)
                prefix['attention_mask']=mask
                prefix['position_ids']=(self.constant_length+local_mask.long().cumsum(-1)-1).clamp_min(0)
                source0=torch.zeros(self.prefill_batch_size,device=self.device,dtype=torch.long)
                constant_cache=DynamicCache([(layer.keys.index_select(0,source0),layer.values.index_select(0,source0)) for layer in self._instruction_states.layers])
                states=self.model.model(**prefix,past_key_values=constant_cache,use_cache=True).past_key_values
                jobs=[(j,pair,tail) for j,q in enumerate(qs) for pair,tail in grouped[q]]
                for offset in range(0,len(jobs),self.batch_size):
                    now=jobs[offset:offset+self.batch_size]
                    pad=self.batch_size-len(now)
                    source=torch.tensor([j for j,_,_ in now]+[0]*pad,device=self.device)
                    # Each forward receives its own cache; the saved prefix is never mutated.
                    cache=DynamicCache([(layer.keys.index_select(0,source),layer.values.index_select(0,source)) for layer in states.layers])
                    inp=self._pad([tail for _,_,tail in now]+[self.suffix]*pad,self.tail_length)
                    tail_mask=inp['attention_mask'];prefix_mask=mask.index_select(0,source)
                    inp['attention_mask']=torch.cat([prefix_mask,tail_mask],dim=1)
                    inp['position_ids']=(prefix_mask.sum(-1,keepdim=True)+tail_mask.long().cumsum(-1)-1).clamp_min(0)
                    hidden=self.model.model(**inp,past_key_values=cache,use_cache=True).last_hidden_state[:,-1,:]
                    logits=torch.nn.functional.linear(hidden,self.model.lm_head.weight[[self.no,self.yes]]).float()
                    values=(logits[:,1]-logits[:,0]).cpu().numpy()
                    for (_,pair,_),value in zip(now,values):self.cache[pair]=float(value)
                    del cache,hidden,logits
                del states
        if fallback:
            values=self.full_score(fallback)
            self.cache.update(zip(fallback,values.tolist()))
        return np.asarray([self.cache[p] for p in pairs],np.float32)
