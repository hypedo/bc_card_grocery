"""Row-independent ship entity resolution. Learned data is official Train only."""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import math
import re
import unicodedata

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein
from unidecode import unidecode


def text(value):
    return value if isinstance(value, str) else ""


def hangul(value):
    return bool(re.search('[가-힣]', text(value)))


@lru_cache(maxsize=200000)
def norm(value):
    return ''.join(c for c in unicodedata.normalize('NFKC', text(value)).upper() if c.isalnum())


@lru_cache(maxsize=200000)
def clean(value):
    value = unicodedata.normalize('NFKC', text(value)).upper().strip()
    value = re.sub(r'^(?:M[. /]*[VT]|S[. /]*S)[. ]+', '', value)
    value = re.sub(r'호$', '', value)
    value = re.sub(r'제\s*(\d+)', r' NO \1 ', value)
    value = re.sub(r'[^A-Z0-9가-힣]+', ' ', value)
    value = re.sub(r'\bNO\b', ' ', value)
    return ' '.join(value.split())


@lru_cache(maxsize=200000)
def roman(value):
    return norm(unidecode(clean(value)))


@lru_cache(maxsize=200000)
def phon(value):
    value = roman(value).lower()
    # Generic spelling/romanization equivalences, no vessel-specific dictionaries.
    for a,b in [('ph','f'),('ck','k'),('qu','k'),('sh','s'),('ch','c'),('th','t'),
                ('ng','N'),('lr','l'),('rl','l'),('v','b'),('f','p'),('r','l'),
                ('c','k'),('q','k'),('z','j'),('x','ks')]:
        value = value.replace(a,b)
    value = re.sub('[aeiouywh]', '', value)
    return re.sub(r'(.)\1+', r'\1', value).upper()


@lru_cache(maxsize=200000)
def softphon(value):
    value=clean(value)
    if not hangul(value):
        value=value.lower()
        value=re.sub(r'gg?(?=[eiy])','j',value)
        value=re.sub(r'c(?=[eiy])','s',value)
        value=value.replace('th','s')
    return phon(value)


@lru_cache(maxsize=200000)
def numbers(value):
    return tuple(sorted(re.findall(r'\d+', clean(value))))


def pos(value):
    try:
        v = float(value)
    except (ValueError, TypeError):
        return np.nan
    return v if np.isfinite(v) and v > 0 else np.nan


def metrics(df, predictions):
    k = df.vessel_name.map(hangul).to_numpy()
    correct = df.target_vessel_id.to_numpy() == np.asarray(predictions)
    ak = float(correct[k].mean()) if k.any() else None
    ae = float(correct[~k].mean()) if (~k).any() else None
    return dict(n=len(df), hangul_n=int(k.sum()), hangul_accuracy=ak,
                non_hangul_accuracy=ae, accuracy=float(correct.mean()),
                weighted_accuracy=.7*ak+.3*ae if ak is not None and ae is not None else None,
                errors=int((~correct).sum()))


class Knowledge:
    """Immutable maps learned exclusively from the supplied training partition."""
    def __init__(self, train, catalog):
        self.calls = {}
        self.aliases = defaultdict(set)
        self.flag_counts = defaultdict(Counter)
        self.type_counts = defaultdict(Counter)
        self.token_counts = defaultdict(Counter)
        self.en_to_ko = defaultdict(Counter)
        if train is None or len(train)==0:
            return
        cat = catalog.set_index('vessel_id')
        for cs, g in train.groupby('call_sign'):
            ids = g.target_vessel_id.unique()
            if len(ids)==1:
                self.calls[str(cs)] = str(ids[0])
        pairs = train.drop_duplicates(['vessel_name','target_vessel_id'])
        for r in pairs.to_dict('records'):
            c = cat.loc[r['target_vessel_id']]
            self.aliases[norm(r['vessel_name'])].add(r['target_vessel_id'])
            if hangul(r['vessel_name']):
                ko = clean(r['vessel_name']).split()
                en = clean(c.vessel_name).split()
                if len(ko)==len(en):
                    for a,b in zip(ko,en):
                        if hangul(a) and b.isalpha():
                            self.token_counts[a][b]+=1
                            self.en_to_ko[b][a]+=1
        merged = train.merge(catalog,left_on='target_vessel_id',right_on='vessel_id',suffixes=('_q','_c'))
        # Count distinct ships rather than repeated visits.
        for r in merged.drop_duplicates(['target_vessel_id','flag_name_q','vessel_type_q']).to_dict('records'):
            if text(r['flag_name_q']):
                self.flag_counts[r['flag_name_q']][r['flag_name_c']]+=1
            self.type_counts[text(r['vessel_type_q'])][r['vessel_type_c']]+=1
        self.token_map = {k:v.most_common(1)[0][0] for k,v in self.token_counts.items()}
        self.reverse_map = {k:v.most_common(1)[0][0] for k,v in self.en_to_ko.items()}

    def translate(self, name):
        value = clean(name)
        # Longest substring replacement also covers names with omitted spaces.
        for ko in sorted(getattr(self,'token_map',{}),key=lambda x:(-len(x),x)):
            if len(ko)>=2:
                value=value.replace(ko,' '+self.token_map[ko]+' ')
        return norm(unidecode(value))

    def probability(self, field, q, c):
        counts = getattr(self,field+'_counts').get(text(q),{})
        return counts.get(text(c),0)/max(sum(counts.values()),1)


FEATURE_NAMES = [
    'hangul','name_exact','name_ratio','name_wratio','name_jaro','name_levenshtein',
    'roman_ratio','roman_wratio','phon_ratio','phon_wratio','translate_ratio',
    'translate_wratio','translated_phon_ratio','name_length_ratio','phon_length_ratio',
    'name_prefix','name_suffix','number_equal','number_conflict','number_query',
    'gt_gap','igt_gap','best_gt_gap','gt_signed','igt_signed','gt_exact','igt_exact',
    'gt_within_5pct','igt_within_5pct','query_gt_missing','query_igt_missing','catalog_gt_missing',
    'gt_pair_gap','flag_probability','type_probability','flag_missing','query_log_gt',
    'catalog_log_gt','dwt_gap','size_gap','token_overlap','translated_exact',
]


class CatalogIndex:
    def __init__(self, catalog, knowledge=None):
        self.catalog=catalog.reset_index(drop=True)
        self.ids=self.catalog.vessel_id.astype(str).to_numpy()
        self.id_to_index={s:i for i,s in enumerate(self.ids)}
        self.raw=self.catalog.vessel_name.astype(str).tolist()
        self.names=[norm(clean(x)) for x in self.raw]
        self.romans=[roman(x) for x in self.raw]
        self.phons=[phon(x) for x in self.raw]
        self.nums=[numbers(x) for x in self.raw]
        self.flags=self.catalog.flag_name.fillna('').to_numpy()
        self.types=self.catalog.vessel_type.fillna('').to_numpy()
        self.gt=pd.to_numeric(self.catalog.gross_tonnage,errors='coerce').to_numpy(float)
        self.dwt=pd.to_numeric(self.catalog.deadweight_tonnage,errors='coerce').to_numpy(float)
        self.size=pd.to_numeric(self.catalog['size'],errors='coerce').to_numpy(float)
        self.knowledge=knowledge or Knowledge(None,catalog)
        self.valid=np.flatnonzero(np.isfinite(self.gt)&(self.gt>0))
        self.gt_sort=self.valid[np.argsort(self.gt[self.valid],kind='stable')]
        self.gt_values=self.gt[self.gt_sort]
        self.exact=defaultdict(list)
        for i,s in enumerate(self.names):self.exact[s].append(i)
        self._lex_cache={}

    def near_gt(self, value, count=30):
        if not np.isfinite(value):return []
        p=np.searchsorted(self.gt_values,value)
        a=self.gt_sort[max(0,p-count*3):p+count*3]
        return a[np.argsort(np.abs(self.gt[a]-value),kind='stable')[:count]].tolist()

    def lexical(self, name):
        key=(name,)
        if key not in self._lex_cache:
            n=norm(clean(name)); r=roman(name); p=phon(name)
            t=self.knowledge.translate(name)
            out={i for _,_,i in process.extract(n,self.names,scorer=fuzz.ratio,limit=20)}
            if hangul(name):
                out.update(i for _,_,i in process.extract(r,self.romans,scorer=fuzz.ratio,limit=20))
                out.update(i for _,_,i in process.extract(p,self.phons,scorer=fuzz.ratio,limit=30))
                out.update(i for _,_,i in process.extract(t,self.names,scorer=fuzz.ratio,limit=30))
            self._lex_cache[key]=sorted(out)
        return self._lex_cache[key]

    def candidates(self,row,limit=70):
        name=text(row.get('vessel_name')); q=norm(clean(name)); tr=self.knowledge.translate(name)
        p=phon(name); g=pos(row.get('gross_tonnage')); ig=pos(row.get('international_gross_tonnage'))
        out=set(self.lexical(name))
        out.update(self.near_gt(g,30));out.update(self.near_gt(ig,30))
        for v in [g,ig]:
            if not np.isfinite(v):continue
            lo,hi=np.searchsorted(self.gt_values,[v/1.06,v/.94])
            near=self.gt_sort[lo:hi]
            if len(near):
                # A row-specific numerical block rescues difficult transliterations.
                score=np.array([fuzz.ratio(p,self.phons[i])*.45+fuzz.ratio(tr,self.names[i])*.55
                                +10*self.knowledge.probability('type',row.get('vessel_type'),self.types[i])
                                +5*self.knowledge.probability('flag',row.get('flag_name'),self.flags[i]) for i in near])
                out.update(near[np.argsort(-score,kind='stable')[:20]].tolist())
        out.update(self.exact.get(q,[]))
        for vid in self.knowledge.aliases.get(norm(name),[]):
            if vid in self.id_to_index:out.add(self.id_to_index[vid])
        call=self.knowledge.calls.get(text(row.get('call_sign')))
        if call in self.id_to_index:out.add(self.id_to_index[call])
        scored=[]
        for i in sorted(out):
            gap=min(abs(g-self.gt[i])/max(g,self.gt[i]) if np.isfinite(g*self.gt[i]) else 1,
                    abs(ig-self.gt[i])/max(ig,self.gt[i]) if np.isfinite(ig*self.gt[i]) else 1)
            sim=max(fuzz.ratio(q,self.names[i]),fuzz.ratio(tr,self.names[i]))/100
            if hangul(name):sim=max(sim,.88*fuzz.ratio(p,self.phons[i])/100)
            score=sim+.12*(gap==0)+.1*(gap<=.055)+.04*self.knowledge.probability('type',row.get('vessel_type'),self.types[i])
            scored.append((score,-i,i))
        selected=[x[2] for x in sorted(scored,reverse=True)[:limit]]
        return selected or [0]

    def features(self,row,candidates):
        name=text(row.get('vessel_name'));q=norm(clean(name));r=roman(name);p=phon(name)
        t=self.knowledge.translate(name);pt=phon(t);num=numbers(name);ko=hangul(name)
        g=pos(row.get('gross_tonnage'));ig=pos(row.get('international_gross_tonnage'))
        features=[]
        for i in candidates:
            c=self.names[i];cg=self.gt[i]; cp=self.phons[i]
            gg=abs(g-cg)/max(g,cg) if np.isfinite(g*cg) else 1.
            gi=abs(ig-cg)/max(ig,cg) if np.isfinite(ig*cg) else 1.
            signed=(g-cg)/max(g,cg) if np.isfinite(g*cg) else 1.
            si=(ig-cg)/max(ig,cg) if np.isfinite(ig*cg) else 1.
            prefix=0
            for a,b in zip(t,c):
                if a!=b:break
                prefix+=1
            suffix=0
            for a,b in zip(t[::-1],c[::-1]):
                if a!=b:break
                suffix+=1
            ct=set(clean(self.raw[i]).split());qt=set(clean(name).split())
            features.append([
                ko,q==c,fuzz.ratio(q,c)/100,fuzz.WRatio(q,c)/100,JaroWinkler.normalized_similarity(q,c),
                Levenshtein.normalized_similarity(q,c),fuzz.ratio(r,self.romans[i])/100,
                fuzz.WRatio(r,self.romans[i])/100,fuzz.ratio(p,cp)/100,fuzz.WRatio(p,cp)/100,
                fuzz.ratio(t,c)/100,fuzz.WRatio(t,c)/100,fuzz.ratio(pt,cp)/100,
                min(len(t),len(c))/max(len(t),len(c),1),min(len(p),len(cp))/max(len(p),len(cp),1),
                prefix/max(len(t),len(c),1),suffix/max(len(t),len(c),1),
                bool(num) and num==self.nums[i],bool(num) and bool(self.nums[i]) and num!=self.nums[i],bool(num),
                gg,gi,min(gg,gi),signed,si,gg==0,gi==0,gg<=.051,gi<=.051,
                not np.isfinite(g),not np.isfinite(ig),not np.isfinite(cg),
                abs(g-ig)/max(g,ig) if np.isfinite(g*ig) else 1.,
                self.knowledge.probability('flag',row.get('flag_name'),self.flags[i]),
                self.knowledge.probability('type',row.get('vessel_type'),self.types[i]),
                not bool(text(row.get('flag_name'))),math.log1p(g) if np.isfinite(g) else 0.,
                math.log1p(cg) if np.isfinite(cg) else 0.,
                abs(g-self.dwt[i])/max(g,self.dwt[i]) if np.isfinite(g*self.dwt[i]) else 1.,
                abs(g-self.size[i])/max(g,self.size[i]) if np.isfinite(g*self.size[i]) else 1.,
                len(ct&qt)/max(len(ct|qt),1),t==c,
            ])
        return np.asarray(features,dtype=np.float32)

    def heuristic(self,row,candidates,features):
        f=features
        k=hangul(row.get('vessel_name'))
        sim=np.maximum(f[:,2],f[:,10])
        if k:sim=np.maximum(sim,.88*f[:,8])
        return sim+.08*f[:,25]+.12*f[:,26]+.10*f[:,28]+.05*f[:,27]+.06*f[:,33]+.06*f[:,34]-.3*f[:,18]

    def direct(self,row):
        v=self.knowledge.calls.get(text(row.get('call_sign')))
        return v if v in self.id_to_index else None


ENHANCED_FEATURE_NAMES=FEATURE_NAMES+['soft_phon_ratio','soft_phon_wratio','token_sort_roman_ratio']


class EnhancedIndex(CatalogIndex):
    """Alternative soft consonants improve English to Hangul spelling equivalence."""
    def __init__(self,catalog,knowledge=None):
        super().__init__(catalog,knowledge)
        self.softphons=[softphon(x) for x in self.raw]

    def lexical(self,name):
        candidates=super().lexical(name)
        if not hangul(name):return candidates
        # Pure memoization of an immutable row-level function.
        key=('soft',name)
        if key not in self._lex_cache:
            extra=[i for _,_,i in process.extract(softphon(name),self.softphons,scorer=fuzz.ratio,limit=25)]
            self._lex_cache[key]=sorted(set(candidates)|set(extra))
        return self._lex_cache[key]

    def features(self,row,candidates):
        base=super().features(row,candidates)
        name=text(row.get('vessel_name'));s=softphon(name);r=unidecode(clean(name))
        extra=np.array([[fuzz.ratio(s,self.softphons[i])/100,
                         fuzz.WRatio(s,self.softphons[i])/100,
                         fuzz.token_sort_ratio(r,unidecode(clean(self.raw[i])))/100]
                        for i in candidates],dtype=np.float32)
        return np.column_stack([base,extra])


def make_pairs(queries,index,force_positive=False,max_negatives=None):
    xs=[];ys=[];groups=[];cands=[];hits=[]
    for row in queries.to_dict('records'):
        candidates=index.candidates(row)
        true=index.id_to_index.get(row.get('target_vessel_id'),-1)
        hits.append(true in candidates)
        if max_negatives is not None:
            candidates=[i for i in candidates if i==true]+[i for i in candidates if i!=true][:max_negatives]
        if force_positive and true not in candidates:candidates.append(true)
        xs.append(index.features(row,candidates))
        ys.extend(int(i==true) for i in candidates);groups.append(len(candidates));cands.extend(candidates)
    return dict(X=np.concatenate(xs),y=np.asarray(ys,dtype=np.int8),groups=np.asarray(groups),
                candidates=np.asarray(cands,dtype=np.int32),hits=np.asarray(hits))


def choose_predictions(scores,pairs,index,queries,direct=True):
    starts=np.r_[0,np.cumsum(pairs['groups'])]
    pred=[]
    for j,row in enumerate(queries.to_dict('records')):
        a,b=starts[j:j+2]
        chosen=int(pairs['candidates'][a+np.argmax(scores[a:b])])
        pred.append(index.direct(row) if direct and index.direct(row) else index.ids[chosen])
    return pred
