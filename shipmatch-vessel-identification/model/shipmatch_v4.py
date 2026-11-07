"""Normalized row-local retrieval and structured pair inputs, official data only."""
import re
import unicodedata
from functools import lru_cache
from collections import OrderedDict
import math
import numpy as np
from shipmatch import EnhancedIndex, clean, text, norm, pos, hangul
from shipmatch_v3 import RobustIndex, V3_FEATURE_NAMES
from learned_translit import TranslitIndex

V4_FEATURE_NAMES = V3_FEATURE_NAMES + ['learned_translit', 'learned_translit_gap', 'learned_translit_rank', 'abbreviation_exact', 'ordered_token_prefix', 'suffix_tokens_match']
JOINT_INSTRUCTION = (
    'Identify the same vessel. Compare Korean or English names by spelling and pronunciation, '
    'including abbreviations and distinguishing numbers. Combine name, gross tonnage, flag and type. '
    'Tonnage can be missing or inaccurate; missing attributes are not mismatches. '
    'A similar fleet prefix alone is insufficient. Answer yes only for the same vessel.'
)


def normalized_text(value):
    if not isinstance(value, str): return ''
    value = unicodedata.normalize('NFKC', value).replace('\u200b', '').strip()
    return '' if value.lower() in ['nan', 'none', 'null', 'n/a', '-', 'unknown'] else value


def numeric(value):
    if isinstance(value, str):
        s = normalized_text(value).replace(',', '').replace(' ', '')
        s = re.sub(r'(?i)(?:G/T|GT|TONS?|톤)$', '', s)
        value = s
    return pos(value)


def canonical_row(row):
    r = dict(row)
    for name in ['vessel_name', 'call_sign', 'flag_name', 'vessel_type']:
        r[name] = normalized_text(r.get(name))
    r['call_sign'] = r['call_sign'].upper()
    for name in ['gross_tonnage', 'international_gross_tonnage']:
        r[name] = numeric(r.get(name))
    return r


def number(value):
    x = numeric(value)
    return str(round(x, 2)).rstrip('0').rstrip('.') if np.isfinite(x) and x % 1 else str(int(x)) if np.isfinite(x) else '?'


def query_text(row):
    r = canonical_row(row)
    return ('Name: ' + r['vessel_name'] + '\nGT: ' + number(r.get('gross_tonnage')) +
            '; IGT: ' + number(r.get('international_gross_tonnage')) +
            '\nFlag: ' + (r['flag_name'] or '?') + '; Type: ' + (r['vessel_type'] or '?'))


def document_text(row):
    return ('Name: ' + normalized_text(row.get('vessel_name')) +
            '\nGT: ' + number(row.get('gross_tonnage')) +
            '; DWT: ' + number(row.get('deadweight_tonnage')) +
            '\nFlag: ' + normalized_text(row.get('flag_name')) +
            '; Type: ' + normalized_text(row.get('vessel_type')))


def joint_route(row, features, gap, english_gate=4., mode='expanded'):
    row=canonical_row(row);name=row['vessel_name']
    if hangul(name):return True
    old=gap<english_gate and features[1]<1.
    if mode=='old':return old
    words=clean(name).split();n=norm(name)
    short=(len(words)>1 and all(len(w)<=3 for w in words)) or (bool(n) and len(n)<=5 and not n.isdigit())
    fully_supported=features[1]==1 and features[22]<=.01 and features[33]>.5 and features[34]>.5
    numeric_present=not(features[29] and features[30])
    contradiction=features[1]==1 and numeric_present and features[22]>.15
    unfamiliar=bool(name) and not name.isascii()
    return old or (short and not fully_supported) or contradiction or unfamiliar


class V4Index(RobustIndex):
    def __init__(self, catalog, knowledge, translit_path, sparse_path=None, aux_dense=None):
        super().__init__(catalog, knowledge, dense=None, sparse_path=sparse_path, auxiliary=True)
        self.translit = TranslitIndex(translit_path, catalog)
        self._trans_views = {}
        self.aux_dense=aux_dense
        self._dense_views={}
        self._feature_cache=OrderedDict()
        self._flag_norm=np.asarray([norm(v) for v in self.flags])
        self._type_norm=np.asarray([norm(v) for v in self.types])
        self.documents = [document_text(r) for r in catalog.to_dict('records')]
        self.abbreviations={}
        self.word_names=[clean(n).split() for n in self.raw]
        for i,words in enumerate(self.word_names):
            if len(words)<2:continue
            for width in [1,2,3]:
                key=''.join(w[:width] for w in words)
                if len(key)>=2:self.abbreviations.setdefault(key,[]).append(i)

    def trans_view(self, name):
        name = normalized_text(name)
        if name not in self._trans_views:
            if len(self._trans_views)>=256:self._trans_views.pop(next(iter(self._trans_views)))
            scores = self.translit.scores(name)
            order = np.argpartition(-scores, 127)[:128]
            order = order[np.lexsort((self.ids[order], -scores[order]))]
            self._trans_views[name] = (order, scores)
        return self._trans_views[name]

    def views(self,name):
        if name not in self._views and len(self._views)>=256:self._views.pop(next(iter(self._views)))
        return super().views(name)

    def base_candidates(self, row, limit=200):
        row = canonical_row(row)
        old = super().candidates(row, limit=120)
        numeric = self.numeric_candidates(row)
        abbr=self.abbreviations.get(norm(row['vessel_name']),[])
        if len(abbr)>64:
            af=EnhancedIndex.features(self,row,abbr)
            abbr=[abbr[t] for t in np.argsort(af[:,22]-.05*af[:,33]-.05*af[:,34],kind='stable')[:64]]
        if not hangul(row['vessel_name']): return list(dict.fromkeys(abbr + old + numeric))
        learned, _ = self.trans_view(row['vessel_name'])
        # Reserve independent channels before filling the remainder.
        return list(dict.fromkeys(abbr + old[:48] + numeric[:32] + learned[:80].tolist() + old[48:] + numeric[32:] + learned[80:].tolist()))[:limit]

    def candidates(self,row,limit=340):
        row=canonical_row(row);base=self.base_candidates(row)
        if self.aux_dense is not None and hangul(row['vessel_name']):
            dense,_=self.dense_view(row['vessel_name'])
            return list(dict.fromkeys(base+dense))[:limit]
        return base

    def dense_view(self,name):
        name=normalized_text(name)
        if name not in self._dense_views:
            if len(self._dense_views)>=256:self._dense_views.pop(next(iter(self._dense_views)))
            raw=self.aux_dense.vectors@self.aux_dense.vector(name)
            scores=raw[self.aux_dense.catalog_to_row]
            order=np.argpartition(-scores,127)[:128]
            order=order[np.lexsort((self.ids[order],-scores[order]))].tolist()
            self._dense_views[name]=(order,scores)
        return self._dense_views[name]

    def auxiliary_scores(self,row,candidates):
        if self.aux_dense is None or not hangul(canonical_row(row)['vessel_name']):return np.zeros(len(candidates),np.float32)
        return self.dense_view(canonical_row(row)['vessel_name'])[1][candidates]

    @lru_cache(maxsize=96)
    def metadata_probability(self, field, value):
        counts = getattr(self.knowledge, field + '_counts').get(text(value), {})
        total = max(sum(counts.values()), 1)
        source = self.flags if field == 'flag' else self.types
        return np.asarray([counts.get(text(v), 0) / total for v in source], np.float32)

    def numeric_candidates(self, row):
        g, ig = row['gross_tonnage'], row['international_gross_tonnage']
        if not np.isfinite(g) and not np.isfinite(ig): return []
        def gap(value):
            return np.where(np.isfinite(value*self.gt), np.abs(value-self.gt)/np.maximum(value,self.gt), 1.)
        distance = np.minimum(gap(g), gap(ig))
        score = -6*distance + .35*self.metadata_probability('type',row['vessel_type']) + .2*self.metadata_probability('flag',row['flag_name'])
        out = np.argpartition(-score,47)[:48]
        out = out[np.lexsort((self.ids[out],-score[out]))].tolist()
        pure = np.argsort(distance,kind='stable')[:8].tolist()
        return list(dict.fromkeys(pure+out))

    def _features_uncached(self, row, candidates):
        row = canonical_row(row)
        base = super().features(row, candidates)
        if hangul(row['vessel_name']):
            order, scores = self.trans_view(row['vessel_name'])
            lookup = {int(v): 1 / (i + 1) for i, v in enumerate(order)}
            values = scores[candidates]
            extra = np.column_stack([values, values - float(scores[order[0]]), [lookup.get(int(i), 0) for i in candidates]])
        else: extra = np.zeros((len(candidates), 3), np.float32)
        q=clean(row['vessel_name']).split();qn=norm(row['vessel_name']);ab=[]
        for i in candidates:
            words=self.word_names[i]
            exact=bool(qn) and len(words)>1 and any(qn==''.join(w[:width] for w in words) for width in [1,2,3])
            aligned=sum(c.startswith(a) for a,c in zip(q,words))/max(len(words),len(q),1) if len(q)==len(words) else 0.
            suffix=bool(q) and len(q)<len(words) and q==words[-len(q):]
            ab.append([exact,aligned,suffix])
        return np.column_stack([base, extra,np.asarray(ab,np.float32)]).astype(np.float32)

    def features(self,row,candidates):
        row=canonical_row(row);name=row['vessel_name']
        if name not in self._feature_cache:
            self._feature_cache[name]={}
            if len(self._feature_cache)>512:self._feature_cache.popitem(last=False)
        memo=self._feature_cache[name];self._feature_cache.move_to_end(name)
        missing=[i for i in candidates if i not in memo]
        if missing:
            values=self._features_uncached(row,missing)
            memo.update((i,v.copy()) for i,v in zip(missing,values))
        f=np.stack([memo[i] for i in candidates]);c=np.asarray(candidates,dtype=int)
        g,ig=row['gross_tonnage'],row['international_gross_tonnage'];cg=self.gt[c]
        def gap(value,other):return np.where(np.isfinite(value*other),np.abs(value-other)/np.maximum(value,other),1.)
        gg,gi=gap(g,cg),gap(ig,cg)
        f[:,20]=gg;f[:,21]=gi;f[:,22]=np.minimum(gg,gi)
        f[:,23]=np.where(np.isfinite(g*cg),(g-cg)/np.maximum(g,cg),1.)
        f[:,24]=np.where(np.isfinite(ig*cg),(ig-cg)/np.maximum(ig,cg),1.)
        f[:,25]=gg==0;f[:,26]=gi==0;f[:,27]=gg<=.051;f[:,28]=gi<=.051
        f[:,29]=not np.isfinite(g);f[:,30]=not np.isfinite(ig)
        f[:,32]=abs(g-ig)/max(g,ig) if np.isfinite(g*ig) else 1.
        f[:,33]=self.metadata_probability('flag',row['flag_name'])[c]
        f[:,34]=self.metadata_probability('type',row['vessel_type'])[c]
        f[:,35]=not bool(row['flag_name']);f[:,36]=math.log1p(g) if np.isfinite(g) else 0.
        f[:,38]=gap(g,self.dwt[c]);f[:,39]=gap(g,self.size[c])
        f[:,47]=(norm(row['flag_name'])==self._flag_norm[c])&(self._flag_norm[c]!='')
        f[:,48]=(norm(row['vessel_type'])==self._type_norm[c])&(self._type_norm[c]!='')
        f[:,49]=gap(ig,self.dwt[c])
        return f

    def direct(self, row):
        row = canonical_row(row)
        v = self.knowledge.calls.get(row['call_sign'])
        if v not in self.id_to_index: return None
        i = self.id_to_index[v]
        f = EnhancedIndex.features(self, row, [i])[0]
        similarity = max(f[2], f[6], f[10], f[42])
        numeric_ok = f[22] <= .055
        missing = not np.isfinite(row['gross_tonnage']) and not np.isfinite(row['international_gross_tonnage'])
        return v if similarity >= .84 and (numeric_ok or (missing and similarity >= .97)) and not f[18] else None


def shortlist_v4(row, candidates, features, scores, k=6, index=None):
    order = np.argsort(-scores, kind='stable').tolist()
    row=canonical_row(row)
    numeric = np.argsort(features[:,22],kind='stable')[:2].tolist()
    if not hangul(row['vessel_name']): return list(dict.fromkeys(order[:6]+numeric))[:8]
    learned = np.argsort(-features[:, 52], kind='stable')[:2].tolist()
    phonetic = np.argsort(-np.maximum(features[:, 42], features[:, 10]), kind='stable')[:2].tolist()
    chosen=list(dict.fromkeys(order[:k] + learned + phonetic + numeric))[:12]
    if index is not None and index.aux_dense is not None:
        # A rescue channel must retain the pre-existing shortlist rather than evict it.
        base=set(index.base_candidates(row))
        positions=[j for j,c in enumerate(candidates) if c in base]
        local=shortlist_v4(row,[candidates[j] for j in positions],features[positions],scores[positions],k=k)
        chosen=[positions[j] for j in local]
        dense=np.argsort(-index.auxiliary_scores(row,candidates),kind='stable')[:4].tolist()
        chosen=list(dict.fromkeys(chosen+dense+order[:2]))[:18]
    return chosen
