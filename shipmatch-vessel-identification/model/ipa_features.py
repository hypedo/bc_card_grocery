"""Catalog-only IPA features shared by CatBoost training and inference.

The index consumes generic IPA-derived Hangul aliases, never query labels.
Catalog IDs/names and file hashes bind a serialized matrix to its row mapping.
"""
from collections import OrderedDict
from pathlib import Path
import hashlib
import json
import re
import unicodedata

import joblib
import numpy as np
from rapidfuzz import fuzz
from scipy.sparse import load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from shipmatch_v4 import canonical_row

IPA_FEATURE_NAMES = [
    'ipa_cosine', 'ipa_margin_to_catalog_best', 'ipa_reciprocal_rank',
    'ipa_margin_to_best_other', 'ipa_jamo_edit_ratio', 'ipa_alias_length_ratio',
    'ipa_alias_available', 'ipa_query_hangul_length', 'ipa_query_token_count',
    'ipa_query_has_digits', 'ipa_digits_equal', 'ipa_digits_conflict',
    'ipa_minus_compositional', 'ipa_minus_existing_alias',
]
INDEX_FILES = ['vectorizer.joblib', 'matrix.npz', 'rows.npy', 'aliases.jsonl']
SCHEMA = 'ipa-catboost-v1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def catalog_signature(catalog):
    pairs = sorted(zip(catalog.vessel_id.astype(str), catalog.vessel_name.astype(str)))
    return hashlib.sha256(json.dumps(pairs, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def jamo(text):
    text = re.sub(r'[^A-Z0-9가-힣]', '', unicodedata.normalize('NFKC', str(text)).upper())
    for a, b in [('씨', '시'), ('쓰', '스'), ('쎄', '세'), ('앤', '엔')]:
        text = text.replace(a, b)
    return unicodedata.normalize('NFD', text)


def digits(text):
    return tuple(str(int(n)) for n in re.findall(r'\d+', str(text)))


def build_index(catalog, aliases_path, output):
    """Build from catalog-name aliases. No Train or validation labels are used."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists():
        raise FileExistsError(f'IPA index already exists: {output}')
    if not catalog.vessel_id.is_unique:
        raise ValueError('Duplicate catalog vessel IDs')
    aliases = {}
    catalog_names = set(catalog.vessel_name.astype(str))
    with Path(aliases_path).open(encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            name = str(item['name'])
            if name not in catalog_names:
                continue
            aliases.setdefault(name, set()).update(
                str(a) for a in item['aliases'] if re.search('[가-힣]', str(a)))
    ordered = catalog.sort_values('vessel_id', kind='stable')
    texts, rows = [], []
    with (output / 'aliases.jsonl').open('w', encoding='utf-8') as f:
        for i, item in enumerate(ordered.itertuples()):
            options = sorted(aliases.get(str(item.vessel_name), []))
            f.write(json.dumps({'id': str(item.vessel_id), 'aliases': options}, ensure_ascii=False) + '\n')
            for alias in options:
                texts.append(jamo(alias)); rows.append(i)
    if not texts:
        raise ValueError('No catalog IPA Hangul aliases found')
    tf = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), lowercase=False,
                         dtype=np.float32, sublinear_tf=True, norm='l2')
    matrix = tf.fit_transform(texts).tocsr()
    joblib.dump(tf, output / 'vectorizer.joblib')
    save_npz(output / 'matrix.npz', matrix)
    np.save(output / 'rows.npy', np.asarray(rows, np.int32))
    manifest = {'schema': SCHEMA, 'feature_names': IPA_FEATURE_NAMES,
                'catalog_signature': catalog_signature(catalog),
                'ids': ordered.vessel_id.astype(str).tolist(),
                'source_aliases_sha256': digest(aliases_path),
                'source': 'Generic catalog-name IPA to Hangul; no Train aliases',
                'files': {n: digest(output / n) for n in INDEX_FILES}}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


class IPAFeatures:
    def __init__(self, path, catalog, cache_size=64):
        path = Path(path)
        m = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
        if m['schema'] != SCHEMA or m['feature_names'] != IPA_FEATURE_NAMES:
            raise ValueError('IPA feature schema mismatch')
        if m['catalog_signature'] != catalog_signature(catalog):
            raise ValueError('IPA index belongs to a different catalog')
        for name in INDEX_FILES:
            if digest(path / name) != m['files'][name]:
                raise ValueError(f'IPA index checksum mismatch: {name}')
        self.tf = joblib.load(path / 'vectorizer.joblib')
        self.matrix = load_npz(path / 'matrix.npz')
        self.rows = np.load(path / 'rows.npy', allow_pickle=False)
        self.ids = np.asarray(m['ids'])
        expected = sorted(catalog.vessel_id.astype(str))
        if self.ids.tolist() != expected or not catalog.vessel_id.is_unique:
            raise ValueError('IPA index ID mapping mismatch')
        if len(self.rows) != self.matrix.shape[0] or len(self.rows) == 0 or self.rows.min() < 0 or self.rows.max() >= len(self.ids):
            raise ValueError('Invalid IPA matrix row mapping')
        lookup = {v: i for i, v in enumerate(self.ids)}
        self.catalog_to_sorted = np.asarray([lookup[str(v)] for v in catalog.vessel_id], np.int32)
        self.names = catalog.vessel_name.astype(str).tolist()
        self.aliases = {}
        with (path / 'aliases.jsonl').open(encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.aliases[item['id']] = item['aliases']
        self.catalog_ids = catalog.vessel_id.astype(str).tolist()
        self.cache = OrderedDict()
        self.cache_size = cache_size

    def view(self, name):
        name = str(name)
        if name not in self.cache:
            raw = (self.matrix @ self.tf.transform([jamo(name)]).T).toarray().ravel()
            scores = np.zeros(len(self.ids), np.float32)
            np.maximum.at(scores, self.rows, raw)
            order = np.argsort(-scores, kind='stable')  # ties: ascending vessel ID
            reciprocal = np.zeros(len(scores), np.float32)
            positive = order[scores[order] > 0]
            reciprocal[positive] = 1. / np.arange(1, len(positive) + 1)
            self.cache[name] = (scores, order, reciprocal)
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(name)
        return self.cache[name]

    def features(self, row, candidates, base_features, phonetic):
        row = canonical_row(row)
        candidates = np.asarray(candidates, dtype=np.int64)
        if not len(candidates):
            return np.empty((0, len(IPA_FEATURE_NAMES)), np.float32)
        if not re.search('[가-힣]', row['vessel_name']):
            return np.zeros((len(candidates), len(IPA_FEATURE_NAMES)), np.float32)
        scores, order, reciprocal = self.view(row['vessel_name'])
        name = row['vessel_name']; q = jamo(name); qdigits = digits(name)
        best = float(scores[order[0]])
        second = float(scores[order[1]]) if len(order) > 1 else 0.
        length = len(re.findall('[가-힣]', name)); tokens = len(name.split())
        output = []
        for offset, candidate in enumerate(candidates):
            i = int(candidate); si = int(self.catalog_to_sorted[i])
            aliases = self.aliases.get(self.catalog_ids[i], [])
            options = [(fuzz.ratio(q, jamo(a)) / 100., jamo(a)) for a in aliases]
            ratio, alias = max(options, default=(0., ''))
            cdigits = digits(self.names[i]); score = float(scores[si])
            other = second if si == int(order[0]) else best
            output.append([
                score, score - best, float(reciprocal[si]), score - other,
                ratio, min(len(q), len(alias)) / max(len(q), len(alias), 1),
                float(bool(aliases)), length, tokens, float(bool(qdigits)),
                float(bool(qdigits) and qdigits == cdigits),
                float(bool(qdigits) and bool(cdigits) and qdigits != cdigits),
                score - float(phonetic[offset, 0]), score - float(base_features[offset, 58]),
            ])
        result = np.asarray(output, np.float32)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite IPA features')
        return result
