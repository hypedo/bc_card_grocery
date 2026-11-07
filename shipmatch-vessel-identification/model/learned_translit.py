"""Official-Train-only compositional English-to-Korean-jamo retrieval."""
from pathlib import Path
import unicodedata
import re
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
from shipmatch import clean, hangul


def jamo(name):
    return unicodedata.normalize('NFD', clean(str(name)))


def fit_translit(train, catalog, path, alpha=3.0, components=96):
    lookup = catalog.set_index('vessel_id').vessel_name
    pairs = set()
    for r in train.drop_duplicates(['vessel_name', 'target_vessel_id']).itertuples():
        if not hangul(r.vessel_name):
            continue
        ko, en = clean(r.vessel_name), clean(lookup.loc[r.target_vessel_id])
        pairs.add((en, ko))
        a, b = en.split(), ko.split()
        if len(a) == len(b):
            pairs.update((e, k) for e, k in zip(a, b) if len(k) >= 2 and hangul(k) and len(e) >= 2)
    pairs = sorted(pairs)
    en, ko = zip(*pairs)
    ex = TfidfVectorizer(analyzer='char', ngram_range=(1, 4), lowercase=False,
                         sublinear_tf=True, dtype=np.float32, max_features=24000)
    ky = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), lowercase=False,
                         sublinear_tf=True, dtype=np.float32)
    x = ex.fit_transform(en)
    y = ky.fit_transform([jamo(k) for k in ko])
    svd = TruncatedSVD(n_components=min(components, y.shape[0] - 1, y.shape[1] - 1), random_state=20261023)
    latent = svd.fit_transform(y).astype(np.float32)
    reg = Ridge(alpha=alpha, solver='lsqr', tol=1e-5, max_iter=1000)
    reg.fit(x, latent)
    vectors = normalize(reg.predict(ex.transform(catalog.vessel_name.map(clean))).astype(np.float32))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'english': ex, 'korean': ky, 'svd': svd, 'regressor': reg,
                 'ids': catalog.vessel_id.astype(str).to_numpy(), 'vectors': vectors.astype(np.float16),
                 'training_pairs': len(pairs), 'alpha': alpha}, path, compress=3)


class TranslitIndex:
    def __init__(self, path, catalog):
        self.model = joblib.load(path)
        lookup = {str(v): i for i, v in enumerate(self.model['ids'])}
        self.ids = catalog.vessel_id.astype(str).to_numpy()
        self.vectors = self.model['vectors'][[lookup[v] for v in self.ids]].astype(np.float32)
        self.cache = {}

    def vector(self, name):
        name = str(name)
        if name not in self.cache:
            y = self.model['korean'].transform([jamo(name)])
            v = self.model['svd'].transform(y).astype(np.float32)
            self.cache[name] = normalize(v)[0]
        return self.cache[name]

    def scores(self, name):
        return self.vectors @ self.vector(name)

    def retrieve(self, name, k=96):
        scores = self.scores(name)
        k = min(k, len(scores))
        out = np.argpartition(-scores, k - 1)[:k]
        return out[np.lexsort((self.ids[out], -scores[out]))].tolist()
