"""Frozen text encoders (no gradients flow into these from the TFM).

Default: TF-IDF (no downloads, no GPU, reproducible) - the same encoder used
for the coherence / ablation numbers in the notebook. Swap for a MiniLM/E5
sentence transformer later without changing the fusion interface: both expose
``fit_transform(texts)`` / ``transform(texts)`` returning numpy arrays.
"""

import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class FrozenTfidfEncoder:
    def __init__(self, max_features=300):
        self.max_features = max_features
        self.vec = TfidfVectorizer(max_features=max_features)

    def fit_transform(self, texts):
        return self.vec.fit_transform(texts).toarray().astype(np.float32)

    def transform(self, texts):
        return self.vec.transform(texts).toarray().astype(np.float32)

    @property
    def dim(self):
        return len(self.vec.vocabulary_)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.vec, f)

    @classmethod
    def load(cls, path, max_features=300):
        enc = cls(max_features=max_features)
        with open(path, "rb") as f:
            enc.vec = pickle.load(f)
        return enc
