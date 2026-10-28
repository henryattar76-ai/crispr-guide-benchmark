"""Two deliberately simple baselines.

If a deep model cannot clearly beat these, its added value is small. Both only
need the 30-nt context, so they run on any harmonized dataset.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from harmonize import GUIDE_END, GUIDE_START

NUC_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}


def gc_content(contexts) -> np.ndarray:
    """GC fraction of the 20-nt protospacer (no training required)."""
    return np.array(
        [sum(c in "GC" for c in ctx[GUIDE_START:GUIDE_END]) / (GUIDE_END - GUIDE_START) for ctx in contexts]
    )


def one_hot(contexts) -> np.ndarray:
    """Position-specific one-hot encoding of the 30-nt context (30 x 4 = 120 features)."""
    contexts = list(contexts)
    X = np.zeros((len(contexts), 30 * 4))
    for i, ctx in enumerate(contexts):
        for pos, nt in enumerate(ctx):
            X[i, pos * 4 + NUC_INDEX[nt]] = 1.0
    return X


class PositionLogit:
    """Logistic regression on position-specific nucleotides.

    Trained to separate guides above vs below a rank threshold in the training
    dataset (default: median), then scored with P(high activity).
    """

    def __init__(self, threshold: float = 0.5, C: float = 1.0):
        self.threshold = threshold
        self.model = LogisticRegression(C=C, max_iter=2000)

    def fit(self, contexts, label_rank):
        y = (np.asarray(label_rank) > self.threshold).astype(int)
        self.model.fit(one_hot(contexts), y)
        return self

    def predict(self, contexts) -> np.ndarray:
        return self.model.predict_proba(one_hot(contexts))[:, 1]
