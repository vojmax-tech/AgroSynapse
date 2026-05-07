"""Shared transformer utilities used by training and feature transform scripts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ToStringTransformer(BaseEstimator, TransformerMixin):
    """Normalize categorical inputs to consistent string values for encoders."""

    def fit(self, X, y=None):  # type: ignore[override]
        return self

    def transform(self, X):  # type: ignore[override]
        if isinstance(X, pd.DataFrame):
            out = X.copy()
            for col in out.columns:
                out[col] = out[col].astype("string").fillna("__missing__").astype(str)
            return out

        arr = np.asarray(X, dtype=object)
        arr = np.where(pd.isna(arr), "__missing__", arr)
        return arr.astype(str)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([], dtype=object)
        return np.asarray(input_features, dtype=object)


class ToNumericTransformer(BaseEstimator, TransformerMixin):
    """Coerce numeric-like columns to float and map non-numeric values to NaN."""

    def fit(self, X, y=None):  # type: ignore[override]
        return self

    def transform(self, X):  # type: ignore[override]
        if isinstance(X, pd.DataFrame):
            return X.apply(pd.to_numeric, errors="coerce")

        arr = np.asarray(X, dtype=object)
        coerced = pd.DataFrame(arr).apply(pd.to_numeric, errors="coerce")
        return coerced.to_numpy()

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([], dtype=object)
        return np.asarray(input_features, dtype=object)
