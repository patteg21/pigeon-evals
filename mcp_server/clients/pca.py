from __future__ import annotations
from typing import Iterable, Dict
import os
import numpy as np
import joblib
from sklearn.decomposition import PCA
import sklearn


class PCArtifactNotFoundError(FileNotFoundError):
    """Raised when the expected PCA artifact file is not found."""
    pass


def _l2_normalize(X: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)

def _as_float32_array(rows: Iterable[Iterable[float]]) -> np.ndarray:
    X = np.asarray(list(rows), dtype=np.float32)
    if np.any(~np.isfinite(X)):
        raise ValueError("Embeddings contain NaN/Inf.")
    return X


class PCALoader:
    """
    Train once, save, and later load to transform new embeddings consistently.
    """
    def __init__(self, path: str = "artifacts/pca_512.joblib", target_dim: int = 512, seed: int = 42):
        self.path = path
        self.target_dim = target_dim
        self.seed = seed
        self.model: PCA | None = None

    def fit(self, embeddings: Iterable[Iterable[float]]) -> "PCALoader":
        X = _as_float32_array(embeddings)
        n_comp = min(self.target_dim, X.shape[1])
        # Train PCA model on embedding data
        self.model = PCA(n_components=n_comp, random_state=self.seed).fit(X)
        return self

    def transform(self, embeddings: Iterable[Iterable[float]]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("PCA model not loaded/fitted. Call load() or fit() first.")
        X = _as_float32_array(embeddings)
        Z = self.model.transform(X)
        return _l2_normalize(Z)

    def transform_one(self, vec: list[float]) -> list[float]:
        if self.model is None:
            raise RuntimeError("PCA model not loaded.")
        Z = self.model.transform(np.asarray([vec], dtype=np.float32))
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
        return Z[0].astype(np.float32).tolist()

    def save(self) -> None:
        if self.model is None:
            raise RuntimeError("Nothing to save: fit() a model first.")
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        payload: Dict = {
            "model": self.model,
            "meta": {
                "sklearn_version": sklearn.__version__,
                "target_dim": self.target_dim,
                "seed": self.seed,
            },
        }
        joblib.dump(payload, self.path)

    def load(self) -> "PCALoader":
        if not os.path.exists(self.path):
            raise PCArtifactNotFoundError(f"PCA artifact not found at: {self.path}")
        payload = joblib.load(self.path)
        self.model = payload["model"]
        return self
