from __future__ import annotations
from typing import List, Dict, Any, Iterable
import os
import numpy as np
import joblib
from sklearn.decomposition import PCA
import sklearn

from .base import BaseDimensionalReducer

from evals.src.utils import logger

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


class PCAReducer(BaseDimensionalReducer):
    """PCA-based dimensional reduction for embeddings."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.target_dim = self.config.get("dims", 512)
        self.seed = self.config.get("seed", 42)
        self.path = self.config.get("path", f"artifacts/pca_{self.target_dim}.joblib")
        self.model: PCA | None = None
    
    @property
    def name(self) -> str:
        return "PCA"
    
    def fit(self, embeddings: List[List[float]]) -> "PCAReducer":
        """Fit PCA on embeddings data."""
        logger.warning("PCA is being Fit...")
        X = _as_float32_array(embeddings)
        n_comp = min(self.target_dim, X.shape[1])
        # Train PCA model on embedding data
        self.model = PCA(n_components=n_comp, random_state=self.seed).fit(X)
        self.is_fitted = True
        logger.warning("PCA finishing fitting...")

        return self
    
    def transform(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Transform embeddings using fitted PCA."""
        if self.model is None:
            raise RuntimeError("PCA model not loaded/fitted. Call load() or fit() first.")
        
        X = _as_float32_array(embeddings)
        Z = self.model.transform(X)
        Z_normalized = _l2_normalize(Z)
        
        # Convert back to list of lists
        return Z_normalized.astype(np.float32).tolist()
    
    def transform_one(self, vec: List[float]) -> List[float]:
        """Transform a single embedding vector."""
        if self.model is None:
            raise RuntimeError("PCA model not loaded.")
        
        Z = self.model.transform(np.asarray([vec], dtype=np.float32))
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
        return Z[0].astype(np.float32).tolist()
    
    def save(self, path: str = None) -> None:
        """Save the fitted PCA model."""
        if self.model is None:
            raise RuntimeError("Nothing to save: fit() a model first.")
        
        save_path = path or self.path
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        
        payload: Dict = {
            "model": self.model,
            "meta": {
                "sklearn_version": sklearn.__version__,
                "target_dim": self.target_dim,
                "seed": self.seed,
            },
        }
        joblib.dump(payload, save_path)
    
    def load(self, path: str = None) -> "PCAReducer":
        """Load a fitted PCA model."""
        load_path = path or self.path
        
        if not os.path.exists(load_path):
            raise PCArtifactNotFoundError(f"PCA artifact not found at: {load_path}")
        
        payload = joblib.load(load_path)
        self.model = payload["model"]
        self.is_fitted = True
        return self