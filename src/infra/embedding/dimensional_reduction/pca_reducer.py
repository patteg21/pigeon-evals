from __future__ import annotations
from typing import List, Iterable
import os
import numpy as np
import joblib
from sklearn.decomposition import PCA
import sklearn

from .base import BaseDimensionalReducer
from models.configs.embedding import DimensionReduction

from utils import logger
from utils.dry_run import dry_response

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
    
    def __init__(self, config: DimensionReduction):
        super().__init__(config)
        self.target_dim = self.config.dims
        self.seed = self.config.seed
        self.path = self.config.path or f"data/artifacts/pca_{self.target_dim}.joblib"
        self.model: PCA | None = None
    
    @property
    def name(self) -> str:
        return "PCA"
    
    @dry_response(mock_factory=lambda self, embeddings: self._mock_fit(embeddings))
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

    def _mock_fit(self, embeddings: List[List[float]]) -> "PCAReducer":
        """Mock fit method for dry run mode."""
        logger.warning("DRY RUN: Mocking PCA fit...")
        self.is_fitted = True
        return self
    
    @dry_response(mock_factory=lambda self, embeddings: self._mock_transform(embeddings))
    def transform(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Transform embeddings using fitted PCA."""
        if self.model is None:
            raise RuntimeError("PCA model not loaded/fitted. Call load() or fit() first.")

        X = _as_float32_array(embeddings)
        Z = self.model.transform(X)
        Z_normalized = _l2_normalize(Z)

        # Convert back to list of lists
        return Z_normalized.astype(np.float32).tolist()

    def _mock_transform(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Mock transform method for dry run mode."""
        logger.warning("DRY RUN: Mocking PCA transform...")
        # Return mock embeddings with target dimensions
        import random
        random.seed(42)
        return [[random.uniform(-1.0, 1.0) for _ in range(self.target_dim)] for _ in embeddings]
    
    @dry_response(mock_factory=lambda self, vec: self._mock_transform_one(vec))
    def transform_one(self, vec: List[float]) -> List[float]:
        """Transform a single embedding vector."""
        if self.model is None:
            raise RuntimeError("PCA model not loaded.")

        Z = self.model.transform(np.asarray([vec], dtype=np.float32))
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
        return Z[0].astype(np.float32).tolist()

    def _mock_transform_one(self, vec: List[float]) -> List[float]:
        """Mock transform_one method for dry run mode."""
        logger.warning("DRY RUN: Mocking PCA transform_one...")
        import random
        random.seed(42)
        return [random.uniform(-1.0, 1.0) for _ in range(self.target_dim)]
    
    @dry_response(mock_factory=lambda self, path=None: self._mock_save(path))
    def save(self, path: str = None) -> None:
        """Save the fitted PCA model."""
        if self.model is None:
            raise RuntimeError("Nothing to save: fit() a model first.")

        save_path = path or self.path
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        payload = {
            "model": self.model,
            "meta": {
                "sklearn_version": sklearn.__version__,
                "target_dim": self.target_dim,
                "seed": self.seed,
            },
        }
        joblib.dump(payload, save_path)

    def _mock_save(self, path: str = None) -> None:
        """Mock save method for dry run mode."""
        save_path = path or self.path
        logger.warning(f"DRY RUN: Mocking PCA save to {save_path}...")
    
    @dry_response(mock_factory=lambda self, path=None: self._mock_load(path))
    def load(self, path: str = None) -> "PCAReducer":
        """Load a fitted PCA model."""
        load_path = path or self.path

        if not os.path.exists(load_path):
            raise PCArtifactNotFoundError(f"PCA artifact not found at: {load_path}")

        payload = joblib.load(load_path)
        self.model = payload["model"]
        self.is_fitted = True
        return self

    def _mock_load(self, path: str = None) -> "PCAReducer":
        """Mock load method for dry run mode."""
        load_path = path or self.path
        logger.warning(f"DRY RUN: Mocking PCA load from {load_path}...")
        self.is_fitted = True
        return self

    def clear(self) -> None:
        """Clear saved PCA model artifacts."""
        if os.path.exists(self.path):
            os.remove(self.path)
            logger.info(f"Cleared PCA model artifact at {self.path}")

        # Also clear the in-memory model
        self.model = None
        self.is_fitted = False