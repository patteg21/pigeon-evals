from typing import Literal

Pooling = Literal["mean", "max", "weighted", "smooth_decay"]

FormType = Literal["10K", "10Q"]

EntityType = Literal["Part", "Item", "Table", "Page"]

DistanceMetric = Literal["cosine", "euclidean"]
