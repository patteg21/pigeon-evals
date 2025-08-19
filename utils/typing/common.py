from typing import Literal, List, Dict, Any, Union, Optional
from pydantic import BaseModel

Pooling = Literal["mean", "max", "weighted", "smooth_decay"]

FormType = Literal["10K", "10Q"]

EntityType = Literal["Part", "Item", "Table", "Page"]


class TableImageData(BaseModel):
    headers: List[str]
    rows: List[List[Union[str, int, float]]]
    title: Optional[str] = None
    caption: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
