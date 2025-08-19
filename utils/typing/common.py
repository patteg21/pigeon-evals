from typing import Literal, List, Dict, Any, Union, Optional
from pydantic import BaseModel
from datetime import datetime

Pooling = Literal["mean", "max", "weighted", "smooth_decay"]

FormType = Literal["10K", "10Q"]

EntityType = Literal["Part", "Item", "Table", "Page"]

ChartType = Literal["line", "bar", "candlestick", "area"]

LineStyle = Literal["solid", "dashed", "dotted"]

class TableImageData(BaseModel):
    headers: List[str]
    rows: List[List[Union[str, int, float]]]
    title: Optional[str] = None
    caption: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class DataPoint(BaseModel):
    x: Union[str, int, float, datetime]
    y: Union[int, float]
    label: Optional[str] = None

class FinancialDataPoint(BaseModel):
    date: Union[str, datetime]
    open: float
    high: float
    low: float
    close: float
    volume: Optional[int] = None
    label: Optional[str] = None

class ChartData(BaseModel):
    data: List[DataPoint]
    title: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    chart_type: ChartType = "line"
    line_style: LineStyle = "solid"
    color: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class BarChartData(BaseModel):
    categories: List[str]
    values: List[Union[int, float]]
    title: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    colors: Optional[List[str]] = None
    horizontal: bool = False
    metadata: Optional[Dict[str, Any]] = None

class FinancialChartData(BaseModel):
    data: List[FinancialDataPoint]
    title: Optional[str] = None
    ticker: Optional[str] = None
    chart_type: Literal["candlestick", "ohlc", "line"] = "candlestick"
    show_volume: bool = True
    metadata: Optional[Dict[str, Any]] = None
