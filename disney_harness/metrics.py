import numpy as np
import pandas as pd

def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    aligned = _align(y_true, y_pred)
    return float(np.mean(np.abs(aligned["y_true"] - aligned["y_pred"]))) if not aligned.empty else float("nan")

def mape(y_true: pd.Series, y_pred: pd.Series, eps: float = 1e-6) -> float:
    aligned = _align(y_true, y_pred)
    denom = np.maximum(np.abs(aligned["y_true"]), eps)
    return float(np.mean(np.abs((aligned["y_true"] - aligned["y_pred"]) / denom))) if not aligned.empty else float("nan")

def _align(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    df = pd.concat([y_true.rename("y_true"), y_pred.rename("y_pred")], axis=1).dropna()
    return df
