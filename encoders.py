# encoders.py
from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class PlantTargetEncoder(BaseEstimator, TransformerMixin):
    """Encode plant average price without target leakage.

    Computes plant-specific average price on training data only,
    then applies those averages to both train and test sets.
    Adds 'Plant_avg_price'.
    """
    def __init__(self):
        self.plant_avg_price_: Dict[object, float] = {}
        self.global_avg_price_: float = 0.0

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame) or "Plant" not in X.columns:
            raise ValueError("PlantTargetEncoder requires DataFrame input with 'Plant' column")
        df = X.copy()
        df["_target"] = y
        self.plant_avg_price_ = df.groupby("Plant")["_target"].mean().to_dict()
        self.global_avg_price_ = float(np.asarray(y, dtype=float).mean())
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame) or "Plant" not in X.columns:
            raise ValueError("PlantTargetEncoder requires DataFrame input with 'Plant' column")
        X_out = X.copy()
        X_out["Plant_avg_price"] = X_out["Plant"].map(self.plant_avg_price_).fillna(self.global_avg_price_)
        return X_out


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the same derived numeric features as training."""
    enriched = df.copy()

    # Polynomial features
    enriched["Travel_sq"] = enriched["Travel"] ** 2
    enriched["Yardage_filled_sq"] = enriched["Yardage_filled"] ** 2
    enriched["Travel_cube"] = enriched["Travel"] ** 3
    enriched["Yardage_filled_cube"] = enriched["Yardage_filled"] ** 3

    # Interaction features
    enriched["Travel_times_Yardage"] = enriched["Travel"] * enriched["Yardage_filled"]

    # Ratio features
    denom = enriched["Travel"].replace(0, np.nan)
    ratio = enriched["Yardage_filled"] / denom
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    enriched["Yardage_per_travel"] = ratio.fillna(0.0)

    # Inverse ratio
    yardage_denom = enriched["Yardage_filled"].replace(0, np.nan)
    inverse_ratio = enriched["Travel"] / yardage_denom
    inverse_ratio = inverse_ratio.replace([np.inf, -np.inf], np.nan)
    enriched["Travel_per_yardage"] = inverse_ratio.fillna(0.0)

    # Square root features
    enriched["Travel_sqrt"] = np.sqrt(enriched["Travel"].clip(lower=0))
    enriched["Yardage_filled_sqrt"] = np.sqrt(enriched["Yardage_filled"].clip(lower=0))

    # Log features
    enriched["log_Travel"] = np.log1p(enriched["Travel"].clip(lower=0))
    enriched["log_Yardage_filled"] = np.log1p(enriched["Yardage_filled"].clip(lower=0))

    return enriched


def normalise_plant(value: object) -> Optional[str]:
    """Return a cleaned plant identifier string, or None when not provided."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        numeric = float(text)
        if np.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
        return str(numeric)
    except (ValueError, TypeError):
        return text
