# app.py
from __future__ import annotations
import json, os, sys
from typing import List

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Import your training-time symbols so unpickling can resolve them
from encoders import PlantTargetEncoder, add_derived_features, normalise_plant

# Shim: your pickle likely references __main__.PlantTargetEncoder
sys.modules['__main__'].__dict__['PlantTargetEncoder'] = PlantTargetEncoder

st.set_page_config(page_title="PCI Bid Analaysis by Market", layout="centered")
st.title("PCI Bids: Market Price Estimation")
st.text("Market-aware price estimate per yd with uncertainty.")
st.text("Does not account for mix design: use estimation for environemental impact calculations and then manually adjust for mix design")
st.caption("Model trained on PCI bid data from 2025.")


@st.cache_resource
def load_artifacts():
    here = os.path.dirname(__file__)
    model_path = os.path.join(here, "price_model.joblib")
    report_path = os.path.join(here, "price_model_report.json")

    model = joblib.load(model_path)

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    # Try to extract categorical choices from the fitted pipeline
    plants, quarters = [], ["Q1","Q2","Q3","Q4"]
    try:
        pre = model.named_steps["preprocess"]
        ohe = pre.named_transformers_["plant"].named_steps["onehot"]
        cats = ohe.categories_
        # 0 -> Plant, 1 -> Quarter (per build_model)
        plants = list(cats[0])
        if len(cats) > 1:
            quarters = [q for q in cats[1] if isinstance(q, str)] or quarters
    except Exception:
        pass

    # Feature schema tells us exactly which columns the model expects
    feature_schema: List[str] = report.get("feature_schema") or []

    # Grab RMSE/R2 from various possible report layouts
    metrics = report.get("metrics", {}) or {}
    holdout = (
        metrics.get("holdout_test")
        or metrics.get("holdout")      # fallback if older key
        or {}
    )
    rmse = float(holdout.get("rmse") or metrics.get("rmse", 0.0) or 0.0) or None
    r2 = holdout.get("r2") or metrics.get("r2")

    return model, report, plants, quarters, feature_schema, rmse, r2

model, report, plant_choices, quarter_choices, feature_schema, report_rmse, report_r2 = load_artifacts()

# Sort plant choices numerically when possible
def sort_plants(plants):
    def to_number_or_str(p):
        try:
            return (0, float(p))  # numeric first
        except (ValueError, TypeError):
            return (1, str(p))    # then alphabetic
    return sorted(plants, key=to_number_or_str)

plant_choices = sort_plants(plant_choices)
# ----- UI -----
plant = st.selectbox("Plant", plant_choices) if plant_choices else st.text_input("Plant (ID or code)")
# Travel Distance Input (text -> float)
travel_str = st.text_input("Travel distance (miles)", value="10.0")
try:
    travel = float(travel_str)
    if travel < 0:
        st.error("❌ Travel distance cannot be negative.")
        st.stop()
except ValueError:
    st.error("❌ Please enter a valid number for travel distance.")
    st.stop()

#Yardage Input (no +/- buttons)
yardage_str = st.text_input("Yardage", value="1000")
try:
    yardage = float(yardage_str)
    if yardage < 0:
        st.error("❌ Yardage cannot be negative.")
        st.stop()
except ValueError:
    st.error("❌ Please enter a valid number for yardage.")
    st.stop()

quarter_ui = st.selectbox("Quarter", ["UNKNOWN"] + quarter_choices)

st.divider()

def prepare_row(
    plant_val, quarter_val, travel_val, yardage_val, feature_schema: List[str]
) -> pd.DataFrame:
    """Create a one-row DataFrame that matches the training schema.

    - Computes Yardage_filled and Yardage_missing_flag
    - Applies add_derived_features (same as training)
    - Ensures that every column in feature_schema exists; fills unknowns with NaN
      so your SimpleImputer can apply training medians.
    """
    yardage_missing = pd.isna(yardage_val)
    yardage_filled = float(yardage_val) if not yardage_missing else np.nan

    base = pd.DataFrame([{
        "Plant": normalise_plant(plant_val),
        "Quarter": str(quarter_val).strip().upper() if quarter_val else "UNKNOWN",
        "Travel": float(travel_val),
        "Yardage": np.nan if yardage_missing else float(yardage_val),
        "Yardage_filled": yardage_filled,
        "Yardage_missing_flag": int(bool(yardage_missing)),
    }])

    # Add derived numeric features exactly like training
    enriched = add_derived_features(base.assign(Price=np.nan))

    # If the report includes a schema, enforce it; otherwise fall back to known core features
    if feature_schema:
        X = pd.DataFrame(columns=feature_schema)
        # copy columns we have
        for col in enriched.columns.intersection(X.columns):
            X[col] = enriched[col].values
        # any missing columns stay as NaN → imputed by the pipeline
        # (numeric imputer handles these; categorical are handled by the cat pipeline)
        # Ensure correct dtypes where possible
        return X[feature_schema]
    else:
        # Minimal fallback: include the core columns (the pipeline's ColumnTransformer
        # will fail if it expects more; feature_schema is strongly recommended)
        core = ["Plant","Quarter","Travel","Yardage_filled","Yardage_missing_flag"]
        for c in core:
            if c not in enriched:
                enriched[c] = np.nan
        return enriched[core]

if st.button("Estimate"):
    X = prepare_row(plant, quarter_ui, travel, yardage, feature_schema)
    pred = float(model.predict(X)[0])

    # Uncertainty: prefer holdout RMSE from report; otherwise show ±10 as a fallback
    rmse = report_rmse if report_rmse is not None and report_rmse > 0 else 10.0
    lower, upper = pred - rmse, pred + rmse

    st.subheader(f"Estimated market price: ${pred:,.2f} / yd")
    st.write(f"Uncertainty (~1 RMSE): **${lower:,.2f} — ${upper:,.2f}**")

    if report_r2 is not None:
        st.caption(f"Validation (hold-out): R² ≈ {float(report_r2):.3f}, RMSE ≈ ${rmse:,.2f}")
st.text("Adjust estimate based on mix design, admixtures, and other factors not included in this model.")