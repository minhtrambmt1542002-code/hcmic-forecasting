# -*- coding: utf-8 -*-
"""
HCMIC EMS Data Mining & Warehouse Planning System
=================================================

Run with:   streamlit run hcmic_ems_warehouse_planning.py

Required packages (also suitable for requirements.txt):
    streamlit
    pandas
    numpy
    plotly
    scikit-learn
    openpyxl
    xgboost
    lightgbm
    catboost

Framework implemented
---------------------
STAGE 1 - Data Mining & Predictive Forecasting
    Fixed historical design: 17 months = 3 feature warm-up + 8 pre-test modelling + 6 final holdout
    Historical EMS Data -> Data Preprocessing -> Feature Engineering
    -> Demand Pattern Analysis -> CV-Based Statistical Demand Segmentation
    -> Machine Learning Predictive Modeling -> Model Validation & Selection
    -> 6-Month Raw Material Inventory Forecast

    [ Exogenous Forecast Input ]

STAGE 2 - Dynamic Warehouse Planning
    Inventory Dynamics -> Warehouse Operational Requirement Estimation
    -> Pallet / Bin Requirement -> Warehouse Capacity Evaluation
    -> Contract Flexibility -> Warehouse Cost Evaluation
    -> Warehouse Planning Decision

Stage 2 never retrains or feeds back into Stage 1: it only consumes the
6-month RawMaterialInventory forecast as an exogenous input.
"""

import html
import math
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =============================================================================
# CONSTANTS
# =============================================================================
APP_TITLE = "HCMIC EMS Data Mining & Warehouse Planning System"
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---- Data schema -------------------------------------------------------------
REQUIRED_COLS = [
    "Month", "ProfitCenter", "ProductionRevenue", "RawMaterialInventory",
    "ReceivingTransaction", "LocationTransferTransaction", "ShippingTransaction",
    "FG_Pallet", "RM_Pallet", "NoOfBin",
]
NUMERIC_COLS = [
    "ProductionRevenue", "RawMaterialInventory", "ReceivingTransaction",
    "LocationTransferTransaction", "ShippingTransaction",
    "FG_Pallet", "RM_Pallet", "NoOfBin",
]
TARGET = "RawMaterialInventory"           # primary Stage 1 forecasting target
MONTH_INPUT_FORMAT = "%b'%y"               # e.g. Mar'26
MONTH_LABEL_FORMAT = "%b'%y"

# ---- Stage 1: feature engineering / modelling ---------------------------------
FEATURE_COLS = [
    "RM_Lag1", "RM_Lag2", "RM_MA3", "RM_STD3", "RM_CV3",
    "Year", "MonthNumber", "Quarter", "MonthSin", "MonthCos",
]
ROLLING_WINDOW = 3
FEATURE_WARMUP_ROWS = 3                    # rows lost per ProfitCenter to shift(1).rolling(3)
FORECAST_HORIZON = 6                       # production forecast: next 6 months
TEST_HORIZON = 6                           # final hold-out test: last 6 months
DEFAULT_VALIDATION_FRACTION = 0.20         # later 20% of pre-test rows = validation
MIN_VALIDATION_ROWS = 2
MIN_TRAIN_FIT_ROWS = 4
REQUIRED_HISTORY_MONTHS = 17               # fixed research design: 3 warm-up + 8 pre-test + 6 test
PRETEST_MODELLING_ROWS = 8                  # usable modelling rows after the 3-month feature warm-up
NEAR_ZERO_TOL = 1e-6                       # |actual| below this is excluded from MAPE / WMAPE
EPS = 1e-9

VALIDATION_MODES = {
    "Recursive multi-step (same protocol as the final forecast)": "recursive",
    "One-step-ahead (actual lag features inside validation set)": "one_step",
}

# ---- CV-based statistical demand segmentation ---------------------------------
CV_STABLE_MAX = 0.20                       # CV <  0.20        -> Stable
CV_VOLATILE_MIN = 0.50                     # CV >= 0.50        -> Volatile (else Moderate)

# ---- Stage 2: warehouse planning (unchanged from the original application) -----
ALPHA = 0.20                               # capacity flexibility / buffer
BETA = 0.10                                # capacity reduction / adjustment factor
REVENUE_WINDOW_MONTHS = 12
RATIO_CLIP_MAX = 5
HOLDING_COST_RATE = 0.15
SHORTAGE_COST_RATE = 0.30
CAPACITY_COST_RATE = 2.00
TRANSACTION_COST_RATE = 0.50

# Chart colours (navy / blue / teal academic palette)
COLOR_ACTUAL = "#1F3A5F"
COLOR_TEST = "#E07B00"
COLOR_FUTURE = "#0F7C8A"


class DataError(Exception):
    """Raised for fatal input-data problems (the app shows the message and stops)."""


# =============================================================================
# GENERIC UI / UTILITY HELPERS
# =============================================================================
def show_df(df, **kwargs):
    """st.dataframe wrapper that works across Streamlit versions."""
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except Exception:
        try:
            st.dataframe(df, use_container_width=True, **kwargs)
        except Exception:
            st.dataframe(df.astype(str), **kwargs)   # e.g. Arrow type problems


def show_chart(fig):
    """st.plotly_chart wrapper that works across Streamlit versions."""
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


def show_messages(messages):
    """Display (level, text) messages produced by the pipeline functions."""
    for level, text in messages:
        if level == "error":
            st.error(text)
        elif level == "warning":
            st.warning(text)
        else:
            st.info(text)


def round_df(df, decimals=2):
    """Round every float column (display / CSV friendliness)."""
    out = df.copy()
    float_cols = out.select_dtypes(include=["float", "float32", "float64"]).columns
    out[float_cols] = out[float_cols].round(decimals)
    return out


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def month_diff(later, earlier):
    """Whole months between two timestamps (later - earlier)."""
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


# =============================================================================
# STAGE 1 - STEP 1: DATA PREPROCESSING
# =============================================================================
def parse_month_column(series):
    """Parse the Month column to month-start timestamps (primary format: Mar'26)."""
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
    else:
        text = series.astype(str).str.strip()
        parsed = pd.to_datetime(text, format=MONTH_INPUT_FORMAT, errors="coerce")
        failed = parsed.isna()
        if failed.any():                       # tolerate other common date formats
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed.loc[failed] = pd.to_datetime(text[failed], errors="coerce")
    return parsed.dt.to_period("M").dt.to_timestamp()


def build_data_quality_table(df):
    """Per-ProfitCenter history length, date span, eligibility and gap diagnostics."""
    rows = []
    for pc, g in df.groupby("ProfitCenter", sort=True):
        g = g.sort_values("MonthDate")
        first, last = g["MonthDate"].min(), g["MonthDate"].max()
        span = month_diff(last, first) + 1
        zero_activity = bool((g["TotalActivity"] <= 0).all())
        data_length = len(g)
        missing_in_span = span - data_length

        if zero_activity:
            eligible = "No"
            reason = "Zero activity"
        elif data_length != REQUIRED_HISTORY_MONTHS:
            eligible = "No"
            reason = f"Insufficient / invalid history: exactly {REQUIRED_HISTORY_MONTHS} months required"
        elif missing_in_span > 0:
            eligible = "No"
            reason = "Missing month(s) inside the historical span"
        else:
            eligible = "Yes"
            reason = "Eligible"

        rows.append({
            "ProfitCenter": pc,
            "DataLength": data_length,
            "FirstMonth": first.strftime(MONTH_LABEL_FORMAT),
            "LastMonth": last.strftime(MONTH_LABEL_FORMAT),
            "MissingMonthsInSpan": missing_in_span,
            "ZeroActivity": zero_activity,
            "ConstantRMSeries": bool(g[TARGET].nunique() <= 1),
            "MLEligible": eligible,
            "Reason": reason,
        })
    return pd.DataFrame(rows)


def preprocess_data(raw_df):
    """
    Clean the uploaded dataset and build a monthly series per ProfitCenter.

    Fixed research design: each ML-eligible ProfitCenter must contain exactly
    17 consecutive historical months. Zero-activity ProfitCenters are retained
    for data-quality reporting but excluded from ML forecasting.

    Returns (clean_df, report). Raises DataError for fatal input problems.
    """
    report = {"messages": []}
    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise DataError(f"Missing Columns: {missing_cols}")

    df = df[REQUIRED_COLS].copy()
    report["rows_input"] = len(df)

    # ---- ProfitCenter ----
    df["ProfitCenter"] = (
        df["ProfitCenter"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    )
    bad_pc = df["ProfitCenter"].isin(["", "nan", "None", "NaT", "<NA>"])
    report["rows_missing_profitcenter"] = int(bad_pc.sum())
    df = df[~bad_pc].copy()

    # ---- Numeric conversion + missing values ----
    missing_filled = {}
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        n_missing = int(df[col].isna().sum())
        if n_missing:
            missing_filled[col] = n_missing
    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(0)
    report["missing_filled"] = missing_filled
    if missing_filled:
        report["messages"].append((
            "warning",
            f"Missing / non-numeric values were replaced by 0 in: {missing_filled}. "
            "Check that a filled RawMaterialInventory value does not hide a true gap."))

    # ---- Month ----
    df["MonthDate"] = parse_month_column(df["Month"])
    if df["MonthDate"].isna().all():
        raise DataError("Month format invalid. Use a valid monthly date such as Mar'26.")
    n_bad_dates = int(df["MonthDate"].isna().sum())
    report["rows_invalid_date"] = n_bad_dates
    if n_bad_dates:
        report["messages"].append(("warning", f"{n_bad_dates} row(s) with an invalid Month were dropped."))
    df = df.dropna(subset=["MonthDate"]).copy()

    # ---- Duplicate (ProfitCenter, Month) rows ----
    dup_mask = df.duplicated(["ProfitCenter", "MonthDate"], keep=False)
    report["rows_duplicate_merged"] = int(dup_mask.sum())
    if dup_mask.any():
        df = df.groupby(["ProfitCenter", "MonthDate"], as_index=False)[NUMERIC_COLS].sum()
        report["messages"].append((
            "warning",
            f"{int(dup_mask.sum())} rows shared the same ProfitCenter/Month and were merged by summation."))

    df["Month"] = df["MonthDate"].dt.strftime(MONTH_LABEL_FORMAT)

    # ---- Activity flag: retain zero-activity ProfitCenters for reporting ----
    df["TotalActivity"] = (
        df["RawMaterialInventory"]
        + df["ReceivingTransaction"]
        + df["ShippingTransaction"]
    )
    df["ZeroActivity"] = df["TotalActivity"] <= 0

    # IMPORTANT: do NOT delete zero-activity rows here.
    report["rows_inactive_removed"] = 0
    report["zero_activity_rows"] = int(df["ZeroActivity"].sum())
    report["zero_activity_profitcenters"] = sorted(
        df.loc[df["ZeroActivity"], "ProfitCenter"].unique().tolist()
    )
    if report["zero_activity_profitcenters"]:
        report["messages"].append((
            "warning",
            "Zero-activity ProfitCenter(s) are retained in the data-quality report but excluded from ML forecasting: "
            + ", ".join(report["zero_activity_profitcenters"])))

    df = df.sort_values(["ProfitCenter", "MonthDate"]).reset_index(drop=True)
    report["rows_output"] = len(df)
    report["quality_table"] = build_data_quality_table(df)

    q = report["quality_table"]
    if (q["DataLength"] != REQUIRED_HISTORY_MONTHS).any():
        bad = q.loc[q["DataLength"] != REQUIRED_HISTORY_MONTHS, ["ProfitCenter", "DataLength"]]
        report["messages"].append((
            "warning",
            f"The application requires exactly {REQUIRED_HISTORY_MONTHS} historical months per ML-eligible ProfitCenter. "
            f"Non-conforming ProfitCenters: {bad.to_dict('records')}"))
    if (q["MissingMonthsInSpan"] > 0).any():
        report["messages"].append((
            "warning",
            "Some ProfitCenters have missing months inside their date span. "
            "Those ProfitCenters are not eligible for ML forecasting."))
    if q["ConstantRMSeries"].any():
        report["messages"].append((
            "warning",
            "Constant RawMaterialInventory series detected for: "
            f"{q.loc[q['ConstantRMSeries'], 'ProfitCenter'].tolist()}. "
            "Zero-activity series are excluded from ML; non-zero constant series remain eligible if they satisfy the 17-month rule."))
    return df, report


# =============================================================================
# STAGE 1 - STEP 2: FEATURE ENGINEERING (leakage-free)
# =============================================================================
def add_calendar_features(frame):
    """Add Year, MonthNumber, Quarter and cyclical MonthSin/MonthCos from MonthDate."""
    out = frame.copy()
    dates = out["MonthDate"]
    out["Year"] = dates.dt.year
    out["MonthNumber"] = dates.dt.month
    out["Quarter"] = dates.dt.quarter
    out["MonthSin"] = np.sin(2 * np.pi * out["MonthNumber"] / 12)
    out["MonthCos"] = np.cos(2 * np.pi * out["MonthNumber"] / 12)
    return out


def create_features(df):
    """
    Leakage-free features per ProfitCenter.

    RM_Lag1, RM_Lag2          : RawMaterialInventory shifted by 1 / 2 months
    RM_MA3, RM_STD3           : shift(1).rolling(3) mean / std  (current month EXCLUDED)
    RM_CV3                    : RM_STD3 / RM_MA3
    Year, MonthNumber, Quarter, MonthSin, MonthCos : calendar features

    The first FEATURE_WARMUP_ROWS rows of each ProfitCenter have NaN features
    (insufficient history) and are excluded from modelling.
    """
    out = df.sort_values(["ProfitCenter", "MonthDate"]).reset_index(drop=True).copy()
    grp = out.groupby("ProfitCenter")[TARGET]

    out["RM_Lag1"] = grp.shift(1)
    out["RM_Lag2"] = grp.shift(2)
    out["RM_MA3"] = grp.transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW).mean())
    out["RM_STD3"] = grp.transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW).std())

    ma, sd = out["RM_MA3"], out["RM_STD3"]
    cv = sd / ma.where(ma.abs() > EPS)                         # NaN when the mean is ~0
    out["RM_CV3"] = cv.mask(ma.abs().le(EPS) & sd.notna(), 0.0)  # all-zero window -> CV = 0

    return add_calendar_features(out)


def build_feature_row(history, month_date):
    """
    Feature vector (1-row DataFrame) for the month AFTER `history`.
    Uses exactly the same definitions as create_features(), computed from the last
    three values of `history` (historical actuals and/or previous predictions).
    """
    h = np.asarray(history, dtype=float)
    if len(h) < ROLLING_WINDOW:
        raise ValueError(f"At least {ROLLING_WINDOW} historical values are required to build features.")
    window = h[-ROLLING_WINDOW:]
    ma = float(window.mean())
    sd = float(window.std(ddof=1))                              # same ddof as pandas rolling().std()
    cv = sd / ma if abs(ma) > EPS else 0.0
    row = pd.DataFrame({
        "MonthDate": [pd.Timestamp(month_date)],
        "RM_Lag1": [h[-1]], "RM_Lag2": [h[-2]],
        "RM_MA3": [ma], "RM_STD3": [sd], "RM_CV3": [cv],
    })
    return add_calendar_features(row)[FEATURE_COLS]


# =============================================================================
# STAGE 1 - STEP 3: DEMAND PATTERN ANALYSIS + CV-BASED STATISTICAL SEGMENTATION
# =============================================================================
def classify_trend(growth_pct):
    """Trend label from first-to-last growth % (thresholds unchanged from the original app)."""
    if growth_pct >= 50:
        return "Strong Increasing"
    if growth_pct >= 15:
        return "Increasing"
    if growth_pct >= 5:
        return "Slight Increasing"
    if growth_pct <= -50:
        return "Strong Decreasing"
    if growth_pct <= -15:
        return "Decreasing"
    if growth_pct <= -5:
        return "Slight Decreasing"
    return "Stable"


def analyze_demand_patterns(df):
    """Per-ProfitCenter descriptive statistics of RawMaterialInventory (data-mining step)."""
    rows = []
    for pc, g in df.groupby("ProfitCenter", sort=True):
        g = g.sort_values("MonthDate")
        rm = g[TARGET]
        n = len(g)
        rm_mean = float(rm.mean())
        rm_std = float(rm.std()) if n >= 2 else np.nan
        if n < 2:
            cv = np.nan
        else:
            cv = rm_std / rm_mean if rm_mean > EPS else 0.0

        if n < 2:
            growth_pct, trend = 0.0, "Insufficient Data"
        else:
            first_v, last_v = float(rm.iloc[0]), float(rm.iloc[-1])
            growth_pct = (last_v - first_v) / first_v * 100 if first_v != 0 else 0.0
            trend = classify_trend(growth_pct)

        zero_activity = bool((g["TotalActivity"] <= 0).all())
        if zero_activity:
            trend = "Zero Activity"

        rows.append({
            "ProfitCenter": pc,
            "DataLength": n,
            "RM_Mean": rm_mean,
            "RM_STD": rm_std,
            "CV": cv,
            "RM_Min": float(rm.min()),
            "RM_Max": float(rm.max()),
            "ZeroRM_Months": int((rm <= 0).sum()),
            "ZeroActivity": zero_activity,
            "MLEligible": "No" if zero_activity else "Pending",
            "ReceivingTransaction_Mean": float(g["ReceivingTransaction"].mean()),
            "NoOfBin_Mean": float(g["NoOfBin"].mean()),
            "GrowthPercent": round(growth_pct, 2),
            "Trend": trend,
        })
    return pd.DataFrame(rows)


def segment_customers(pattern_df):
    """
    CV-Based Statistical Demand Segmentation (statistical thresholds, NOT clustering).
    Zero-activity ProfitCenters are labelled separately and are not treated as Stable.
    """
    out = pattern_df.copy()
    out["DemandSegment"] = np.select(
        [out["ZeroActivity"], out["CV"].isna(), out["CV"] >= CV_VOLATILE_MIN, out["CV"] >= CV_STABLE_MAX],
        ["Zero Activity", "Undetermined", "Volatile", "Moderate"],
        default="Stable",
    )
    out["DemandBehavior"] = np.where(
        out["ZeroActivity"],
        "Zero activity, excluded from ML",
        out["DemandSegment"] + " variability, " + out["Trend"].str.lower() + " level",
    )
    return out


# =============================================================================
# STAGE 1 - STEP 4: MACHINE LEARNING PREDICTIVE MODELING
# =============================================================================
def build_ml_models():
    """
    Return (factories, unavailable).
    factories   : {model name: zero-argument callable returning a fresh regressor}
    unavailable : {model name: reason the library could not be imported}
    Hyper-parameters are deliberately conservative for small monthly datasets.
    """
    factories, unavailable = {}, {}

    try:
        from xgboost import XGBRegressor
        factories["XGBoost"] = lambda: XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3, min_child_weight=1,
            subsample=1.0, colsample_bytree=0.8, reg_lambda=1.0,
            objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=1, verbosity=0)
    except Exception as exc:                                   # ImportError, OSError, ...
        unavailable["XGBoost"] = f"{type(exc).__name__}: {exc}"

    try:
        from lightgbm import LGBMRegressor
        factories["LightGBM"] = lambda: LGBMRegressor(
            n_estimators=200, learning_rate=0.05, num_leaves=7, max_depth=3,
            min_child_samples=2, min_data_in_bin=1,            # tiny monthly datasets
            subsample=1.0, colsample_bytree=0.8, random_state=RANDOM_STATE,
            n_jobs=1, verbose=-1)
    except Exception as exc:
        unavailable["LightGBM"] = f"{type(exc).__name__}: {exc}"

    try:
        from catboost import CatBoostRegressor
        factories["CatBoost"] = lambda: CatBoostRegressor(
            iterations=300, learning_rate=0.05, depth=4, loss_function="RMSE",
            random_seed=RANDOM_STATE, verbose=0, allow_writing_files=False, thread_count=1)
    except Exception as exc:
        unavailable["CatBoost"] = f"{type(exc).__name__}: {exc}"

    return factories, unavailable


def fit_model(factory, train_df):
    """Fit a fresh model on the engineered features -> RawMaterialInventory."""
    model = factory()
    model.fit(train_df[FEATURE_COLS], train_df[TARGET].astype(float).values)
    return model


def calculate_metrics(actual, predicted):
    """
    MAE, RMSE, MAPE (%), WMAPE (%).
    MAPE ignores observations with |actual| <= NEAR_ZERO_TOL (returns NaN if none);
    WMAPE = sum|error| / sum|actual| (returns NaN if the denominator is ~0).
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if a.size == 0 or a.size != p.size:
        raise ValueError("Actual and predicted arrays are empty or have different lengths.")
    if not np.all(np.isfinite(p)):
        raise ValueError("Predictions contain NaN or infinite values.")

    mae = float(mean_absolute_error(a, p))
    rmse = float(np.sqrt(mean_squared_error(a, p)))
    nonzero = np.abs(a) > NEAR_ZERO_TOL
    mape = float(np.mean(np.abs((a[nonzero] - p[nonzero]) / a[nonzero])) * 100) if nonzero.any() else np.nan
    denom = float(np.abs(a).sum())
    wmape = float(np.abs(a - p).sum() / denom * 100) if denom > NEAR_ZERO_TOL else np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "WMAPE": wmape}


def recursive_forecast_ml(model, history_values, forecast_dates):
    """
    Multi-step recursive forecast.

    For every future month:
        1. build RM_Lag1, RM_Lag2, RM_MA3, RM_STD3, RM_CV3 from the history so far
           (historical actuals + PREVIOUS PREDICTIONS) and the calendar features;
        2. predict the month;
        3. append the prediction to the history and move to the next month.
    No future actual RawMaterialInventory is ever used.
    """
    history = [float(v) for v in history_values]
    predictions = []
    for month_date in forecast_dates:
        x_row = build_feature_row(history, month_date)
        y_hat = float(model.predict(x_row)[0])
        y_hat = max(y_hat, 0.0)                                 # inventory cannot be negative
        predictions.append(y_hat)
        history.append(y_hat)
    return np.array(predictions)


# =============================================================================
# STAGE 1 - STEP 5: MODEL VALIDATION & SELECTION
# =============================================================================
def evaluate_models(factories, train_df, val_df, history_before_val, mode):
    """
    Validate every candidate model on the time-ordered validation set
    (the later part of the PRE-TEST data; the final test set is never touched).

    mode = "recursive" : validation months are forecast recursively from history_before_val
    mode = "one_step"  : validation months are predicted from their actual lag features
    """
    rows = []
    for name, factory in factories.items():
        row = {"Model": name, "Validation_MAE": np.nan, "Validation_RMSE": np.nan,
               "Validation_MAPE": np.nan, "Validation_WMAPE": np.nan,
               "Status": "OK", "Error": ""}
        try:
            model = fit_model(factory, train_df)
            if mode == "recursive":
                pred = recursive_forecast_ml(model, history_before_val, val_df["MonthDate"])
            else:
                pred = np.clip(np.asarray(model.predict(val_df[FEATURE_COLS]), dtype=float), 0, None)
            m = calculate_metrics(val_df[TARGET].values, pred)
            row.update({"Validation_MAE": m["MAE"], "Validation_RMSE": m["RMSE"],
                        "Validation_MAPE": m["MAPE"], "Validation_WMAPE": m["WMAPE"]})
        except Exception as exc:
            row["Status"] = "Failed"
            row["Error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        rows.append(row)
    return rows


def select_best_model(validation_rows):
    """Best model = lowest Validation MAE; Validation RMSE breaks ties. Returns None if all failed."""
    ok = [r for r in validation_rows
          if r["Status"] == "OK" and np.isfinite(r["Validation_MAE"])]
    if not ok:
        return None
    best = min(ok, key=lambda r: (r["Validation_MAE"],
                                  r["Validation_RMSE"] if np.isfinite(r["Validation_RMSE"]) else np.inf))
    return best["Model"]


def forecast_profitcenter(pc, pc_df, segment, factories, global_last_date,
                          val_fraction, validation_mode):
    """
    Complete Stage 1 workflow for ONE ProfitCenter.

    1. chronological dataset with leakage-free features
    2. drop rows whose lag/rolling features are unavailable
    3. final hold-out TEST = last TEST_HORIZON months (never used for selection)
    4. fixed 17-month design: 3 warm-up + 8 usable pre-test rows + 6 final test months
    5. time-ordered train / validation split inside the pre-test data
    6. validate XGBoost / LightGBM / CatBoost -> select by validation MAE (RMSE tie-break)
    7. retrain the selected model on ALL pre-test rows -> recursive forecast of the test months
    7. evaluate on the test set (MAE, RMSE, MAPE, WMAPE)
    8. retrain the selected model on ALL history -> recursive forecast of the next 6 months

    Never raises: problems are reported through result["Status"] / result["Reason"].
    """
    res = {
        "ProfitCenter": pc, "DataLength": len(pc_df), "DemandSegment": segment,
        "Status": "Forecasted", "Reason": "", "Notes": [],
        "SelectedModel": None, "ValidationRows": [],
        "TrainRows": 0, "ValRows": 0,
        "TestMetrics": None, "TestPredictions": None, "Forecast": None,
    }
    n = len(pc_df)

    if bool((pc_df["TotalActivity"] <= 0).all()):
        res.update(Status="Zero activity", Reason="Zero activity ProfitCenter is retained for reporting but excluded from ML forecasting.")
        return res

    # ---- fixed 17-month research design ----
    if n != REQUIRED_HISTORY_MONTHS:
        res.update(Status="Insufficient data", Reason=(
            f"{n} months available; exactly {REQUIRED_HISTORY_MONTHS} consecutive months are required "
            f"({FEATURE_WARMUP_ROWS} feature warm-up + {PRETEST_MODELLING_ROWS} pre-test modelling rows + {TEST_HORIZON} final test months)."))
        return res

    # ---- chronological split: 8 usable pre-test rows | final 6-month test ----
    test_start = n - TEST_HORIZON
    model_rows = pc_df.iloc[:test_start].dropna(subset=FEATURE_COLS)
    if len(model_rows) != PRETEST_MODELLING_ROWS:
        res.update(Status="Insufficient data", Reason=(
            f"Expected exactly {PRETEST_MODELLING_ROWS} usable pre-test modelling rows after the "
            f"{FEATURE_WARMUP_ROWS}-month feature warm-up, but found {len(model_rows)}."))
        return res

    n_val = max(MIN_VALIDATION_ROWS, int(math.ceil(len(model_rows) * val_fraction)))
    train_fit, val_df = model_rows.iloc[:-n_val], model_rows.iloc[-n_val:]
    if len(train_fit) < MIN_TRAIN_FIT_ROWS:
        res.update(Status="Insufficient data", Reason=(
            f"Validation split leaves {len(train_fit)} training rows (minimum {MIN_TRAIN_FIT_ROWS})."))
        return res
    res["TrainRows"], res["ValRows"] = len(train_fit), len(val_df)

    # ---- model validation & selection (pre-test data only) ----
    history_before_val = pc_df[TARGET].iloc[:test_start - n_val].values
    val_rows = evaluate_models(factories, train_fit, val_df, history_before_val, validation_mode)
    selected = select_best_model(val_rows)
    res["ValidationRows"] = val_rows
    if selected is None:
        errors = "; ".join(f"{r['Model']}: {r['Error']}" for r in val_rows if r["Error"])
        res.update(Status="Training failed", Reason=f"All candidate models failed validation. {errors}")
        return res
    res["SelectedModel"] = selected
    factory = factories[selected]

    # ---- final test: retrain on ALL pre-test data, forecast the hold-out months ----
    test_df = pc_df.iloc[test_start:]
    try:
        model_pre = fit_model(factory, model_rows)
        test_pred = recursive_forecast_ml(model_pre, pc_df[TARGET].iloc[:test_start].values, test_df["MonthDate"])
        res["TestMetrics"] = calculate_metrics(test_df[TARGET].values, test_pred)
        res["TestPredictions"] = pd.DataFrame({
            "ProfitCenter": pc,
            "Month": test_df["MonthDate"].dt.strftime(MONTH_LABEL_FORMAT).values,
            "MonthDate": test_df["MonthDate"].values,
            "Actual": test_df[TARGET].values,
            "Predicted": test_pred,
            "SelectedModel": selected,
        })
    except Exception as exc:
        res.update(Status="Training failed",
                   Reason=f"Test-period forecast failed for {selected}: {type(exc).__name__}: {exc}")
        return res

    # ---- production forecast: retrain on ALL history, recursive 6 months ahead ----
    try:
        model_all = fit_model(factory, pc_df.dropna(subset=FEATURE_COLS))
        last_date = pc_df["MonthDate"].max()
        gap = max(month_diff(global_last_date, last_date), 0)
        if gap > 0:
            res["Notes"].append(
                f"Series ends {gap} month(s) before the dataset's last month; the recursion "
                f"bridges the gap with its own predictions before the {FORECAST_HORIZON}-month forecast.")
        dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=gap + FORECAST_HORIZON, freq="MS")
        preds = recursive_forecast_ml(model_all, pc_df[TARGET].values, dates)[-FORECAST_HORIZON:]
        dates = dates[-FORECAST_HORIZON:]
        res["Forecast"] = pd.DataFrame({
            "ProfitCenter": pc,
            "Month": dates.strftime(MONTH_LABEL_FORMAT),
            "MonthDate": dates,
            "RM_Forecast": preds,
            "SelectedModel": selected,
            "DemandSegment": segment,
        })
    except Exception as exc:
        res.update(Status="Training failed",
                   Reason=f"Production forecast failed for {selected}: {type(exc).__name__}: {exc}")
    return res


@st.cache_data(show_spinner=False)
def run_stage1_pipeline(feature_df, segmented_df, global_last_date,
                        val_fraction, validation_mode):
    """
    Run Stage 1 for every ProfitCenter and assemble the result tables.
    (No Streamlit calls in here, so the cached result can be replayed safely.)
    """
    messages = []
    factories, unavailable = build_ml_models()
    out = {"available_models": list(factories), "unavailable_models": unavailable,
           "messages": messages, "validation_df": pd.DataFrame(), "test_metrics_df": pd.DataFrame(),
           "test_pred_df": pd.DataFrame(), "forecast_df": pd.DataFrame(),
           "coverage_df": pd.DataFrame(), "summary_df": pd.DataFrame()}

    if unavailable:
        messages.append(("warning",
            "Unavailable ML models: " + ", ".join(f"{k} ({v})" for k, v in unavailable.items()) +
            ". Install with:  pip install xgboost lightgbm catboost"))
    if not factories:
        messages.append(("error",
            "No ML library is available, so no forecast can be produced. "
            "Install at least one candidate model:  pip install xgboost lightgbm catboost"))
        return out

    segment_map = segmented_df.set_index("ProfitCenter")["DemandSegment"].to_dict()
    val_tables, test_tables, fc_tables, pred_tables, coverage, summary = [], [], [], [], [], []

    for pc, g in feature_df.groupby("ProfitCenter", sort=True):
        g = g.sort_values("MonthDate").reset_index(drop=True)
        segment = segment_map.get(pc, "Undetermined")
        try:
            if bool((g["TotalActivity"] <= 0).all()):
                res = {
                    "ProfitCenter": pc, "DataLength": len(g), "DemandSegment": "Zero Activity",
                    "Status": "Zero activity",
                    "Reason": "Zero activity ProfitCenter retained for data-quality reporting; excluded from ML.",
                    "Notes": [], "SelectedModel": None, "ValidationRows": [],
                    "TestMetrics": None, "TestPredictions": None, "Forecast": None,
                    "TrainRows": 0, "ValRows": 0
                }
            else:
                res = forecast_profitcenter(pc, g, segment, factories, global_last_date,
                                            val_fraction, validation_mode)
        except Exception as exc:                                # last-resort guard: never crash the app
            res = {"ProfitCenter": pc, "DataLength": len(g), "DemandSegment": segment,
                   "Status": "Training failed", "Reason": f"Unexpected error: {type(exc).__name__}: {exc}",
                   "Notes": [], "SelectedModel": None, "ValidationRows": [], "TestMetrics": None,
                   "TestPredictions": None, "Forecast": None, "TrainRows": 0, "ValRows": 0}

        for note in res["Notes"]:
            messages.append(("warning", f"[{pc}] {note}"))
        if res["Status"] != "Forecasted":
            messages.append(("warning", f"[{pc}] {res['Status']}: {res['Reason']}"))

        # --- validation comparison rows ---
        sel_val = {}
        for r in res["ValidationRows"]:
            is_selected = (r["Model"] == res["SelectedModel"])
            if is_selected:
                sel_val = r
            val_tables.append({
                "ProfitCenter": pc, "DemandSegment": segment, "Model": r["Model"],
                "Validation_MAE": r["Validation_MAE"], "Validation_RMSE": r["Validation_RMSE"],
                "Validation_MAPE": r["Validation_MAPE"], "Validation_WMAPE": r["Validation_WMAPE"],
                "SelectedModel": "Yes" if is_selected else "No",
                "TrainRows": res["TrainRows"], "ValRows": res["ValRows"],
                "Status": r["Status"], "Error": r["Error"],
            })

        tm = res["TestMetrics"]
        if tm is not None:
            test_tables.append({
                "ProfitCenter": pc, "SelectedModel": res["SelectedModel"],
                "Test_MAE": tm["MAE"], "Test_RMSE": tm["RMSE"],
                "Test_MAPE": tm["MAPE"], "Test_WMAPE": tm["WMAPE"],
            })
        if res["TestPredictions"] is not None:
            pred_tables.append(res["TestPredictions"])
        if res["Forecast"] is not None:
            fc_tables.append(res["Forecast"])

        coverage.append({"ProfitCenter": pc, "DataLength": res["DataLength"], "DemandSegment": segment,
                         "Status": res["Status"], "SelectedModel": res["SelectedModel"] or "-",
                         "Reason": res["Reason"]})
        summary.append({
            "ProfitCenter": pc, "DataLength": res["DataLength"], "DemandSegment": segment,
            "SelectedModel": res["SelectedModel"] or "-",
            "ValidationMAE": sel_val.get("Validation_MAE", np.nan),
            "ValidationRMSE": sel_val.get("Validation_RMSE", np.nan),
            "TestMAE": tm["MAE"] if tm else np.nan, "TestRMSE": tm["RMSE"] if tm else np.nan,
            "TestMAPE": tm["MAPE"] if tm else np.nan, "TestWMAPE": tm["WMAPE"] if tm else np.nan,
            "Status": res["Status"],
        })

    def _concat(frames):
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    out["validation_df"] = pd.DataFrame(val_tables)
    out["test_metrics_df"] = pd.DataFrame(test_tables)
    out["test_pred_df"] = _concat(pred_tables)
    out["forecast_df"] = _concat(fc_tables)
    out["coverage_df"] = pd.DataFrame(coverage)
    out["summary_df"] = pd.DataFrame(summary)
    return out


# =============================================================================
# STAGE 2: DYNAMIC WAREHOUSE PLANNING (logic preserved from the original app)
# =============================================================================
def calculate_operational_ratios(hist):
    """Historical operational ratios of one ProfitCenter (clipped to [0, RATIO_CLIP_MAX])."""
    s = hist[NUMERIC_COLS].sum()

    def ratio(numerator, denominator):
        return float(np.clip(numerator / max(denominator, 1), 0, RATIO_CLIP_MAX))

    return {
        "ReceivingRatio": ratio(s["ReceivingTransaction"], s["RawMaterialInventory"]),
        "ShippingRatio": ratio(s["ShippingTransaction"], s["ProductionRevenue"]),
        "TransferRatio": ratio(s["LocationTransferTransaction"], s["ReceivingTransaction"]),
        "FG_PalletRatio": ratio(s["FG_Pallet"], s["ProductionRevenue"]),
        "RM_PalletRatio": ratio(s["RM_Pallet"], s["RawMaterialInventory"]),
        "BinRatio": ratio(s["NoOfBin"], s["RawMaterialInventory"]),
        "ConsumptionRatio": ratio(s["RawMaterialInventory"], s["ProductionRevenue"]),
    }


def calculate_warehouse_cost(raw_material, inventory_next, warehouse_capacity, total_transaction):
    """Warehouse Cost Evaluation: holding + shortage + capacity + transaction cost."""
    holding = raw_material * HOLDING_COST_RATE
    shortage = max(raw_material - inventory_next, 0) * SHORTAGE_COST_RATE
    capacity = warehouse_capacity * CAPACITY_COST_RATE
    transaction = total_transaction * TRANSACTION_COST_RATE
    return {"HoldingCost": holding, "ShortageCost": shortage, "CapacityCost": capacity,
            "TransactionCost": transaction, "WarehouseCost": holding + shortage + capacity + transaction}


def calculate_warehouse_requirements(pc, hist, rm_forecast, months, selected_model, segment):
    """
    Stage 2 for ONE ProfitCenter.

    Input : Stage 1 forecast of RawMaterialInventory (Exogenous Forecast Input).
    Output: one row per future month with revenue, transactions, pallets, bins,
            inventory dynamics, capacity and cost. Returns (rows, ratios).
    """
    rm_forecast = np.clip(np.asarray(rm_forecast, dtype=float), 0, None)
    ratios = calculate_operational_ratios(hist)

    # Production revenue estimate: rolling 12-month revenue scaled by forecast / average RM
    window = min(REVENUE_WINDOW_MONTHS, len(hist))
    avg_rev = hist["ProductionRevenue"].iloc[-window:].mean()
    avg_rm = hist[TARGET].iloc[-window:].mean()
    if avg_rm > 0:
        rev_forecast = np.clip(avg_rev * (rm_forecast / avg_rm), 0, None)
    else:
        rev_forecast = np.full(len(rm_forecast), avg_rev)

    inventory = float(hist[TARGET].iloc[-1])                    # opening inventory = last actual
    rows = []
    for i, month in enumerate(months):
        production_revenue = max(rev_forecast[i], 0)
        raw_material = max(rm_forecast[i], 0)

        receiving = raw_material * ratios["ReceivingRatio"]
        shipping = production_revenue * ratios["ShippingRatio"]
        transfer = receiving * ratios["TransferRatio"]
        fg_pallet = production_revenue * ratios["FG_PalletRatio"]
        rm_pallet = raw_material * ratios["RM_PalletRatio"]
        no_of_bin = raw_material * ratios["BinRatio"]

        total_transaction = receiving + shipping + transfer
        no_of_pallet = fg_pallet + rm_pallet

        # Inventory dynamics
        rm_consumed = production_revenue * ratios["ConsumptionRatio"]
        inventory_next = max(inventory + receiving - rm_consumed, 0)

        # Capacity (contract flexibility) and cost
        warehouse_capacity = no_of_pallet * (1 + ALPHA - BETA)
        cost = calculate_warehouse_cost(raw_material, inventory_next, warehouse_capacity, total_transaction)

        rows.append({
            "Month": month,
            "ProfitCenter": pc,
            "ForecastModel": selected_model,
            "ProductionRevenue": round(production_revenue, 0),
            "RawMaterialInventory": round(raw_material, 0),
            "ReceivingTransaction": round(receiving, 0),
            "LocationTransferTransaction": round(transfer, 0),
            "ShippingTransaction": round(shipping, 0),
            "TotalTransaction": round(total_transaction, 0),
            "FG_Pallet": round(fg_pallet, 0),
            "RM_Pallet": round(rm_pallet, 0),
            "NoOfPallet": round(no_of_pallet, 0),
            "NoOfBin": round(no_of_bin, 0),
            "WarehouseCapacity": round(warehouse_capacity, 0),
            "WarehouseCost": round(cost["WarehouseCost"], 0),
            # additional Stage 2 detail (new columns, appended)
            "DemandSegment": segment,
            "RM_Consumed": round(rm_consumed, 0),
            "NextInventory": round(inventory_next, 0),
            "HoldingCost": round(cost["HoldingCost"], 2),
            "ShortageCost": round(cost["ShortageCost"], 2),
            "CapacityCost": round(cost["CapacityCost"], 2),
            "TransactionCost": round(cost["TransactionCost"], 2),
        })
        inventory = inventory_next
    return rows, ratios


def run_stage2(hist_df, stage1_forecast_df, segment_map, future_months):
    """Apply Stage 2 to every ProfitCenter that has a Stage 1 forecast."""
    all_rows, ratio_rows = [], []
    for pc, fc in stage1_forecast_df.groupby("ProfitCenter", sort=True):
        fc = fc.sort_values("MonthDate")
        hist = hist_df[hist_df["ProfitCenter"] == pc].sort_values("MonthDate")
        rows, ratios = calculate_warehouse_requirements(
            pc, hist, fc["RM_Forecast"].values, fc["Month"].tolist(),
            fc["SelectedModel"].iloc[0], segment_map.get(pc, "Undetermined"))
        all_rows.extend(rows)
        ratio_rows.append({"ProfitCenter": pc, **ratios})

    forecast_df = pd.DataFrame(all_rows)
    if forecast_df.empty:
        return forecast_df, pd.DataFrame(ratio_rows)
    forecast_df["MonthDate"] = pd.to_datetime(forecast_df["Month"], format=MONTH_LABEL_FORMAT, errors="coerce")
    forecast_df = forecast_df.sort_values(["MonthDate", "ProfitCenter"], kind="stable").reset_index(drop=True)
    return forecast_df, pd.DataFrame(ratio_rows)


def generate_planning_matrix(forecast_df, profit_centers):
    """EMS Forecast Planning Matrix: category x ProfitCenter x month."""
    categories = {
        "Production Revenue": "ProductionRevenue",
        "130000 - Raw Materials Inventory": "RawMaterialInventory",
        "Receiving transaction": "ReceivingTransaction",
        "Location transfer transaction": "LocationTransferTransaction",
        "Shipping transaction": "ShippingTransaction",
        "Total transaction": "TotalTransaction",
        "No. of Bin": "NoOfBin",
        "No. of pallet": "NoOfPallet",
        "FG Pallet": "FG_Pallet",
        "Raw Material Pallet": "RM_Pallet",
    }
    rows = []
    for category, col_name in categories.items():
        for pc in profit_centers:
            temp = forecast_df[forecast_df["ProfitCenter"] == pc]
            if temp.empty:
                continue
            row = {"Categories": category, "ProfitCenter": pc}
            row.update({m: round(v, 0) for m, v in zip(temp["Month"], temp[col_name])})
            rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# CHARTS
# =============================================================================
def create_forecast_chart(pc, segment, model_name, hist, test_pred, future):
    """Historical actual + final 6-month test prediction + future 6-month forecast."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["MonthDate"], y=hist[TARGET], mode="lines+markers",
        name="Historical Actual", line=dict(color=COLOR_ACTUAL, width=2)))

    if test_pred is not None and not test_pred.empty:
        fig.add_vrect(
            x0=pd.Timestamp(test_pred["MonthDate"].min()).strftime("%Y-%m-%d"),
            x1=pd.Timestamp(test_pred["MonthDate"].max()).strftime("%Y-%m-%d"),
            fillcolor="rgba(224,123,0,0.08)", line_width=0, layer="below")
        fig.add_trace(go.Scatter(
            x=test_pred["MonthDate"], y=test_pred["Predicted"], mode="lines+markers",
            name="Test Prediction (final 6-month hold-out)",
            line=dict(color=COLOR_TEST, width=2, dash="dash"), marker=dict(symbol="diamond")))

    if future is not None and not future.empty:
        fig.add_trace(go.Scatter(
            x=future["MonthDate"], y=future["RM_Forecast"], mode="lines+markers",
            name="Future Forecast (next 6 months)",
            line=dict(color=COLOR_FUTURE, width=2, dash="dot"), marker=dict(symbol="square")))

    fig.update_layout(
        title=f"ProfitCenter: {pc} | Demand Segment: {segment} | Selected Model: {model_name}",
        xaxis_title="Month", yaxis_title="RawMaterialInventory",
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="left", x=0),
        hovermode="x unified")
    fig.update_xaxes(tickformat="%b'%y")
    return fig


# =============================================================================
# UI RENDERING HELPERS
# =============================================================================
def render_methodology_flow():
    """Visual methodology flow shown at the top of the application."""
    stage1 = ["Historical EMS Data", "Data Preprocessing", "Feature Engineering", "Demand Pattern Analysis",
              "CV-Based Statistical Demand Segmentation", "Machine Learning Predictive Modeling",
              "Model Validation & Selection", "6-Month Raw Material Inventory Forecast"]
    stage2 = ["Inventory Dynamics", "Warehouse Operational Requirement Estimation",
              "Pallet / Bin Requirement", "Warehouse Capacity Requirement", "Contract Flexibility",
              "Warehouse Cost Evaluation", "Warehouse Planning Decision"]

    def chips(items, color):
        arrow = " <span style='color:#7a8aa0'>&rarr;</span> "
        return arrow.join(
            f"<span style='background:{color};color:#fff;padding:4px 10px;border-radius:14px;"
            f"display:inline-block;margin:3px 0;font-size:0.82rem'>{html.escape(t)}</span>"
            for t in items)

    st.markdown(
        "<div style='line-height:2.1'>"
        f"<b>STAGE 1 &mdash; Data Mining &amp; Predictive Forecasting</b><br>{chips(stage1, '#1F3A5F')}<br>"
        "<span style='color:#B7791F;font-size:1.3rem'>&darr;</span> "
        "<span style='background:#B7791F;color:#fff;padding:4px 12px;border-radius:14px;"
        "font-size:0.82rem'>EXOGENOUS FORECAST INPUT</span><br>"
        f"<b>STAGE 2 &mdash; Dynamic Warehouse Planning</b><br>{chips(stage2, '#0F7C8A')}"
        "</div>", unsafe_allow_html=True)


def render_methodology_notes():
    with st.expander("Methodology notes", expanded=False):
        st.markdown(
            "- Data mining is used to discover patterns and develop predictive models from historical EMS operational data.\n"
            "- CV-based statistical segmentation is used to characterize demand variability (statistical thresholds, not clustering).\n"
            "- Machine learning models (XGBoost, LightGBM, CatBoost) are then evaluated for Raw Material Inventory prediction. "
            "They are predictive modeling techniques *within* the broader data mining / predictive analytics stage.\n"
            "- The selected predictive model generates a 6-Month Raw Material Inventory Forecast.\n"
            "- The forecast is passed as an **Exogenous Forecast Input** into the Dynamic Warehouse Planning stage.\n"
            "- The warehouse planning stage converts forecasted inventory into operational, capacity, and cost requirements. "
            "It does not retrain or feed back into Stage 1, and it is a planning / evaluation model, not an optimization solver.")


def render_sidebar():
    st.sidebar.header("Stage 1 settings")
    mode_label = st.sidebar.selectbox("Validation protocol", list(VALIDATION_MODES), index=0)
    val_fraction = st.sidebar.slider(
        "Validation share of the 8-row pre-test period", 0.10, 0.40, DEFAULT_VALIDATION_FRACTION, 0.05,
        help="Chronological validation within the fixed 8 usable pre-test modelling rows. No shuffling.")
    st.sidebar.markdown("---")
    st.sidebar.header("Fixed research design")
    st.sidebar.write(f"Historical data required = {REQUIRED_HISTORY_MONTHS} months")
    st.sidebar.write(f"Feature warm-up = {FEATURE_WARMUP_ROWS} months")
    st.sidebar.write(f"Pre-test modelling period = {PRETEST_MODELLING_ROWS} usable rows")
    st.sidebar.write(f"Final hold-out test = {TEST_HORIZON} months")
    st.sidebar.write(f"Future forecast = {FORECAST_HORIZON} months")
    st.sidebar.markdown("---")
    st.sidebar.header("Stage 2 assumptions (fixed)")
    st.sidebar.write(f"alpha = {ALPHA:.2f}, beta = {BETA:.2f}")
    st.sidebar.write(f"Capacity multiplier = {1 + ALPHA - BETA:.2f}")
    return float(val_fraction), VALIDATION_MODES[mode_label]


def render_preprocessing(raw_df, clean_df, report):
    st.info("**Primary forecasting target: RawMaterialInventory.** It directly affects warehouse utilization "
            "and capacity requirements, so Stage 1 predicts it and Stage 2 translates it into operations.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows read", f"{report['rows_input']:,}")
    c2.metric("Rows after cleaning", f"{report['rows_output']:,}")
    c3.metric("Zero-activity rows retained", f"{report.get('zero_activity_rows', 0):,}")
    c4.metric("Invalid-date rows dropped", f"{report['rows_invalid_date']:,}")
    if report["missing_filled"]:
        st.caption("Missing / non-numeric values filled with 0 (cells per column): "
                   + ", ".join(f"{k}: {v}" for k, v in report["missing_filled"].items()))
    st.markdown("**Data quality per ProfitCenter**")
    st.caption(
        f"ML eligibility requires exactly {REQUIRED_HISTORY_MONTHS} consecutive months, "
        f"with {FEATURE_WARMUP_ROWS} warm-up months + {PRETEST_MODELLING_ROWS} pre-test modelling rows + {TEST_HORIZON} holdout months. "
        "Zero-activity ProfitCenters are retained here but excluded from ML.")
    show_df(report["quality_table"])
    with st.expander("Raw data check per ProfitCenter"):
        for pc in clean_df["ProfitCenter"].unique():
            st.markdown(f"**{pc}**")
            show_df(clean_df.loc[clean_df["ProfitCenter"] == pc, [
                "Month", "RawMaterialInventory", "ProductionRevenue", "ReceivingTransaction",
                "LocationTransferTransaction", "ShippingTransaction"]])


def render_feature_engineering(feature_df):
    st.markdown("**Leakage-free feature engineering** (computed per ProfitCenter)")
    st.markdown(
        "- `RM_Lag1`, `RM_Lag2`: previous 1 and 2 months of RawMaterialInventory\n"
        "- `RM_MA3`, `RM_STD3`: `shift(1).rolling(3)` mean / std - the **current month is excluded**\n"
        "- `RM_CV3 = RM_STD3 / RM_MA3`\n"
        "- `Year`, `MonthNumber`, `Quarter`, `MonthSin`, `MonthCos` (cyclical month encoding)\n"
        f"- First {FEATURE_WARMUP_ROWS} months of each ProfitCenter have no features (insufficient history) and are not used for modelling.")
    with st.expander("Feature tables per ProfitCenter"):
        for pc in feature_df["ProfitCenter"].unique():
            st.markdown(f"**ProfitCenter: {pc}**")
            show_df(round_df(feature_df[feature_df["ProfitCenter"] == pc]
                             .drop(columns=["MonthDate"]), 4))


def render_ml_modeling(stage1, val_fraction, validation_mode):
    st.markdown(
        "XGBoost, LightGBM and CatBoost are the candidate **predictive modeling techniques** of the data-mining stage. "
        "They *compete* per ProfitCenter on validation performance - the demand segment is **not** used to pick a model.")
    status_rows = [{"Model": m, "Status": "Available", "Detail": ""} for m in stage1["available_models"]]
    status_rows += [{"Model": m, "Status": "Not installed", "Detail": d}
                    for m, d in stage1["unavailable_models"].items()]
    show_df(pd.DataFrame(status_rows))

    st.markdown(
        f"**Predictors:** {', '.join(FEATURE_COLS)}.  \n"
        "Operational variables (receiving, shipping, transfers, revenue, pallets, bins) are *not* predictors because "
        "their future values are unknown at prediction time.")
    st.markdown(
        f"**Fixed history rule:** exactly **{REQUIRED_HISTORY_MONTHS} consecutive months** are required per ML-eligible ProfitCenter: "
        f"{FEATURE_WARMUP_ROWS} feature warm-up + {PRETEST_MODELLING_ROWS} usable pre-test modelling rows + {TEST_HORIZON} final test months. "
        "Shorter, longer, gapped, or zero-activity series are excluded from ML and clearly reported.  \n"
        f"**Validation:** later {val_fraction:.0%} of the {PRETEST_MODELLING_ROWS}-row pre-test modelling period (minimum {MIN_VALIDATION_ROWS}), "
        f"time-ordered, protocol = *{'recursive multi-step' if validation_mode == 'recursive' else 'one-step-ahead'}*.")
    st.markdown("**ProfitCenter coverage**")
    show_df(stage1["coverage_df"])


def render_validation_selection(stage1):
    st.markdown("**Model comparison on the validation set** (selection: lowest MAE, RMSE as tie-breaker). "
                "The final test set is not used here.")
    val_df = stage1["validation_df"]
    if val_df.empty:
        st.warning("No validation results are available.")
        return
    show_df(round_df(val_df))
    ok = val_df[val_df["Status"] == "OK"]
    if not ok.empty:
        fig = px.bar(ok, x="ProfitCenter", y="Validation_MAE", color="Model", barmode="group",
                     title="Validation MAE by model and ProfitCenter")
        show_chart(fig)

    st.markdown("**Final test performance** (last 6 months, selected model only)")
    if stage1["test_metrics_df"].empty:
        st.warning("No test results are available.")
    else:
        show_df(round_df(stage1["test_metrics_df"]))
    st.caption("MAPE ignores months with (near-)zero actuals; WMAPE = sum|error| / sum|actual| is the more stable measure.")

    st.markdown("**Model summary**")
    show_df(round_df(stage1["summary_df"]))


def render_rm_forecast(stage1, hist_df, future_months):
    fc = stage1["forecast_df"]
    pivot = (fc.pivot(index="ProfitCenter", columns="Month", values="RM_Forecast")
             .reindex(columns=future_months).round(0).reset_index())
    st.markdown("**6-Month Raw Material Inventory Forecast** (Stage 1 output)")
    show_df(pivot)

    test_pred = stage1["test_pred_df"]
    seg_model = fc.drop_duplicates("ProfitCenter").set_index("ProfitCenter")[["DemandSegment", "SelectedModel"]]
    for i, pc in enumerate(seg_model.index):
        hist = hist_df[hist_df["ProfitCenter"] == pc].sort_values("MonthDate")
        tp = test_pred[test_pred["ProfitCenter"] == pc] if not test_pred.empty else None
        future = fc[fc["ProfitCenter"] == pc].sort_values("MonthDate")
        with st.expander(f"{pc}  |  {seg_model.loc[pc, 'DemandSegment']}  |  {seg_model.loc[pc, 'SelectedModel']}",
                         expanded=(i < 3)):
            show_chart(create_forecast_chart(pc, seg_model.loc[pc, "DemandSegment"],
                                             seg_model.loc[pc, "SelectedModel"], hist, tp, future))

    fig = px.line(fc, x="Month", y="RM_Forecast", color="ProfitCenter", markers=True,
                  title="Raw Material Inventory Forecast", category_orders={"Month": future_months})
    show_chart(fig)


def render_capacity(forecast_df):
    c1, c2, c3 = st.columns(3)
    c1.metric("alpha - capacity flexibility / buffer", f"{ALPHA:.2f}")
    c2.metric("beta - capacity reduction / adjustment factor", f"{BETA:.2f}")
    c3.metric("Capacity multiplier (1 + alpha - beta)", f"{1 + ALPHA - BETA:.2f}")
    st.code(
        "NoOfBin = RawMaterialInventory_forecast * BinRatio\n"
        "warehouse_capacity = no_of_pallet * (1 + alpha - beta)"
    )
    cap = (forecast_df.groupby("ProfitCenter")
           .agg(Avg_NoOfPallet=("NoOfPallet", "mean"),
                Avg_WarehouseCapacity=("WarehouseCapacity", "mean"),
                Peak_WarehouseCapacity=("WarehouseCapacity", "max"))
           .reset_index())
    show_df(round_df(cap, 0))
    fig = px.line(forecast_df, x="Month", y="WarehouseCapacity", color="ProfitCenter", markers=True,
                  title="Warehouse Capacity Requirement (pallets)",
                  category_orders={"Month": forecast_df["Month"].drop_duplicates().tolist()})
    show_chart(fig)


def render_cost(forecast_df):
    st.subheader("KPI Dashboard")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Warehouse Cost", f"{forecast_df['WarehouseCost'].sum():,.0f}")
    c2.metric("Average Warehouse Capacity", f"{forecast_df['WarehouseCapacity'].mean():,.0f}")
    c3.metric("Average Transaction", f"{forecast_df['TotalTransaction'].mean():,.0f}")

    st.caption(
        f"Cost assumptions (unchanged): holding = {HOLDING_COST_RATE} x RM inventory; "
        f"shortage = {SHORTAGE_COST_RATE} x max(RM - next inventory, 0); "
        f"capacity = {CAPACITY_COST_RATE} x warehouse capacity; transaction = {TRANSACTION_COST_RATE} x total transactions.")
    components = ["HoldingCost", "ShortageCost", "CapacityCost", "TransactionCost"]
    cost = forecast_df.groupby("ProfitCenter")[components + ["WarehouseCost"]].sum().reset_index()
    show_df(round_df(cost, 0))
    long = cost.melt(id_vars="ProfitCenter", value_vars=components, var_name="Component", value_name="Cost")
    show_chart(px.bar(long, x="ProfitCenter", y="Cost", color="Component", barmode="stack",
                      title="Warehouse Cost Evaluation (6-month total by component)"))


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def run_app():
    val_fraction, validation_mode = render_sidebar()

    # ---- 1. Upload -------------------------------------------------------------
    st.header("1. Upload Dataset")
    uploaded_file = st.file_uploader("Upload EMS Forecasting Dataset", type=["xlsx"])
    if uploaded_file is None:
        st.info(
            f"Upload an .xlsx file with the required columns and exactly {REQUIRED_HISTORY_MONTHS} consecutive historical months per valid ProfitCenter."
        )
        return
    try:
        raw_df = pd.read_excel(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the Excel file: {exc}")
        return
    st.subheader("Raw Dataset")
    show_df(raw_df)

    # ---- 2. Data preprocessing ---------------------------------------------------
    st.header("2. Data Preprocessing")
    try:
        clean_df, report = preprocess_data(raw_df)
    except DataError as exc:
        st.error(str(exc))
        return
    show_messages(report["messages"])
    render_preprocessing(raw_df, clean_df, report)

    # ---- 3. Data mining & demand pattern analysis ----------------------------------
    st.header("3. Data Mining & Demand Pattern Analysis")
    feature_df = create_features(clean_df)
    render_feature_engineering(feature_df)
    pattern_df = analyze_demand_patterns(clean_df)
    st.markdown("**Demand pattern analysis** (RawMaterialInventory per ProfitCenter)")
    show_df(round_df(pattern_df))

    # ---- 4. CV-based statistical demand segmentation ---------------------------------
    st.header("4. CV-Based Statistical Demand Segmentation")
    st.markdown(
        "The Coefficient of Variation (CV = std / mean) characterizes demand / inventory variability **before** "
        f"predictive modeling. Thresholds: **Stable** CV < {CV_STABLE_MAX:.2f}; **Moderate** "
        f"{CV_STABLE_MAX:.2f} <= CV < {CV_VOLATILE_MIN:.2f}; **Volatile** CV >= {CV_VOLATILE_MIN:.2f}. "
        "This is a statistical rule, not K-means or any other clustering algorithm.")
    segmented_df = segment_customers(pattern_df)
    st.subheader("Customer Demand Segmentation")
    show_df(round_df(segmented_df))
    segment_map = segmented_df.set_index("ProfitCenter")["DemandSegment"].to_dict()

    st.success(
        f"CV-based statistical demand segmentation completed successfully for "
        f"{segmented_df['ProfitCenter'].nunique()} ProfitCenter(s)."
    )

    # ---- 5. ML predictive modeling ---------------------------------------------------
    # IMPORTANT: Stage 1 ML is intentionally executed INSIDE Step 5.
    # This prevents ML dependency/training errors from appearing to be Step 4 errors.
    st.header("5. Machine Learning Predictive Modeling")
    st.info(
        "Stage 5 uses XGBoost, LightGBM and CatBoost to forecast "
        "RawMaterialInventory. The demand segment from Step 4 describes "
        "variability only and does not determine the selected ML model."
    )

    global_last_date = clean_df["MonthDate"].max()
    future_months = [(global_last_date + pd.DateOffset(months=i + 1)).strftime(MONTH_LABEL_FORMAT)
                     for i in range(FORECAST_HORIZON)]

    with st.spinner("Step 5: checking ML libraries, training candidate models, validating and forecasting..."):
        stage1 = run_stage1_pipeline(
            feature_df,
            segmented_df,
            global_last_date,
            val_fraction,
            validation_mode,
        )

    show_messages(stage1["messages"])
    render_ml_modeling(stage1, val_fraction, validation_mode)

    if stage1["forecast_df"].empty:
        st.error("No forecast generated. See the ProfitCenter coverage table above for the reasons "
                 "(missing ML libraries, insufficient history or training errors).")
        return

    # ---- 6. Validation & selection ---------------------------------------------------
    st.header("6. Model Validation & Selection")
    render_validation_selection(stage1)

    # ---- 7. 6-month forecast --------------------------------------------------------
    st.header("7. 6-Month Raw Material Inventory Forecast")
    render_rm_forecast(stage1, clean_df, future_months)

    # ---- 8. Stage 2 ------------------------------------------------------------------
    st.header("8. Stage 2 - Dynamic Warehouse Planning")
    st.markdown(
        "The Stage 1 forecast enters Stage 2 as an **Exogenous Forecast Input**. Stage 2 does not retrain any model: "
        "it converts forecasted RawMaterialInventory into receiving, shipping, transfer, pallet and bin requirements "
        "using each ProfitCenter's historical operational ratios, then applies inventory dynamics, capacity and cost.")
    forecast_df, ratios_df = run_stage2(clean_df, stage1["forecast_df"], segment_map, future_months)
    if forecast_df.empty:
        st.error("Stage 2 produced no output.")
        return
    st.markdown("**Historical operational ratios used by Stage 2**")
    show_df(round_df(ratios_df, 4))
    st.subheader("Forecast Dataset")
    show_df(forecast_df.drop(columns=["MonthDate"]))

    # ---- 9. Capacity -------------------------------------------------------------------
    st.header("9. Warehouse Capacity Evaluation")
    render_capacity(forecast_df)

    # ---- 10. Cost ----------------------------------------------------------------------
    st.header("10. Warehouse Cost Evaluation")
    render_cost(forecast_df)

    # ---- 11. Planning matrix -----------------------------------------------------------
    st.header("11. Planning Matrix")
    st.subheader("EMS Forecast Planning Matrix")
    matrix_df = generate_planning_matrix(forecast_df, forecast_df["ProfitCenter"].unique())
    show_df(matrix_df)

    # ---- 12. Downloads -------------------------------------------------------------------
    st.header("12. Downloads")
    model_map = stage1["summary_df"].set_index("ProfitCenter")["SelectedModel"]
    matrix_with_model = matrix_df.copy()
    matrix_with_model.insert(2, "ForecastModel", matrix_with_model["ProfitCenter"].map(model_map))

    downloads = [
        ("Download Forecast Matrix CSV", matrix_with_model, "EMS_Forecast_Matrix.csv"),
        ("Download Full Forecast Dataset CSV", forecast_df.drop(columns=["MonthDate"]), "EMS_Forecast_Full.csv"),
        ("Download Demand Segmentation CSV", segmented_df, "EMS_Demand_Segmentation.csv"),
        ("Download Validation Results CSV", stage1["validation_df"], "EMS_Model_Validation.csv"),
        ("Download Test Performance CSV", stage1["test_metrics_df"], "EMS_Test_Performance.csv"),
        ("Download Model Summary CSV", stage1["summary_df"], "EMS_Model_Summary.csv"),
        ("Download Stage 1 RM Forecast CSV", stage1["forecast_df"].drop(columns=["MonthDate"]), "EMS_RM_Forecast_Stage1.csv"),
        ("Download Test Predictions CSV", stage1["test_pred_df"].drop(columns=["MonthDate"], errors="ignore"), "EMS_Test_Predictions.csv"),
    ]
    cols = st.columns(2)
    for i, (label, frame, fname) in enumerate(downloads):
        with cols[i % 2]:
            st.download_button(label=label, data=to_csv_bytes(frame), file_name=fname,
                               mime="text/csv", key=f"dl_{i}")


def main():
    st.set_page_config(page_title="HCMIC EMS Planning System", layout="wide")
    st.title(APP_TITLE)
    st.write("Enterprise EMS Predictive Forecasting & Dynamic Warehouse Planning")
    render_methodology_flow()
    render_methodology_notes()
    run_app()
    st.markdown("---")
    st.caption(APP_TITLE)


if __name__ == "__main__":
    main()
