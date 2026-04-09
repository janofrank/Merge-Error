"""
Reefer Peak Load Forecasting – Baseline Model
===============================================
LightGBM with Optuna hyperparameter optimization.
Two models: point forecast (MAE) + P90 quantile regression.
Weather data excluded.

Outputs:
  - predictions.csv  (submission file)
  - validation plots  (actual vs predicted)
  - feature importance table
"""

from __future__ import annotations

import warnings
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(r"c:\Users\asmus\Documents\Hackathon\participant_package\participant_package")
REEFER_CSV = BASE_DIR / "reefer_release" / "reefer_release.csv"
TARGETS_CSV = BASE_DIR / "target_timestamps.csv"
OUTPUT_DIR = Path(r"c:\Users\asmus\Documents\Hackathon\output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Data Loading & Hourly Aggregation
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1: Loading reefer data ...")
print("=" * 70)

# Read the large CSV – semicolon-delimited, comma decimals
df_raw = pd.read_csv(
    REEFER_CSV,
    sep=";",
    decimal=",",
    usecols=["EventTime", "AvPowerCons", "TemperatureSetPoint",
             "TemperatureAmbient", "TemperatureReturn", "RemperatureSupply",
             "ContainerSize", "container_visit_uuid"],
    dtype={
        "AvPowerCons": "float64",
        "TemperatureSetPoint": "float64",
        "TemperatureAmbient": "float64",
        "TemperatureReturn": "float64",
        "RemperatureSupply": "float64",
        "ContainerSize": "str",
        "container_visit_uuid": "str",
    },
    parse_dates=["EventTime"],
    low_memory=False,
)
print(f"  Raw rows loaded: {len(df_raw):,}")

# Floor timestamps to hour
df_raw["hour"] = df_raw["EventTime"].dt.floor("h")

# Convert power: Watts -> kW per container-hour
df_raw["power_kw"] = df_raw["AvPowerCons"] / 1000.0

# Aggregate per hour
hourly_agg = df_raw.groupby("hour").agg(
    total_power_kw=("power_kw", "sum"),
    container_count=("container_visit_uuid", "nunique"),
    mean_setpoint=("TemperatureSetPoint", "mean"),
    mean_ambient=("TemperatureAmbient", "mean"),
    mean_return_temp=("TemperatureReturn", "mean"),
    mean_supply_temp=("RemperatureSupply", "mean"),
    std_power=("power_kw", "std"),
    max_power=("power_kw", "max"),
    min_power=("power_kw", "min"),
).reset_index()

# Ensure continuous hourly index
full_range = pd.date_range(
    start=hourly_agg["hour"].min(),
    end=hourly_agg["hour"].max(),
    freq="h",
)
hourly = pd.DataFrame({"hour": full_range}).merge(hourly_agg, on="hour", how="left")
hourly = hourly.sort_values("hour").reset_index(drop=True)

# Forward-fill short gaps (up to 3 hours)
hourly["total_power_kw"] = hourly["total_power_kw"].interpolate(method="linear", limit=3)
for col in ["container_count", "mean_setpoint", "mean_ambient",
            "mean_return_temp", "mean_supply_temp", "std_power", "max_power", "min_power"]:
    hourly[col] = hourly[col].interpolate(method="linear", limit=3)

print(f"  Hourly time series: {len(hourly):,} hours")
print(f"  Date range: {hourly['hour'].min()} -> {hourly['hour'].max()}")

# Free raw memory
del df_raw

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Feature engineering ...")
print("=" * 70)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all features from the hourly aggregated DataFrame."""
    df = df.copy()

    # ── Calendar features ──
    df["hour_of_day"] = df["hour"].dt.hour
    df["day_of_week"] = df["hour"].dt.dayofweek          # 0=Mon
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["day_of_month"] = df["hour"].dt.day
    df["week_of_year"] = df["hour"].dt.isocalendar().week.astype(int)
    df["month"] = df["hour"].dt.month

    # ── Cyclical encoding for hour and day-of-week ──
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # ── Lag features (total_power_kw) ──
    lag_hours = [24, 48, 72, 168]
    for lag in lag_hours:
        df[f"lag_{lag}h"] = df["total_power_kw"].shift(lag)

    # ── Lag features (container count) ──
    for lag in [24, 48, 168]:
        df[f"container_count_lag_{lag}h"] = df["container_count"].shift(lag)

    # ── Lag features (ambient temperature) ──
    for lag in [24, 48]:
        df[f"ambient_temp_lag_{lag}h"] = df["mean_ambient"].shift(lag)

    # ── Rolling statistics (on total_power_kw) ──
    for window in [24, 48, 168]:
        rolled = df["total_power_kw"].shift(24).rolling(window, min_periods=1)
        df[f"roll_mean_{window}h"] = rolled.mean()
        df[f"roll_std_{window}h"] = rolled.std()
        df[f"roll_max_{window}h"] = rolled.max()
        df[f"roll_min_{window}h"] = rolled.min()

    # ── Rolling statistics (container count) ──
    for window in [24, 168]:
        rolled = df["container_count"].shift(1).rolling(window, min_periods=1)
        df[f"container_roll_mean_{window}h"] = rolled.mean()

    # ── Diff features ──
    df["diff_24h"] = df["total_power_kw"].shift(1) - df["total_power_kw"].shift(25)
    df["ratio_to_24h_ago"] = df["total_power_kw"].shift(1) / df["total_power_kw"].shift(25).replace(0, np.nan)

    # ── Same-hour-yesterday and same-hour-last-week ──
    df["same_hour_yesterday"] = df["total_power_kw"].shift(24)
    df["same_hour_last_week"] = df["total_power_kw"].shift(168)

    # ── Interaction features ──
    df["container_x_ambient"] = df["container_count"].shift(24) * df["mean_ambient"].shift(24)

    return df


hourly = build_features(hourly)

# Define feature columns (everything except target and hour)
target_col = "total_power_kw"
exclude_cols = {
    "hour", "total_power_kw", 
    "container_count", "mean_setpoint", "mean_ambient", 
    "mean_return_temp", "mean_supply_temp", 
    "std_power", "max_power", "min_power"
}
feature_cols = [c for c in hourly.columns if c not in exclude_cols and hourly[c].dtype in ["float64", "int64", "int32", "uint32"]]

print(f"  Total features: {len(feature_cols)}")
print(f"  Feature list: {feature_cols[:10]} ... (truncated)")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Train / Validation Split
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: Train/validation split ...")
print("=" * 70)

# Drop rows with NaN target or too many NaN features
hourly_clean = hourly.dropna(subset=[target_col]).copy()

# Validation: last 10 days of available data (before target period)
val_start = pd.Timestamp("2025-12-22")
val_end = pd.Timestamp("2025-12-31 23:00:00")

train_mask = hourly_clean["hour"] < val_start
val_mask = (hourly_clean["hour"] >= val_start) & (hourly_clean["hour"] <= val_end)

df_train = hourly_clean[train_mask].copy()
df_val = hourly_clean[val_mask].copy()

# Drop rows where features are mostly NaN (beginning of time series)
min_features_needed = len(feature_cols) // 2
df_train = df_train.dropna(subset=feature_cols, thresh=min_features_needed)
df_val = df_val.dropna(subset=feature_cols, thresh=min_features_needed)

X_train = df_train[feature_cols]
y_train = df_train[target_col]
X_val = df_val[feature_cols]
y_val = df_val[target_col]

print(f"  Training set:   {len(X_train):,} hours  ({df_train['hour'].min()} -> {df_train['hour'].max()})")
print(f"  Validation set: {len(X_val):,} hours  ({df_val['hour'].min()} -> {df_val['hour'].max()})")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Optuna Hyperparameter Optimization
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: Optuna hyperparameter search (this may take a few minutes) ...")
print("=" * 70)

N_TRIALS = 80


def objective_point(trial: optuna.Trial) -> float:
    """Objective for point forecast (MAE)."""
    params = {
        "objective": "mae",
        "metric": "mae",
        "verbosity": -1,
        "n_jobs": -1,
        "seed": 42,
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 256),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    preds = model.predict(X_val)
    mae = np.mean(np.abs(y_val.values - preds))
    return mae


study_point = optuna.create_study(direction="minimize", study_name="point_forecast")
study_point.optimize(objective_point, n_trials=N_TRIALS, show_progress_bar=True)

best_params_point = study_point.best_params
print(f"\n  Best MAE on validation: {study_point.best_value:.2f} kW")
print(f"  Best params: {best_params_point}")


def objective_quantile(trial: optuna.Trial) -> float:
    """Objective for P90 quantile forecast (pinball loss)."""
    params = {
        "objective": "quantile",
        "alpha": 0.9,
        "metric": "quantile",
        "verbosity": -1,
        "n_jobs": -1,
        "seed": 42,
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 256),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    preds = model.predict(X_val)
    # Pinball loss for quantile 0.9
    errors = y_val.values - preds
    pinball = np.mean(np.where(errors >= 0, 0.9 * errors, -0.1 * errors))
    return pinball


study_quantile = optuna.create_study(direction="minimize", study_name="quantile_p90")
study_quantile.optimize(objective_quantile, n_trials=N_TRIALS, show_progress_bar=True)

best_params_quantile = study_quantile.best_params
print(f"\n  Best Pinball (P90) on validation: {study_quantile.best_value:.2f}")
print(f"  Best params: {best_params_quantile}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: Train Final Models on Full Training Data
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: Training final models with best params ...")
print("=" * 70)

# For final models, train on ALL data up to end of December 2025
final_train_mask = hourly_clean["hour"] <= val_end
df_final_train = hourly_clean[final_train_mask].copy()
df_final_train = df_final_train.dropna(subset=feature_cols, thresh=min_features_needed)

X_final = df_final_train[feature_cols]
y_final = df_final_train[target_col]

# Point model
final_point_params = {
    "objective": "mae",
    "metric": "mae",
    "verbosity": -1,
    "n_jobs": -1,
    "seed": 42,
    **best_params_point,
}
model_point = lgb.LGBMRegressor(**final_point_params)
model_point.fit(X_final, y_final)
print(f"  Point model trained on {len(X_final):,} samples")

# Quantile model (P90)
final_q90_params = {
    "objective": "quantile",
    "alpha": 0.9,
    "metric": "quantile",
    "verbosity": -1,
    "n_jobs": -1,
    "seed": 42,
    **best_params_quantile,
}
model_p90 = lgb.LGBMRegressor(**final_q90_params)
model_p90.fit(X_final, y_final)
print(f"  P90 model trained on {len(X_final):,} samples")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: Generate Predictions for Target Timestamps
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 6: Generating predictions for target timestamps ...")
print("=" * 70)

# Load target timestamps
targets = pd.read_csv(TARGETS_CSV)
targets["timestamp_utc_parsed"] = pd.to_datetime(targets["timestamp_utc"]).dt.tz_localize(None)

# We need features for these target hours.
# Extend our hourly DataFrame to cover the target period.
# The target timestamps are Jan 1-10, 2026 – we need to build features
# using lag data from the historical period.

# First, extend the hourly index to cover target period
target_max = targets["timestamp_utc_parsed"].max()
extended_range = pd.date_range(
    start=hourly["hour"].min(),
    end=target_max,
    freq="h",
)
hourly_extended = pd.DataFrame({"hour": extended_range}).merge(
    hourly[["hour", "total_power_kw", "container_count", "mean_setpoint",
            "mean_ambient", "mean_return_temp", "mean_supply_temp",
            "std_power", "max_power", "min_power"]],
    on="hour", how="left"
)
hourly_extended = hourly_extended.sort_values("hour").reset_index(drop=True)

# Rebuild features on the extended set
hourly_extended = build_features(hourly_extended)

# Extract rows for target timestamps
target_set = set(targets["timestamp_utc_parsed"])
target_rows = hourly_extended[hourly_extended["hour"].isin(target_set)].copy()
missing_targets = target_set - set(target_rows["hour"])
if missing_targets:
    print(f"  WARNING: {len(missing_targets)} target timestamps not found in extended data!")

X_target = target_rows[feature_cols]

# Predict
pred_power = model_point.predict(X_target)
pred_p90 = model_p90.predict(X_target)

# Enforce constraint: pred_p90 >= pred_power
pred_p90 = np.maximum(pred_p90, pred_power)

# Ensure non-negative
pred_power = np.maximum(pred_power, 0)
pred_p90 = np.maximum(pred_p90, 0)

target_rows = target_rows.copy()
target_rows["pred_power_kw"] = pred_power
target_rows["pred_p90_kw"] = pred_p90

print(f"  Predictions generated for {len(target_rows)} timestamps")
print(f"  pred_power_kw  — mean: {pred_power.mean():.1f}, min: {pred_power.min():.1f}, max: {pred_power.max():.1f}")
print(f"  pred_p90_kw    — mean: {pred_p90.mean():.1f}, min: {pred_p90.min():.1f}, max: {pred_p90.max():.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7: Validation Plots & Feature Importance
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 7: Validation analysis & plots ...")
print("=" * 70)

# -- 7a: Validation predictions --
val_pred_point = model_point.predict(X_val)
val_pred_p90 = model_p90.predict(X_val)
val_pred_p90 = np.maximum(val_pred_p90, val_pred_point)

val_actuals = y_val.values
val_hours = df_val["hour"].values

# Metrics
mae_val = np.mean(np.abs(val_actuals - val_pred_point))
mape_val = np.mean(np.abs((val_actuals - val_pred_point) / np.maximum(val_actuals, 1))) * 100
errors_q = val_actuals - val_pred_p90
pinball_val = np.mean(np.where(errors_q >= 0, 0.9 * errors_q, -0.1 * errors_q))

# Identify peak hours (top 10% of actual load in validation)
peak_threshold = np.percentile(val_actuals, 90)
peak_mask = val_actuals >= peak_threshold
mae_peak = np.mean(np.abs(val_actuals[peak_mask] - val_pred_point[peak_mask])) if peak_mask.sum() > 0 else np.nan

# Combined score (as per evaluation doc)
combined_score = 0.5 * mae_val + 0.3 * mae_peak + 0.2 * pinball_val

print(f"\n  VALIDATION METRICS:")
print(f"  {'─' * 40}")
print(f"  MAE (all hours):    {mae_val:>10.2f} kW")
print(f"  MAE (peak hours):   {mae_peak:>10.2f} kW")
print(f"  MAPE:               {mape_val:>10.2f} %")
print(f"  Pinball (P90):      {pinball_val:>10.2f}")
print(f"  Combined Score:     {combined_score:>10.2f}")
print(f"  Peak threshold:     {peak_threshold:>10.2f} kW (90th percentile)")
print(f"  Peak hours count:   {peak_mask.sum()}")

# -- 7b: Actual vs Predicted time series plot --
fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=False)

# Plot 1: Full validation period – actual vs predicted
ax1 = axes[0]
ax1.plot(val_hours, val_actuals, label="Actual", color="#2563eb", linewidth=1.5, alpha=0.9)
ax1.plot(val_hours, val_pred_point, label="Predicted (point)", color="#dc2626", linewidth=1.2, alpha=0.8, linestyle="--")
ax1.fill_between(val_hours, val_pred_point, val_pred_p90, alpha=0.2, color="#f97316", label="P90 band")
ax1.axhline(y=peak_threshold, color="#16a34a", linestyle=":", alpha=0.6, label=f"Peak threshold ({peak_threshold:.0f} kW)")
ax1.set_ylabel("Total Reefer Load (kW)")
ax1.set_title("Validation Period: Actual vs Predicted (Dec 22–31, 2025)", fontsize=14, fontweight="bold")
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_major_locator(mdates.DayLocator())
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

# Plot 2: Scatter – predicted vs actual
ax2 = axes[1]
ax2.scatter(val_actuals, val_pred_point, alpha=0.5, s=20, c="#2563eb", edgecolors="none")
lims = [
    min(val_actuals.min(), val_pred_point.min()) * 0.95,
    max(val_actuals.max(), val_pred_point.max()) * 1.05,
]
ax2.plot(lims, lims, "k--", alpha=0.5, linewidth=1, label="Perfect forecast")
ax2.set_xlabel("Actual (kW)")
ax2.set_ylabel("Predicted (kW)")
ax2.set_title("Scatter: Actual vs Predicted", fontsize=14, fontweight="bold")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(lims)
ax2.set_ylim(lims)

# Plot 3: Error distribution
ax3 = axes[2]
errors = val_actuals - val_pred_point
ax3.hist(errors, bins=50, color="#7c3aed", alpha=0.7, edgecolor="white")
ax3.axvline(x=0, color="black", linestyle="--", linewidth=1)
ax3.axvline(x=np.mean(errors), color="#dc2626", linestyle="-", linewidth=1.5, label=f"Mean error: {np.mean(errors):.1f} kW")
ax3.set_xlabel("Error (Actual – Predicted) kW")
ax3.set_ylabel("Count")
ax3.set_title("Error Distribution", fontsize=14, fontweight="bold")
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "validation_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: {OUTPUT_DIR / 'validation_plots.png'}")

# -- 7c: Feature Importance (plot + table) --
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance_point": model_point.feature_importances_,
    "importance_p90": model_p90.feature_importances_,
}).sort_values("importance_point", ascending=False).reset_index(drop=True)
importance_df["rank"] = range(1, len(importance_df) + 1)

# Plot top 30
top_n = min(30, len(importance_df))
fig, axes = plt.subplots(1, 2, figsize=(16, 10))

ax = axes[0]
top = importance_df.head(top_n).sort_values("importance_point")
ax.barh(range(top_n), top["importance_point"].values, color="#2563eb", alpha=0.8)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top["feature"].values, fontsize=8)
ax.set_xlabel("Importance (split count)")
ax.set_title(f"Top {top_n} Features – Point Model", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, axis="x")

ax = axes[1]
top_q = importance_df.sort_values("importance_p90", ascending=False).head(top_n).sort_values("importance_p90")
ax.barh(range(top_n), top_q["importance_p90"].values, color="#f97316", alpha=0.8)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_q["feature"].values, fontsize=8)
ax.set_xlabel("Importance (split count)")
ax.set_title(f"Top {top_n} Features – P90 Model", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR / 'feature_importance.png'}")

# Print tabular feature importance
print(f"\n  FEATURE IMPORTANCE TABLE (Top 30 – Point Model):")
print(f"  {'─' * 60}")
print(f"  {'Rank':<6} {'Feature':<35} {'Importance':>12}")
print(f"  {'─' * 60}")
for _, row in importance_df.head(30).iterrows():
    print(f"  {int(row['rank']):<6} {row['feature']:<35} {int(row['importance_point']):>12,}")
print(f"  {'─' * 60}")

print(f"\n  FEATURE IMPORTANCE TABLE (Top 30 – P90 Model):")
print(f"  {'─' * 60}")
print(f"  {'Rank':<6} {'Feature':<35} {'Importance':>12}")
print(f"  {'─' * 60}")
top_p90 = importance_df.sort_values("importance_p90", ascending=False).head(30).reset_index(drop=True)
for i, row in top_p90.iterrows():
    print(f"  {i+1:<6} {row['feature']:<35} {int(row['importance_p90']):>12,}")
print(f"  {'─' * 60}")

# -- 7d: Worst predictions table --
worst_df = df_val[["hour"]].copy()
worst_df["actual_kw"] = val_actuals
worst_df["predicted_kw"] = val_pred_point
worst_df["error_kw"] = val_actuals - val_pred_point
worst_df["abs_error_kw"] = np.abs(worst_df["error_kw"])
worst_df = worst_df.sort_values("abs_error_kw", ascending=False).head(20)

print(f"\n  WORST 20 PREDICTIONS (by absolute error):")
print(f"  {'─' * 75}")
print(f"  {'Hour':<22} {'Actual':>12} {'Predicted':>12} {'Error':>12} {'AbsErr':>12}")
print(f"  {'─' * 75}")
for _, row in worst_df.iterrows():
    print(f"  {str(row['hour']):<22} {row['actual_kw']:>12.1f} {row['predicted_kw']:>12.1f} {row['error_kw']:>12.1f} {row['abs_error_kw']:>12.1f}")
print(f"  {'─' * 75}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8: Write Submission File
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 8: Writing predictions.csv ...")
print("=" * 70)

submission = target_rows[["hour", "pred_power_kw", "pred_p90_kw"]].copy()
submission["timestamp_utc"] = submission["hour"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
submission = submission[["timestamp_utc", "pred_power_kw", "pred_p90_kw"]]

# Merge with target list to ensure exact order and completeness
targets_sorted = targets[["timestamp_utc"]].copy()
submission = targets_sorted.merge(submission, on="timestamp_utc", how="left")

# Check for missing predictions
missing = submission["pred_power_kw"].isna().sum()
if missing > 0:
    print(f"  WARNING: {missing} timestamps have no prediction – filling with naive fallback")
    # Fallback: use last available lag if prediction is missing
    submission["pred_power_kw"] = submission["pred_power_kw"].fillna(submission["pred_power_kw"].median())
    submission["pred_p90_kw"] = submission["pred_p90_kw"].fillna(submission["pred_p90_kw"].median())

# Final validation
assert len(submission) == len(targets), f"Row count mismatch: {len(submission)} vs {len(targets)}"
assert submission["timestamp_utc"].nunique() == len(targets), "Duplicate timestamps!"
assert (submission["pred_power_kw"] >= 0).all(), "Negative pred_power_kw!"
assert (submission["pred_p90_kw"] >= submission["pred_power_kw"]).all(), "pred_p90_kw < pred_power_kw!"

submission.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
print(f"  Saved: {OUTPUT_DIR / 'predictions.csv'}")
print(f"  Rows: {len(submission)}")
print(f"  Columns: {list(submission.columns)}")

# Also save predictions vs actuals for validation period
val_comparison = pd.DataFrame({
    "timestamp_utc": pd.to_datetime(val_hours).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "actual_kw": val_actuals,
    "pred_power_kw": val_pred_point,
    "pred_p90_kw": val_pred_p90,
    "error_kw": val_actuals - val_pred_point,
})
val_comparison.to_csv(OUTPUT_DIR / "validation_comparison.csv", index=False)
print(f"  Saved: {OUTPUT_DIR / 'validation_comparison.csv'}")

print("\n" + "=" * 70)
print("DONE! All outputs saved to:", OUTPUT_DIR)
print("=" * 70)
