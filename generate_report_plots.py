"""
Generate all report plots from saved models — no retraining needed.
Usage: python generate_report_plots.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
try:
    from tensorflow.keras.models import load_model
    HAS_TF = True
except ImportError:
    HAS_TF = False

sns.set_style("whitegrid")
OUT = "docs/pictures"
os.makedirs(OUT, exist_ok=True)

# ── Reproduce the exact same X, y, train/val split as the notebook ───────────
df = pd.read_csv("./data/engineered/features.csv")
df = df.drop(columns=[c for c in df.columns if c.strip() == "smooth pathway to front door"])

df["host_response_rate"] = (
    df["host_response_rate"].astype(str).str.rstrip("%")
    .replace("nan", np.nan).astype(float)
)
df["host_response_rate"]    = df["host_response_rate"].fillna(-1)
df["review_scores_rating"]  = df["review_scores_rating"].fillna(-1)
df["walkscore"]             = df["walkscore"].fillna(df["walkscore"].median())
df["transitscore"]          = df["transitscore"].fillna(df["transitscore"].median())
df["bathrooms"]             = df["bathrooms"].fillna(0)
df["beds"]                  = df["beds"].fillna(0)
df["bedrooms"]              = df["bedrooms"].fillna(0)
df["DateDiffHostSince"]     = df["DateDiffHostSince"].fillna(-1)

df = pd.get_dummies(df,
    columns=["property_type", "room_type", "bed_type", "cancellation_policy", "city"],
    drop_first=False)
bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

y = df["log_price"].values
X = df.drop(columns=["log_price"]).values
feature_names = df.drop(columns=["log_price"]).columns.tolist()

X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler().fit(X_tr)
X_tr_s = sc.transform(X_tr)
X_va_s = sc.transform(X_va)

print(f"Features: {X.shape[1]}  |  Val set: {X_va.shape[0]} samples")

# ── Load saved models ────────────────────────────────────────────────────────
lr_model  = joblib.load("saved/models/model_linreg.joblib")
xgb_model = joblib.load("saved/models/model_xgb.joblib")
hgb_model = joblib.load("saved/models/model_histgb.joblib")
svr_model = joblib.load("saved/models/model_linsvr.joblib")
preds = {
    "LinearReg":     lr_model.predict(X_va),
    "XGBoost":       xgb_model.predict(X_va),
    "HistGradBoost": hgb_model.predict(X_va),
    "LinearSVR":     svr_model.predict(X_va_s),
}

if HAS_TF:
    ann_model = load_model("saved/models/model_ann.keras")
    sc_ann    = joblib.load("saved/models/scaler_ann.joblib")
    preds["ANN"] = ann_model.predict(sc_ann.transform(X_va), verbose=0).ravel()
else:
    print("TensorFlow not found — skipping ANN, plotting 4 models.")

cmap = {
    "LinearReg": "#4c72b0", "XGBoost": "#dd8452", "HistGradBoost": "#55a868",
    "LinearSVR": "#c44e52", "ANN": "#8172b2",
}

# ═══ PLOT 1: Actual vs Predicted — all 5 models ═════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes_flat = axes.ravel()
lims = [y_va.min() - 0.3, y_va.max() + 0.3]

for ax, (name, yhat) in zip(axes_flat, preds.items()):
    ax.scatter(y_va, yhat, alpha=0.12, s=6, color=cmap[name])
    ax.plot(lims, lims, "r--", lw=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    r2_v  = r2_score(y_va, yhat)
    rmse_v = np.sqrt(mean_squared_error(y_va, yhat))
    ax.set_title(f"{name}\nR²={r2_v:.4f}  RMSE={rmse_v:.4f}")
    ax.set_aspect("equal")

axes_flat[-1].axis("off")
fig.suptitle("Actual vs. Predicted — All Models", fontsize=14)
fig.tight_layout()
fig.savefig(f"{OUT}/actual_vs_predicted_all.png", dpi=200, bbox_inches="tight")
plt.close()
print("1/6  actual_vs_predicted_all.png")

# ═══ PLOT 2: Residuals — XGBoost ════════════════════════════════════════════
residuals = y_va - preds["XGBoost"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(residuals, bins=60, kde=True, color="#dd8452", ax=ax1)
ax1.axvline(0, color="red", ls="--")
ax1.set_xlabel("Residual (Actual − Predicted)")
ax1.set_ylabel("Count")
ax1.set_title("XGBoost Residual Distribution")

ax2.scatter(preds["XGBoost"], residuals, alpha=0.1, s=6, color="#dd8452")
ax2.axhline(0, color="red", ls="--")
ax2.set_xlabel("Predicted Log Price")
ax2.set_ylabel("Residual")
ax2.set_title("Residuals vs. Predicted Values")
fig.tight_layout()
fig.savefig(f"{OUT}/residuals_xgboost.png", dpi=200)
plt.close()
print("2/6  residuals_xgboost.png")

# ═══ PLOT 3: XGBoost feature importance (top 25) ════════════════════════════
importances = pd.Series(xgb_model.feature_importances_, index=feature_names)
top25 = importances.nlargest(25)

fig, ax = plt.subplots(figsize=(10, 8))
top25.sort_values().plot.barh(color="#dd8452", ax=ax)
ax.set_xlabel("Feature Importance (Gain)")
ax.set_title("XGBoost — Top 25 Feature Importances")
fig.tight_layout()
fig.savefig(f"{OUT}/xgb_feature_importance.png", dpi=200)
plt.close()
print("3/6  xgb_feature_importance.png")

# ═══ PLOT 4: Cross-validation RMSE boxplot ══════════════════════════════════
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_configs = {
    "LinearReg": Pipeline([("sc", StandardScaler()), ("m", LinearRegression())]),
    "XGBoost": XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6,
                             n_jobs=-1, random_state=42, verbosity=0),
    "HistGradBoost": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                                    max_depth=8, random_state=42),
}

cv_rmses = {}
for name, model in cv_configs.items():
    neg_mse = cross_val_score(model, X, y, cv=kf,
                              scoring="neg_mean_squared_error", n_jobs=-1)
    cv_rmses[name] = np.sqrt(-neg_mse)
    print(f"     CV {name}: {cv_rmses[name].mean():.4f} ± {cv_rmses[name].std():.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
bp = ax.boxplot(cv_rmses.values(), labels=cv_rmses.keys(), patch_artist=True)
for patch, name in zip(bp["boxes"], cv_rmses.keys()):
    patch.set_facecolor(cmap[name])
    patch.set_alpha(0.7)
ax.set_ylabel("RMSE (5-fold CV)")
ax.set_title("Cross-Validation RMSE Distribution")
fig.tight_layout()
fig.savefig(f"{OUT}/cv_rmse_boxplot.png", dpi=200)
plt.close()
print("4/6  cv_rmse_boxplot.png")

# ═══ PLOT 5: Holdout metrics bar chart — RMSE, R², MAE ═════════════════════
metrics = {}
for name, yhat in preds.items():
    metrics[name] = {
        "RMSE": np.sqrt(mean_squared_error(y_va, yhat)),
        "R²":   r2_score(y_va, yhat),
        "MAE":  mean_absolute_error(y_va, yhat),
    }
metrics_df = pd.DataFrame(metrics).T

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, col in zip(axes, ["RMSE", "R²", "MAE"]):
    bars = ax.bar(metrics_df.index, metrics_df[col],
                  color=[cmap[n] for n in metrics_df.index],
                  edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, metrics_df[col]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    better = "lower is better" if col != "R²" else "higher is better"
    ax.set_title(f"{col}  ({better})")
    ax.tick_params(axis="x", rotation=25)

fig.suptitle("Holdout Metrics Across All 5 Models", fontsize=14)
fig.tight_layout()
fig.savefig(f"{OUT}/model_metrics_all.png", dpi=200, bbox_inches="tight")
plt.close()
print("5/6  model_metrics_all.png")

# ═══ PLOT 6: Summary table as figure ════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.axis("off")
table_data = []
for name in preds.keys():
    m = metrics[name]
    table_data.append([name, f"{m['RMSE']:.4f}", f"{m['R²']:.4f}", f"{m['MAE']:.4f}"])

table = ax.table(cellText=table_data,
                 colLabels=["Model", "RMSE", "R²", "MAE"],
                 loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.8)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#4c72b0")
        cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#f0f0f0")
ax.set_title("Model Performance Summary (Holdout Set)", fontsize=14, pad=20)
fig.tight_layout()
fig.savefig(f"{OUT}/model_summary_table.png", dpi=200, bbox_inches="tight")
plt.close()
print("6/6  model_summary_table.png")

print(f"\n{'='*50}")
print(f"All 6 plots saved to {OUT}/")
print(f"{'='*50}")
