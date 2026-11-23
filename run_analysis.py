"""Reproduce every table and figure in the README.

Usage:
    python run_analysis.py

Outputs are written to results/ (tables) and figures/ (charts).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.forecasting import (
    MODELS,
    accuracy,
    bayesian_log_linear,
    linear_trend,
    log_linear_trend,
    rolling_origin,
)

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

COLORS = {
    "Actual": "#1f2937",
    "Linear trend (original 2015 thesis)": "#d1495b",
    "Log-linear trend": "#2a9d8f",
    "Holt exponential smoothing (log)": "#e9c46a",
    "ARIMA(0,1,0) with drift (log)": "#6a4c93",
    "Bayesian log-linear": "#264653",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})

df = pd.read_csv(ROOT / "data" / "kalabaydh_revenue_1999_2012.csv")
years = df["year"].to_numpy()
revenue = df["revenue_sl_sh_million"].to_numpy(dtype=float)


def fmt(table: pd.DataFrame) -> pd.DataFrame:
    return table.round({"Bias": 1, "MAD": 1, "MAPE (%)": 1, "MSE": 0})


# ---------------------------------------------------------------------------
# 1. Reproduce the original thesis model
# ---------------------------------------------------------------------------
orig = linear_trend(years, revenue, horizon=1)
a, b = orig["coef"]
r2_lin = 1 - np.sum((revenue - orig["fitted"]) ** 2) / np.sum((revenue - revenue.mean()) ** 2)
loglin = log_linear_trend(years, revenue, horizon=1)
r2_log = 1 - np.sum((revenue - loglin["fitted"]) ** 2) / np.sum((revenue - revenue.mean()) ** 2)

summary = [
    f"Original linear trend: revenue = {a:,.0f} + {b:,.1f} * year",
    f"  R^2 = {r2_lin:.3f}; fitted value for 1999 = {orig['fitted'][0]:,.0f} million SL Sh",
    f"Log-linear trend: annual growth = {100 * loglin['growth_rate']:.1f}%",
    f"  R^2 (original scale) = {r2_log:.3f}",
]
(RESULTS / "model_summary.txt").write_text("\n".join(summary) + "\n")
print("\n".join(summary))

# ---------------------------------------------------------------------------
# 2. In-sample fit of every model
# ---------------------------------------------------------------------------
in_sample = {}
for name, model in MODELS.items():
    out = model(years, revenue, horizon=1)
    in_sample[name] = accuracy(revenue, out["fitted"])
in_sample = fmt(pd.DataFrame(in_sample).T)
in_sample.to_csv(RESULTS / "in_sample_accuracy.csv")

# ---------------------------------------------------------------------------
# 3. Out-of-sample: one-step-ahead rolling-origin forecasts, 2007-2012
# ---------------------------------------------------------------------------
oos, oos_preds = {}, {}
for name, model in MODELS.items():
    when, actual, preds = rolling_origin(years, revenue, model, first_train=8)
    oos[name] = accuracy(actual, preds)
    oos_preds[name] = preds
oos = fmt(pd.DataFrame(oos).T).sort_values("MAPE (%)")
oos.to_csv(RESULTS / "out_of_sample_accuracy.csv")
pd.DataFrame(oos_preds, index=when).assign(Actual=actual).round(0).to_csv(
    RESULTS / "out_of_sample_forecasts.csv", index_label="year"
)

print("\nIn-sample accuracy\n", in_sample.to_string())
print("\nOut-of-sample accuracy (one-step-ahead, 2007-2012)\n", oos.to_string())

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

# Figure 1: original vs log-linear fit
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(years, revenue, color="#cbd5e1", label="Actual revenue", zorder=2)
ax.plot(years, orig["fitted"], "-o", color=COLORS["Linear trend (original 2015 thesis)"],
        label=f"Linear trend (2015 thesis), R² = {r2_lin:.2f}", zorder=3)
ax.plot(years, loglin["fitted"], "-o", color=COLORS["Log-linear trend"],
        label=f"Log-linear trend, R² = {r2_log:.2f}", zorder=3)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Kalabaydh customs revenue, 1999–2012: original vs improved model")
ax.set_xlabel("Year")
ax.set_ylabel("Revenue (million SL Sh)")
ax.legend(frameon=False)
ax.annotate("Linear model predicts\nnegative revenue", xy=(1999, orig["fitted"][0]),
            xytext=(2001.0, -1250), arrowprops=dict(arrowstyle="->"), fontsize=9)
ax.set_ylim(bottom=orig["fitted"][0] * 1.35)
fig.tight_layout()
fig.savefig(FIGURES / "01_original_vs_loglinear.png")
plt.close(fig)

# Figure 2: out-of-sample one-step-ahead forecasts
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(when, actual, "-o", color=COLORS["Actual"], lw=2.5, label="Actual")
for name, preds in oos_preds.items():
    ax.plot(when, preds, "--o", color=COLORS[name], label=name, ms=4)
ax.set_title("One-step-ahead forecasts using only past data (2007–2012)")
ax.set_xlabel("Year forecast")
ax.set_ylabel("Revenue (million SL Sh)")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES / "02_out_of_sample_forecasts.png")
plt.close(fig)

# Figure 3: MAPE comparison
fig, ax = plt.subplots(figsize=(8, 3.8))
ordered = oos["MAPE (%)"].sort_values()
ax.barh(ordered.index, ordered.values, color=[COLORS[n] for n in ordered.index])
for i, v in enumerate(ordered.values):
    ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Mean absolute percentage error (lower is better)")
ax.set_title("Out-of-sample forecast error by model")
fig.tight_layout()
fig.savefig(FIGURES / "03_mape_comparison.png")
plt.close(fig)

# Figure 4: Bayesian forecast with uncertainty (illustration, 2013-2015)
bayes = bayesian_log_linear(years, revenue, horizon=3)
future = years[-1] + np.arange(1, 4)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(years, revenue, "-o", color=COLORS["Actual"], label="Actual")
ax.plot(future, bayes["forecast"], "-o", color=COLORS["Bayesian log-linear"],
        label="Bayesian forecast (median)")
ax.fill_between(future, bayes["lower"], bayes["upper"], color=COLORS["Bayesian log-linear"],
                alpha=0.2, label="95% prediction interval")
g = bayes["growth_rate_draws"]
ax.set_title(f"Bayesian forecast: growth {100 * np.median(g):.0f}% a year "
             f"(95% CI {100 * np.percentile(g, 2.5):.0f}–{100 * np.percentile(g, 97.5):.0f}%)")
ax.set_xlabel("Year")
ax.set_ylabel("Revenue (million SL Sh)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIGURES / "04_bayesian_forecast.png")
plt.close(fig)

pd.DataFrame({
    "year": future,
    "median": bayes["forecast"].round(0),
    "lower_95": bayes["lower"].round(0),
    "upper_95": bayes["upper"].round(0),
}).to_csv(RESULTS / "bayesian_forecast_2013_2015.csv", index=False)

print("\nFigures saved to", FIGURES)
