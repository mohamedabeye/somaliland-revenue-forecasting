"""Forecasting models and accuracy measures for short annual revenue series.

Every model exposes the same interface:

    fit_predict(years, revenue, horizon) -> dict with keys
        "fitted"  : in-sample fitted values (np.ndarray, same length as revenue)
        "forecast": point forecasts for the next `horizon` years
        "lower", "upper": 95% prediction interval for the forecasts

Revenue is in millions of Somaliland Shillings (SL Sh). Models fitted on the
log scale are transformed back with exp(), so their forecasts are medians.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Accuracy measures (the four used in the original 2015 thesis)
# ---------------------------------------------------------------------------

def accuracy(actual, forecast) -> dict:
    """Bias, MAD, MAPE (%) and MSE, as defined in the thesis (Section 3.5.1).

    Bias = mean(actual - forecast): positive means forecasts are too low.
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    err = actual - forecast
    return {
        "Bias": err.mean(),
        "MAD": np.abs(err).mean(),
        "MAPE (%)": 100 * np.mean(np.abs(err) / actual),
        "MSE": np.mean(err ** 2),
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _ols_interval(x, y, x_new, level=0.95):
    """Ordinary least squares with classical prediction intervals."""
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, p = X.shape
    sigma2 = resid @ resid / (n - p)
    X_new = np.column_stack([np.ones_like(x_new), x_new])
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(sigma2 * (1 + np.sum(X_new @ xtx_inv * X_new, axis=1)))
    t = stats.t.ppf(0.5 + level / 2, n - p)
    mean = X_new @ beta
    return X @ beta, mean, mean - t * se, mean + t * se, beta


def linear_trend(years, revenue, horizon):
    """Original thesis model: revenue = a + b * year (ordinary least squares)."""
    years = np.asarray(years, float)
    future = years[-1] + np.arange(1, horizon + 1)
    fitted, mean, lo, hi, beta = _ols_interval(years, np.asarray(revenue, float), future)
    return {"fitted": fitted, "forecast": mean, "lower": lo, "upper": hi, "coef": beta}


def log_linear_trend(years, revenue, horizon):
    """log(revenue) = a + b * year, i.e. constant percentage growth."""
    years = np.asarray(years, float)
    future = years[-1] + np.arange(1, horizon + 1)
    fitted, mean, lo, hi, beta = _ols_interval(years, np.log(revenue), future)
    return {
        "fitted": np.exp(fitted),
        "forecast": np.exp(mean),
        "lower": np.exp(lo),
        "upper": np.exp(hi),
        "coef": beta,
        "growth_rate": np.exp(beta[1]) - 1,
    }


def holt_log(years, revenue, horizon):
    """Holt's linear exponential smoothing on log(revenue) (additive trend)."""
    y = np.log(np.asarray(revenue, float))
    model = ExponentialSmoothing(y, trend="add", initialization_method="estimated").fit()
    fc = model.forecast(horizon)
    resid_sd = np.std(y - model.fittedvalues, ddof=2)
    steps = np.arange(1, horizon + 1)
    half = 1.96 * resid_sd * np.sqrt(steps)  # simple widening interval
    return {
        "fitted": np.exp(model.fittedvalues),
        "forecast": np.exp(fc),
        "lower": np.exp(fc - half),
        "upper": np.exp(fc + half),
    }


def arima_drift_log(years, revenue, horizon):
    """ARIMA(0,1,0) with drift on log(revenue): a random walk with constant growth."""
    y = np.log(np.asarray(revenue, float))
    res = ARIMA(y, order=(0, 1, 0), trend="t").fit()
    pred = res.get_forecast(horizon)
    ci = pred.conf_int(alpha=0.05)
    fitted = res.fittedvalues.copy()
    fitted[0] = y[0]  # first value has no prior observation
    return {
        "fitted": np.exp(fitted),
        "forecast": np.exp(pred.predicted_mean),
        "lower": np.exp(ci[:, 0]),
        "upper": np.exp(ci[:, 1]),
    }


def bayesian_log_linear(years, revenue, horizon, n_draws=20000, seed=42):
    """Bayesian log-linear trend with a weakly-informative conjugate prior.

    log(revenue_t) = a + b * (year_t - mean_year) + e_t,  e_t ~ N(0, s^2)
    Prior: (a, b) | s^2 ~ N(0, s^2 * 100 I),  s^2 ~ Inverse-Gamma(1, 0.1)

    The posterior predictive distribution is simulated, which gives honest
    uncertainty for a 14-observation series.
    """
    rng = np.random.default_rng(seed)
    years = np.asarray(years, float)
    centre = years.mean()
    x = years - centre
    y = np.log(np.asarray(revenue, float))
    X = np.column_stack([np.ones_like(x), x])

    # Conjugate Normal-Inverse-Gamma update
    V0_inv = np.eye(2) / 100.0
    m0 = np.zeros(2)
    a0, b0 = 1.0, 0.1
    Vn = np.linalg.inv(V0_inv + X.T @ X)
    mn = Vn @ (V0_inv @ m0 + X.T @ y)
    an = a0 + len(y) / 2
    bn = b0 + 0.5 * (y @ y + m0 @ V0_inv @ m0 - mn @ np.linalg.inv(Vn) @ mn)

    sigma2 = 1 / rng.gamma(an, 1 / bn, size=n_draws)
    L = np.linalg.cholesky(Vn)
    beta = mn + (L @ rng.standard_normal((2, n_draws))).T * np.sqrt(sigma2)[:, None]

    x_new = years[-1] + np.arange(1, horizon + 1) - centre
    mu_new = beta[:, [0]] + beta[:, [1]] * x_new[None, :]
    draws = mu_new + rng.standard_normal(mu_new.shape) * np.sqrt(sigma2)[:, None]
    fitted = np.exp(X @ mn)

    return {
        "fitted": fitted,
        "forecast": np.exp(np.median(draws, axis=0)),
        "lower": np.exp(np.percentile(draws, 2.5, axis=0)),
        "upper": np.exp(np.percentile(draws, 97.5, axis=0)),
        "growth_rate_draws": np.exp(beta[:, 1]) - 1,
    }


MODELS = {
    "Linear trend (original 2015 thesis)": linear_trend,
    "Log-linear trend": log_linear_trend,
    "Holt exponential smoothing (log)": holt_log,
    "ARIMA(0,1,0) with drift (log)": arima_drift_log,
    "Bayesian log-linear": bayesian_log_linear,
}


def rolling_origin(years, revenue, model, first_train=8):
    """One-step-ahead forecasts from an expanding window.

    The model is refitted on years[:k] and used to forecast year k, for every
    k from `first_train` to the end. This mimics how a revenue office would
    really forecast next year's collections using only past data.
    """
    years = np.asarray(years)
    revenue = np.asarray(revenue, float)
    preds, actual, when = [], [], []
    for k in range(first_train, len(revenue)):
        out = model(years[:k], revenue[:k], horizon=1)
        preds.append(float(out["forecast"][0]))
        actual.append(revenue[k])
        when.append(int(years[k]))
    return np.array(when), np.array(actual), np.array(preds)
