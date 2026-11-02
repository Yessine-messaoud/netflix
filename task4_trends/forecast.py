"""Task 4 - Trend analysis and forecasting of Netflix content growth."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "netflix_clean.csv"
HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

TEST_MONTHS = 12
HORIZON = 24


def design(t, months, degree=1):
    """Trend terms (t, t^2, ...) plus one-hot month-of-year seasonality."""
    trend = np.column_stack([t ** d for d in range(1, degree + 1)])
    season = np.eye(12)[months - 1][:, 1:]
    return np.hstack([trend, season])


class TrendSeasonal:
    def __init__(self, degree, alpha=None):
        self.degree = degree
        reg = Ridge(alpha=alpha) if alpha else LinearRegression()
        self.model = make_pipeline(StandardScaler(), reg)

    def fit(self, t, months, y):
        self.model.fit(design(t, months, self.degree), y)
        return self

    def predict(self, t, months):
        return np.clip(self.model.predict(design(t, months, self.degree)), 0, None)


class LagForest:
    """Random forest on lag features, forecasting recursively one month at a time."""
    LAGS = (1, 2, 3, 12)

    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=0)

    def _row(self, history, month):
        return [history[-l] for l in self.LAGS] + [np.mean(history[-3:]), month]

    def fit(self, t, months, y):
        X = [self._row(y[:i], months[i]) for i in range(12, len(y))]
        self.model.fit(X, y[12:])
        self.history = list(y)
        return self

    def predict(self, t, months):
        history, preds = list(self.history), []
        for m in months:
            p = self.model.predict([self._row(history, m)])[0]
            preds.append(p)
            history.append(p)
        return np.array(preds)


class SeasonalNaive:
    def fit(self, t, months, y):
        self.last_year = y[-12:]
        return self

    def predict(self, t, months):
        return np.resize(self.last_year, len(months))


MODELS = {
    "Seasonal naive (baseline)": SeasonalNaive,
    "Linear trend + seasonality": lambda: TrendSeasonal(1),
    "Quadratic trend + seasonality": lambda: TrendSeasonal(2, alpha=1.0),
    "Random forest (lag features)": LagForest,
}


def metrics(y, p):
    return {"MAE": mean_absolute_error(y, p),
            "RMSE": np.sqrt(mean_squared_error(y, p)),
            "MAPE %": np.mean(np.abs((y - p) / np.maximum(y, 1))) * 100}


def main():
    df = pd.read_csv(DATA, parse_dates=["date_added"])

    # Step 1: prepare release-year and monthly addition series
    release = df.groupby(["release_year", "type"]).size().unstack(fill_value=0)
    release = release.loc[1990:]
    # Netflix additions start in earnest in 2016; the last month (Sep 2021) is partial.
    monthly = (df.set_index("date_added").resample("MS").size()
                 .loc["2016-01-01":"2021-08-01"])
    t = np.arange(len(monthly), dtype=float)
    months = monthly.index.month.to_numpy()
    y = monthly.to_numpy(dtype=float)

    # Step 2: yearly trends
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    release.plot(ax=axes[0], color=["#E50914", "#221F1F"], marker=".")
    axes[0].set(title="Titles by release year (catalogue)", xlabel="Release year", ylabel="Titles")
    tv_share = df.groupby("year_added")["type"].apply(lambda s: (s == "TV Show").mean())
    tv_share.loc[2015:].mul(100).plot(ax=axes[1], marker="o", color="#221F1F")
    axes[1].set(title="TV-show share of yearly additions", xlabel="Year added", ylabel="%")
    fig.tight_layout()
    fig.savefig(FIG / "01_yearly_trends.png", dpi=130)
    plt.close(fig)

    # Step 3: build and evaluate forecasting models on a 12-month hold-out
    split = len(y) - TEST_MONTHS
    rows, test_preds = {}, {}
    for name, factory in MODELS.items():
        m = factory().fit(t[:split], months[:split], y[:split])
        test_preds[name] = m.predict(t[split:], months[split:])
        rows[name] = metrics(y[split:], test_preds[name])
    scores = pd.DataFrame(rows).T.round(1).sort_values("MAE")
    best = scores.index[0]
    print(f"Hold-out evaluation (last {TEST_MONTHS} months):\n{scores}\n\nBest model: {best}")

    # Step 4: refit best model on all data and forecast the next 24 months
    future_idx = pd.date_range(monthly.index[-1] + pd.offsets.MonthBegin(), periods=HORIZON,
                               freq="MS")
    t_future = np.arange(len(y), len(y) + HORIZON, dtype=float)
    final = MODELS[best]().fit(t, months, y)
    forecast = pd.Series(final.predict(t_future, future_idx.month.to_numpy()), index=future_idx)
    resid_sd = np.std(y[split:] - test_preds[best])

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(monthly.index, y, color="#221F1F", label="Actual")
    for name, p in test_preds.items():
        ax.plot(monthly.index[split:], p, "--", lw=1, label=f"{name} (test)")
    ax.plot(forecast.index, forecast, color="#E50914", lw=2.2, label=f"Forecast - {best}")
    ax.fill_between(forecast.index, (forecast - 1.96 * resid_sd).clip(lower=0),
                    forecast + 1.96 * resid_sd, color="#E50914", alpha=0.15, label="~95% band")
    ax.axvline(monthly.index[split], color="grey", ls=":")
    ax.set(title="Monthly titles added to Netflix: hold-out test and 24-month forecast",
           ylabel="Titles added")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "02_forecast.png", dpi=130)
    plt.close(fig)

    # Step 5: interpret
    yearly_fc = forecast.groupby(forecast.index.year).sum().round()
    last_12 = y[-12:].sum()
    next_12 = forecast.iloc[:12].sum()
    lines = [
        "# Task 4 - Trend prediction results\n",
        f"Hold-out test: last {TEST_MONTHS} months ({monthly.index[split]:%b %Y} to "
        f"{monthly.index[-1]:%b %Y}).\n",
        scores.to_markdown(), "",
        f"**Best model:** {best}\n",
        "## Forecast (titles added per calendar year)\n",
        yearly_fc.rename("forecast").to_markdown(), "",
        "## Interpretation\n",
        f"- Monthly additions grew fast between 2016 and 2019, then levelled off and fell slightly "
        f"in 2020-2021.",
        f"- The TV-show share of additions {'rose' if tv_share.loc[2021] > tv_share.loc[2016] else 'fell'} "
        f"from {tv_share.loc[2016]:.0%} in 2016 to {tv_share.loc[2021]:.0%} in 2021 "
        f"(peak {tv_share.loc[2016:].max():.0%} in {tv_share.loc[2016:].idxmax()}).",
        f"- The forecast is {next_12:.0f} titles over the next 12 months, against {last_12:.0f} "
        f"in the last 12 observed ({(next_12 / last_12 - 1):+.0%}).",
        "- With only about 5 years of monthly history and a clear break around 2020, the "
        "forecast is a rough guide: treat it as a range, not an exact number.",
    ]
    (HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-5:]))
    forecast.rename("forecast").to_csv(HERE / "forecast.csv", index_label="month")


if __name__ == "__main__":
    main()
