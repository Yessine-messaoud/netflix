"""Task 6 - End-to-end business insights dashboard (interactive HTML + report).

Combines the cleaned data (Task 1), EDA (Task 2), recommender (Task 3),
forecasting (Task 4) and classifier (Task 5) into one self-contained page.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from task3_recommender.recommender import Recommender  # noqa: E402
from task4_trends.forecast import MODELS as FC_MODELS, metrics  # noqa: E402
from task5_classification.classify import CLASSES, MODELS as CLF_MODELS, build_features  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

DATA = ROOT / "data" / "netflix_clean.csv"
HERE = Path(__file__).resolve().parent
RED, DARK = "#E50914", "#221F1F"
TYPE_COLORS = {"Movie": RED, "TV Show": DARK}


def explode(df, col):
    return (df.assign(**{col: df[col].str.split(",")}).explode(col)
              .assign(**{col: lambda d: d[col].str.strip()}))


def style(fig, height=380):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=50, b=10),
                      font=dict(family="Inter, Segoe UI, sans-serif"), legend_title_text="")
    return fig


def analytics(df):
    """Step 1: complete data analysis -> figures + numbers used in the narrative."""
    figs, facts = {}, {}
    yearly = df[df["year_added"] >= 2015].groupby(["year_added", "type"]).size().reset_index(name="n")
    figs["yearly"] = style(px.bar(yearly, x="year_added", y="n", color="type", barmode="stack",
                                  color_discrete_map=TYPE_COLORS,
                                  title="Titles added per year (2021 = Jan-Sep)",
                                  labels={"year_added": "", "n": "Titles"}))

    countries = explode(df[df["country"] != "Unknown"], "country")
    top_c = countries["country"].value_counts().head(12).index
    cdat = countries[countries["country"].isin(top_c)].groupby(["country", "type"]).size()
    cdat = cdat.reset_index(name="n")
    figs["countries"] = style(px.bar(cdat, y="country", x="n", color="type", orientation="h",
                                     color_discrete_map=TYPE_COLORS, title="Top 12 countries",
                                     labels={"country": "", "n": "Titles"},
                                     category_orders={"country": list(top_c)}), 420)

    genres = explode(df, "listed_in")
    g = genres["listed_in"].value_counts().head(15).iloc[::-1]
    figs["genres"] = style(px.bar(x=g.values, y=g.index, orientation="h", title="Top 15 genres",
                                  labels={"x": "Titles", "y": ""},
                                  color_discrete_sequence=[RED]), 420)

    # Genre momentum: share of 2020-21 additions vs share of 2016-17 additions
    recent = genres[genres["year_added"] >= 2020]["listed_in"].value_counts(normalize=True)
    early = genres[genres["year_added"].between(2016, 2017)]["listed_in"].value_counts(normalize=True)
    momentum = (recent - early).dropna().sort_values()
    mom = pd.concat([momentum.head(6), momentum.tail(6)]) * 100
    figs["momentum"] = style(px.bar(x=mom.values, y=mom.index, orientation="h",
                                    color=np.where(mom > 0, "Growing", "Shrinking"),
                                    color_discrete_map={"Growing": RED, "Shrinking": "#8a8a8a"},
                                    title="Genre momentum: share of additions, 2020-21 vs 2016-17",
                                    labels={"x": "Change in share (percentage points)", "y": ""}),
                              420)
    facts["growing"] = list(momentum.tail(3).index[::-1])
    facts["shrinking"] = list(momentum.head(3).index)

    rg = df.groupby(["rating_group", "type"]).size().reset_index(name="n")
    figs["ratings"] = style(px.sunburst(rg, path=["type", "rating_group"], values="n",
                                        color="type", color_discrete_map=TYPE_COLORS,
                                        title="Audience mix by type"), 380)

    figs["duration"] = style(px.histogram(df.dropna(subset=["duration_min"]), x="duration_min",
                                          nbins=50, color_discrete_sequence=[RED],
                                          title="Movie runtime distribution",
                                          labels={"duration_min": "Minutes"}))

    facts.update(
        total=len(df), movies=int((df["type"] == "Movie").sum()),
        shows=int((df["type"] == "TV Show").sum()),
        countries=countries["country"].nunique(), genres=genres["listed_in"].nunique(),
        top_countries=list(top_c[:3]), adult_share=(df["rating_group"] == "Adults").mean(),
        kids_share=(df["rating_group"] == "Kids").mean(),
        one_season=(df["seasons"] == 1).sum() / df["seasons"].notna().sum(),
        median_runtime=df["duration_min"].median(),
        intl_share=(df["listed_in"].str.contains("International")).mean(),
        peak_year=int(df.groupby("year_added").size().idxmax()),
    )
    return figs, facts


def forecasting(df):
    monthly = df.set_index(pd.to_datetime(df["date_added"])).resample("MS").size()
    monthly = monthly.loc["2016-01-01":"2021-08-01"]
    t = np.arange(len(monthly), dtype=float)
    months = monthly.index.month.to_numpy()
    y = monthly.to_numpy(dtype=float)
    split = len(y) - 12
    scores = {n: metrics(y[split:], f().fit(t[:split], months[:split], y[:split])
                         .predict(t[split:], months[split:]))["MAE"] for n, f in FC_MODELS.items()}
    best = min(scores, key=scores.get)
    fut = pd.date_range(monthly.index[-1] + pd.offsets.MonthBegin(), periods=24, freq="MS")
    fc = FC_MODELS[best]().fit(t, months, y).predict(
        np.arange(len(y), len(y) + 24, dtype=float), fut.month.to_numpy())

    fig = go.Figure()
    fig.add_scatter(x=monthly.index, y=y, name="Actual", line=dict(color=DARK))
    fig.add_scatter(x=fut, y=fc, name=f"Forecast ({best})", line=dict(color=RED, dash="dash"))
    fig.update_layout(title="Monthly additions and 24-month forecast", yaxis_title="Titles")
    return style(fig, 380), {"best": best, "mae": scores[best], "next12": fc[:12].sum(),
                             "last12": y[-12:].sum()}


def classification(df):
    d = df[df["rating_group"].isin(CLASSES)].reset_index(drop=True)
    X, y = build_features(d), d["rating_group"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    rows = []
    for name, model in CLF_MODELS.items():
        pred = model.fit(X_tr, y_tr).predict(X_te)
        rows.append({"model": name, "Accuracy": accuracy_score(y_te, pred),
                     "Macro F1": f1_score(y_te, pred, average="macro")})
    res = pd.DataFrame(rows).melt(id_vars="model", var_name="metric")
    fig = px.bar(res, x="value", y="model", color="metric", barmode="group", orientation="h",
                 color_discrete_map={"Accuracy": DARK, "Macro F1": RED}, range_x=[0, 1],
                 title="Maturity-group classifier: model comparison (test set)",
                 labels={"value": "", "model": ""})
    best = pd.DataFrame(rows).sort_values("Macro F1").iloc[-1]
    return style(fig, 340), best.to_dict()


def recommendations_payload(df, k=5):
    rec = Recommender(df)
    sim = rec.sim
    np.fill_diagonal(sim, -1)
    top = np.argsort(-sim, axis=1)[:, :k]
    return {
        "titles": rec.df["title"].tolist(),
        "meta": (rec.df["type"].astype(str) + " · " + rec.df["listed_in"]).tolist(),
        "recs": top.tolist(),
        "scores": np.round(np.take_along_axis(sim, top, axis=1), 2).tolist(),
    }


def business_recommendations(f, fc, clf):
    return [
        ("Double down on international originals",
         f"{f['intl_share']:.0%} of titles carry an 'International' tag, and "
         f"{' and '.join(f['top_countries'][1:])} are the biggest producers after the US. "
         "Local productions in these markets give wide reach for the money."),
        ("Grow the kids & family slate",
         f"Only {f['kids_share']:.0%} of the catalogue is rated for kids, against "
         f"{f['adult_share']:.0%} for adults. Family content keeps household subscriptions "
         "and is under-supplied."),
        ("Follow genre momentum",
         f"Fastest-growing genres since 2016-17: {', '.join(f['growing'])}. "
         f"Losing share: {', '.join(f['shrinking'])}."),
        ("Invest in multi-season franchises",
         f"{f['one_season']:.0%} of TV shows have a single season. Renewing proven shows builds "
         "long-term engagement and lowers acquisition cost per viewing hour."),
        ("Plan for a mature catalogue",
         f"Additions peaked in {f['peak_year']}. The {fc['best'].lower()} model projects "
         f"about {fc['next12']:.0f} titles in the next 12 months ({fc['next12'] / fc['last12'] - 1:+.0%}), "
         "so the strategy should move from volume to quality and retention."),
        ("Automate content tagging",
         f"The {clf['model']} classifier predicts a title's maturity group from its metadata with "
         f"{clf['Accuracy']:.0%} accuracy. It can pre-screen new acquisitions for parental-control "
         "review, with a human confirming."),
    ]


def render(figs, facts, fc, clf, payload, recs):
    def div(key, first=False):
        return figs[key].to_html(full_html=False, include_plotlyjs=first,
                                 config={"displaylogo": False, "responsive": True})

    kpis = [("Titles", f"{facts['total']:,}"), ("Movies", f"{facts['movies']:,}"),
            ("TV Shows", f"{facts['shows']:,}"), ("Countries", facts["countries"]),
            ("Genres", facts["genres"]), ("Median runtime", f"{facts['median_runtime']:.0f} min")]
    kpi_html = "".join(f'<div class="kpi"><span>{v}</span><small>{k}</small></div>'
                       for k, v in kpis)
    rec_html = "".join(f"<li><b>{t}</b><p>{d}</p></li>" for t, d in recs)

    template = (HERE / "template.html").read_text(encoding="utf-8")
    return (template
            .replace("{{KPIS}}", kpi_html)
            .replace("{{FIG_YEARLY}}", div("yearly", first=True))
            .replace("{{FIG_FORECAST}}", div("forecast"))
            .replace("{{FIG_COUNTRIES}}", div("countries"))
            .replace("{{FIG_GENRES}}", div("genres"))
            .replace("{{FIG_MOMENTUM}}", div("momentum"))
            .replace("{{FIG_RATINGS}}", div("ratings"))
            .replace("{{FIG_DURATION}}", div("duration"))
            .replace("{{FIG_CLF}}", div("clf"))
            .replace("{{FC_NOTE}}", f"Best model on a 12-month hold-out: <b>{fc['best']}</b> "
                                    f"(MAE {fc['mae']:.0f} titles/month).")
            .replace("{{CLF_NOTE}}", f"Best: <b>{clf['model']}</b>, accuracy {clf['Accuracy']:.0%}, "
                                     f"macro-F1 {clf['Macro F1']:.2f} (majority baseline: 46%).")
            .replace("{{RECS}}", rec_html)
            .replace("{{PAYLOAD}}", json.dumps(payload, separators=(",", ":"))))


def write_report(facts, fc, clf, recs):
    lines = ["# Netflix Content Strategy - Business Insights Report\n",
             "## Executive summary\n",
             f"- Catalogue: {facts['total']:,} titles ({facts['movies']:,} movies, "
             f"{facts['shows']:,} TV shows) from {facts['countries']} countries.",
             f"- Content additions peaked in {facts['peak_year']}; forecast for the next 12 months: "
             f"~{fc['next12']:.0f} titles ({fc['next12'] / fc['last12'] - 1:+.0%} vs last 12 months).",
             f"- Audience mix skews mature: {facts['adult_share']:.0%} adult-rated vs "
             f"{facts['kids_share']:.0%} kids.",
             f"- A {clf['model']} model predicts maturity group with {clf['Accuracy']:.0%} accuracy.",
             "- A content-based recommender matches genres 9x better than random (see Task 3).\n",
             "## Methodology\n",
             "1. **Cleaning** (Task 1): decoded 'Not Given' as missing, removed 4 duplicates, "
             "parsed dates and durations, engineered 13 features.",
             "2. **EDA** (Task 2): distributions by type, country, genre, rating, runtime, timing.",
             "3. **Recommender** (Task 3): TF-IDF over weighted metadata + cosine similarity.",
             "4. **Forecasting** (Task 4): 4 models compared on a 12-month hold-out.",
             "5. **Classification** (Task 5): 4 classifiers, stratified split + 5-fold CV.\n",
             "## Business recommendations\n"]
    lines += [f"{i}. **{t}.** {d}" for i, (t, d) in enumerate(recs, 1)]
    lines += ["", "## Limitations\n",
              "- The dataset has no viewing, revenue or plot-description data, so popularity "
              "and semantic similarity cannot be modelled.",
              "- About 29% of directors are unknown; country is missing for 3% of titles.",
              "- The data ends in September 2021, and the forecast assumes past patterns continue."]
    (HERE / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    df = pd.read_csv(DATA)
    figs, facts = analytics(df)
    figs["forecast"], fc = forecasting(df)
    figs["clf"], clf = classification(df)
    payload = recommendations_payload(df)
    recs = business_recommendations(facts, fc, clf)
    out = HERE / "dashboard.html"
    out.write_text(render(figs, facts, fc, clf, payload, recs), encoding="utf-8")
    write_report(facts, fc, clf, recs)
    print(f"Dashboard -> {out} ({out.stat().st_size / 1e6:.1f} MB)\nReport -> {HERE / 'report.md'}")


if __name__ == "__main__":
    main()
