"""Task 2 - Exploratory Data Analysis of the Netflix dataset."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "netflix_clean.csv"
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")
TYPE_COLORS = {"Movie": "#E50914", "TV Show": "#221F1F"}


def explode(df, col):
    """Split a comma-separated column into one row per value."""
    return (df.assign(**{col: df[col].str.split(",")})
              .explode(col)
              .assign(**{col: lambda d: d[col].str.strip()}))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=130)
    plt.close(fig)


def main():
    df = pd.read_csv(DATA, parse_dates=["date_added"])
    findings = []

    # Step 1: structure and statistics
    print(df.info())
    print(df.describe(include="all").T[["count", "unique", "top", "mean", "min", "max"]])

    # Step 2: content distribution by type
    type_counts = df["type"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(type_counts, labels=type_counts.index, autopct="%1.1f%%",
           colors=[TYPE_COLORS[t] for t in type_counts.index], textprops={"color": "w"})
    ax.legend(loc="lower center", ncol=2)
    ax.set_title("Content split by type")
    save(fig, "01_type_split.png")
    findings.append(f"Movies make up {type_counts['Movie'] / len(df):.0%} of the catalogue "
                    f"({type_counts['Movie']} movies vs {type_counts['TV Show']} TV shows).")

    # Step 3: top countries and categories
    countries = explode(df[df["country"] != "Unknown"], "country")
    top_c = countries.groupby(["country", "type"]).size().unstack(fill_value=0)
    top_c = top_c.loc[top_c.sum(axis=1).nlargest(10).index]
    fig, ax = plt.subplots(figsize=(9, 5))
    top_c.iloc[::-1].plot.barh(stacked=True, color=TYPE_COLORS, ax=ax)
    ax.set(title="Top 10 producing countries", xlabel="Titles", ylabel="")
    save(fig, "02_top_countries.png")
    findings.append(f"Top 3 countries: {', '.join(top_c.index[:3])}. The US alone contributes "
                    f"{top_c.loc['United States'].sum()} titles.")

    genres = explode(df, "listed_in")
    top_g = genres["listed_in"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=top_g.values, y=top_g.index, ax=ax, color="#E50914")
    ax.set(title="Top 15 genres / categories", xlabel="Titles", ylabel="")
    save(fig, "03_top_genres.png")
    findings.append(f"Most common categories: {', '.join(top_g.index[:3])}.")

    # Step 4: visualizations for key metrics
    yearly = df.groupby(["year_added", "type"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    yearly.plot(marker="o", color=TYPE_COLORS, ax=ax)
    ax.set(title="Titles added to Netflix per year", xlabel="Year added", ylabel="Titles")
    save(fig, "04_added_per_year.png")
    peak = yearly.sum(axis=1).idxmax()
    findings.append(f"Additions peaked in {peak} ({yearly.sum(axis=1).max()} titles); "
                    "2021 data stops in September.")

    order = df["rating"].value_counts().index
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(data=df, x="rating", hue="type", order=order, palette=TYPE_COLORS, ax=ax)
    ax.set(title="Maturity rating distribution", xlabel="", ylabel="Titles")
    save(fig, "05_ratings.png")
    findings.append(f"Most frequent rating is {order[0]}; "
                    f"{(df['rating_group'] == 'Adults').mean():.0%} of titles target adults.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(df["duration_min"].dropna(), bins=40, color="#E50914", ax=axes[0])
    axes[0].set(title="Movie duration (minutes)", xlabel="Minutes")
    sns.countplot(x=df["seasons"].dropna().astype(int), color="#221F1F", ax=axes[1])
    axes[1].set(title="TV show seasons", xlabel="Seasons")
    save(fig, "06_durations.png")
    findings.append(f"Median movie runs {df['duration_min'].median():.0f} min; "
                    f"{(df['seasons'] == 1).sum() / df['seasons'].notna().sum():.0%} "
                    "of TV shows have only one season.")

    month = df.groupby("month_name_added").size().reindex(
        pd.date_range("2000-01-01", periods=12, freq="MS").month_name())
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=month.index.str[:3], y=month.values, color="#E50914", ax=ax)
    ax.set(title="Titles added by month", xlabel="", ylabel="Titles")
    save(fig, "07_added_by_month.png")
    findings.append(f"Busiest month for additions: {month.idxmax()}.")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.histplot(data=df, x="years_to_netflix", hue="type", bins=50, palette=TYPE_COLORS,
                 element="step", ax=ax)
    ax.set(title="Gap between release and Netflix addition", xlabel="Years", xlim=(0, 40))
    save(fig, "08_release_gap.png")
    findings.append(f"Median gap from release to Netflix: {df['years_to_netflix'].median():.0f} "
                    f"year(s); {(df['years_to_netflix'] == 0).mean():.0%} were added the year "
                    "they came out.")

    directors = df.loc[df["director"] != "Unknown", "director"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=directors.values, y=directors.index, color="#221F1F", ax=ax)
    ax.set(title="Top 10 directors by number of titles", xlabel="Titles", ylabel="")
    save(fig, "09_top_directors.png")

    # Step 5: summarize findings
    report = "# Task 2 - EDA findings\n\n" + "\n".join(f"- {f}" for f in findings) + "\n"
    (Path(__file__).resolve().parent / "findings.md").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
