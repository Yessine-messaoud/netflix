"""Task 3 - Content-based recommendation system for Netflix titles.

The dataset has no plot descriptions, so each title is represented by its
metadata: genres, director, country, type and maturity group.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "netflix_clean.csv"


def tokens(value, prefix):
    """'Crime TV Shows, TV Dramas' -> 'g_crimetvshows g_tvdramas' (one token per entity)."""
    if pd.isna(value) or value == "Unknown":
        return ""
    return " ".join(prefix + re.sub(r"[^a-z0-9]", "", v.lower()) for v in value.split(","))


class Recommender:
    # Genres are repeated so they dominate similarity; people/places refine it.
    WEIGHTS = {"listed_in": ("g_", 3), "director": ("d_", 2), "country": ("c_", 1),
               "type": ("t_", 1), "rating_group": ("r_", 1)}

    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.df["soup"] = self.df.apply(self._soup, axis=1)
        # Step 2: text preprocessing (tokenised soup -> TF-IDF)
        self.vec = TfidfVectorizer(token_pattern=r"\S+", sublinear_tf=True)
        self.matrix = self.vec.fit_transform(self.df["soup"])
        # Step 3: content similarity (TF-IDF rows are L2-normalised, so dot product = cosine)
        self.sim = linear_kernel(self.matrix, self.matrix)
        self.index = pd.Series(self.df.index, index=self.df["title"].str.lower())

    def _soup(self, row):
        return " ".join(" ".join([tokens(row[c], p)] * w) for c, (p, w) in self.WEIGHTS.items())

    def recommend(self, title, n=10):
        idx = self.index.get(title.lower())
        if idx is None:
            raise KeyError(f"'{title}' not in catalogue")
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]
        scores = self.sim[idx].copy()
        scores[idx] = -1
        top = np.argsort(scores)[::-1][:n]
        out = self.df.loc[top, ["title", "type", "listed_in", "director", "country"]].copy()
        out["similarity"] = scores[top].round(3)
        return out

    # Step 5: evaluation
    def evaluate(self, n=10, sample=500, seed=42):
        """Compare the model's top-n lists against random recommendations."""
        rng = np.random.default_rng(seed)
        genres = self.df["listed_in"].str.split(",").apply(lambda g: {x.strip() for x in g})
        types = self.df["type"].to_numpy()
        queries = rng.choice(len(self.df), size=sample, replace=False)

        def score(rec_lists):
            jac, same_type, hit = [], [], []
            for q, recs in zip(queries, rec_lists):
                for r in recs:
                    inter = len(genres[q] & genres[r])
                    jac.append(inter / len(genres[q] | genres[r]))
                    hit.append(inter > 0)
                    same_type.append(types[q] == types[r])
            return np.mean(jac), np.mean(hit), np.mean(same_type)

        model_lists, recommended = [], set()
        for q in queries:
            s = self.sim[q].copy()
            s[q] = -1
            top = np.argsort(s)[::-1][:n]
            model_lists.append(top)
            recommended.update(top)
        random_lists = [rng.choice(np.delete(np.arange(len(self.df)), q), n, replace=False)
                        for q in queries]

        m, r = score(model_lists), score(random_lists)
        return pd.DataFrame(
            {"Content model": [*m, len(recommended) / len(self.df)],
             "Random baseline": [*r, np.nan]},
            index=[f"Genre Jaccard@{n}", f"Genre hit-rate@{n}", f"Same type@{n}",
                   "Catalogue coverage"]).round(3)


def main():
    # Step 1: extract relevant content features
    df = pd.read_csv(DATA)
    rec = Recommender(df)
    print(f"TF-IDF matrix: {rec.matrix.shape[0]} titles x {rec.matrix.shape[1]} features")

    # Step 4: recommendations for selected titles
    pd.set_option("display.width", 200, "display.max_colwidth", 45)
    lines = ["# Task 3 - Recommendation results\n"]
    for title in ["Midnight Mass", "Stranger Things", "Narcos", "The Irishman", "Ganglands"]:
        try:
            out = rec.recommend(title, n=5)
        except KeyError as e:
            print(e)
            continue
        print(f"\nBecause you watched '{title}':\n{out.to_string(index=False)}")
        lines += [f"## Because you watched *{title}*\n", out.to_markdown(index=False), ""]

    report = rec.evaluate()
    print(f"\nEvaluation (500 random query titles, top-10):\n{report}")
    lines += ["## Evaluation (500 random queries, top-10)\n", report.to_markdown(), ""]
    (Path(__file__).resolve().parent / "results.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
