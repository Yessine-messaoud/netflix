"""Task 5 - ML classification: predict a title's audience maturity group.

Target: rating_group (Kids / Family / Teens / Adults). Predicting Movie vs TV Show
would be trivial (duration units and 'TV ...' genre names give it away), so the
maturity audience is a more useful and honest classification problem.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "netflix_clean.csv"
HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

CLASSES = ["Kids", "Family", "Teens", "Adults"]
TOP_COUNTRIES = 20


def split_list(s):
    return [x.strip() for x in s.split(",")]


def build_features(df):
    """Step 1: numeric + multi-hot genre + multi-hot top-country features."""
    genres = MultiLabelBinarizer()
    G = pd.DataFrame(genres.fit_transform(df["listed_in"].map(split_list)),
                     columns=[f"genre: {g}" for g in genres.classes_], index=df.index)

    countries = df["country"].map(split_list)
    top = countries.explode().value_counts().drop("Unknown", errors="ignore").index[:TOP_COUNTRIES]
    C = pd.DataFrame({f"country: {c}": countries.map(lambda cs, c=c: c in cs).astype(int)
                      for c in top}, index=df.index)

    num = pd.DataFrame({
        "is_movie": df["is_movie"],
        "release_year": df["release_year"],
        "year_added": df["year_added"],
        "years_to_netflix": df["years_to_netflix"],
        "duration_min": df["duration_min"].fillna(0),
        "seasons": df["seasons"].fillna(0),
        "n_genres": df["n_genres"],
        "director_known": (df["director"] != "Unknown").astype(int),
    }, index=df.index)
    return pd.concat([num, G, C], axis=1)


MODELS = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
    "Decision Tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=5,
                                            class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                            class_weight="balanced", n_jobs=-1, random_state=42),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, class_weight="balanced", random_state=42),
}


def main():
    df = pd.read_csv(DATA)
    df = df[df["rating_group"].isin(CLASSES)].reset_index(drop=True)
    X = build_features(df)
    y = df["rating_group"]
    print(f"Samples: {len(X)}, features: {X.shape[1]}")
    print(f"Class balance:\n{y.value_counts(normalize=True).round(3)}\n")

    # Step 2: train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Steps 3-4: train and evaluate
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    rows, fitted = {}, {}
    majority = y_tr.value_counts().idxmax()
    rows["Majority-class baseline"] = {
        "CV macro-F1": np.nan,
        "Test accuracy": accuracy_score(y_te, [majority] * len(y_te)),
        "Test macro-F1": f1_score(y_te, [majority] * len(y_te), average="macro")}
    for name, model in MODELS.items():
        cv_f1 = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="f1_macro", n_jobs=-1)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        fitted[name] = model
        rows[name] = {"CV macro-F1": cv_f1.mean(),
                      "Test accuracy": accuracy_score(y_te, pred),
                      "Test macro-F1": f1_score(y_te, pred, average="macro")}
        print(f"{name:24s} CV F1={cv_f1.mean():.3f}±{cv_f1.std():.3f}  "
              f"test acc={rows[name]['Test accuracy']:.3f}")

    # Step 5: compare
    results = pd.DataFrame(rows).T.round(3).sort_values("Test macro-F1", ascending=False)
    best = results.index[0]
    best_pred = fitted[best].predict(X_te)
    report = classification_report(y_te, best_pred, labels=CLASSES)
    print(f"\n{results}\n\nBest model: {best}\n{report}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    results[["Test accuracy", "Test macro-F1"]].plot.barh(ax=axes[0],
                                                          color=["#221F1F", "#E50914"])
    axes[0].set(title="Model comparison (test set)", xlim=(0, 1))
    axes[0].invert_yaxis()
    ConfusionMatrixDisplay.from_predictions(y_te, best_pred, labels=CLASSES, cmap="Reds",
                                            normalize="true", values_format=".2f", ax=axes[1],
                                            colorbar=False)
    axes[1].set_title(f"Confusion matrix (row-normalised) - {best}")
    fig.tight_layout()
    fig.savefig(FIG / "01_model_comparison.png", dpi=130)
    plt.close(fig)

    rf = fitted["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=X.columns).nlargest(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=imp.values, y=imp.index, color="#E50914", ax=ax)
    ax.set(title="Top 15 features (Random Forest importance)", xlabel="Importance")
    fig.tight_layout()
    fig.savefig(FIG / "02_feature_importance.png", dpi=130)
    plt.close(fig)

    lines = ["# Task 5 - Classification results\n",
             "**Target:** audience maturity group (Kids / Family / Teens / Adults), "
             f"{len(X)} titles, {X.shape[1]} features, 80/20 stratified split, 5-fold CV.\n",
             results.to_markdown(), "", f"**Best model:** {best}\n",
             "```", report, "```", "",
             "## Top features (Random Forest)\n", imp.round(4).to_markdown()]
    (HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
