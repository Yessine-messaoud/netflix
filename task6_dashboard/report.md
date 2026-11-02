# Netflix Content Strategy - Business Insights Report

## Executive summary

- Catalogue: 8,786 titles (6,123 movies, 2,663 TV shows) from 85 countries.
- Content additions peaked in 2019; forecast for the next 12 months: ~1847 titles (-6% vs last 12 months).
- Audience mix skews mature: 46% adult-rated vs 10% kids.
- A Random Forest model predicts maturity group with 61% accuracy.
- A content-based recommender matches genres 9x better than random (see Task 3).

## Methodology

1. **Cleaning** (Task 1): decoded 'Not Given' as missing, removed 4 duplicates, parsed dates and durations, engineered 13 features.
2. **EDA** (Task 2): distributions by type, country, genre, rating, runtime, timing.
3. **Recommender** (Task 3): TF-IDF over weighted metadata + cosine similarity.
4. **Forecasting** (Task 4): 4 models compared on a 12-month hold-out.
5. **Classification** (Task 5): 4 classifiers, stratified split + 5-fold CV.

## Business recommendations

1. **Double down on international originals.** 47% of titles carry an 'International' tag, and India and United Kingdom are the biggest producers after the US. Local productions in these markets give wide reach for the money.
2. **Grow the kids & family slate.** Only 10% of the catalogue is rated for kids, against 46% for adults. Family content keeps household subscriptions and is under-supplied.
3. **Follow genre momentum.** Fastest-growing genres since 2016-17: Comedies, Dramas, Romantic Movies. Losing share: Documentaries, Stand-Up Comedy, International TV Shows.
4. **Invest in multi-season franchises.** 67% of TV shows have a single season. Renewing proven shows builds long-term engagement and lowers acquisition cost per viewing hour.
5. **Plan for a mature catalogue.** Additions peaked in 2019. The random forest (lag features) model projects about 1847 titles in the next 12 months (-6%), so the strategy should move from volume to quality and retention.
6. **Automate content tagging.** The Random Forest classifier predicts a title's maturity group from its metadata with 61% accuracy. It can pre-screen new acquisitions for parental-control review, with a human confirming.

## Limitations

- The dataset has no viewing, revenue or plot-description data, so popularity and semantic similarity cannot be modelled.
- About 29% of directors are unknown; country is missing for 3% of titles.
- The data ends in September 2021, and the forecast assumes past patterns continue.
