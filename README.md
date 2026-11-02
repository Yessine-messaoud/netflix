# Netflix Data Science Project - Auspify Internship

All 6 tasks from the Auspify Data Science task list, run on `Dataset.csv` (8,790 Netflix titles, data up to September 2021).

| # | Task | Folder | Output |
|---|------|--------|--------|
| 1 | Data Cleaning & Preprocessing | `task1_cleaning/` | `data/netflix_clean.csv` |
| 2 | Exploratory Data Analysis | `task2_eda/` | `figures/*.png`, `findings.md` |
| 3 | Recommendation System | `task3_recommender/` | `results.md` |
| 4 | Trend Prediction | `task4_trends/` | `figures/*.png`, `forecast.csv`, `results.md` |
| 5 | ML Classification | `task5_classification/` | `figures/*.png`, `results.md` |
| 6 | Business Insights Dashboard | `task6_dashboard/` | `dashboard.html`, `report.md` |

## Run

```bash
pip install -r requirements.txt
python task1_cleaning/clean.py          # must run first: the other tasks read data/netflix_clean.csv
python task2_eda/eda.py
python task3_recommender/recommender.py
python task4_trends/forecast.py
python task5_classification/classify.py
python task6_dashboard/dashboard.py     # then open task6_dashboard/dashboard.html in a browser
```

## Key results

- **Cleaning:** missing values were hidden as `"Not Given"` (2,588 directors, 287 countries). 4 duplicate titles were removed and 13 features engineered (year/month added, duration in minutes/seasons, maturity group, primary genre...).
- **EDA:** 70% movies / 30% TV shows. Top producers: US, India, UK. Additions peaked in 2019.
- **Recommender:** TF-IDF over weighted metadata (genres, director, country, type, maturity) with cosine similarity. Genre Jaccard@10 = 0.83 vs 0.09 for random picks.
- **Forecasting:** 4 models compared on a 12-month hold-out. A random forest on lag features was best (MAE ≈ 37 titles/month, vs 47 for the seasonal-naive baseline).
- **Classification:** predicts audience maturity group (Kids/Family/Teens/Adults). Random Forest: 61% accuracy, 0.59 macro-F1, vs a 46% majority baseline.
  (Movie vs TV Show was not used as the target: duration units and "TV ..." genre names make that prediction trivial.)
- **Dashboard:** a self-contained interactive HTML page with KPIs, Plotly charts, a live recommender search, and 6 business recommendations.
