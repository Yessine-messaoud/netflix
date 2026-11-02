# Task 4 - Trend prediction results

Hold-out test: last 12 months (Sep 2020 to Aug 2021).

|                               |   MAE |   RMSE |   MAPE % |
|:------------------------------|------:|-------:|---------:|
| Random forest (lag features)  |  36.8 |   45   |     22.1 |
| Quadratic trend + seasonality |  40.3 |   48.6 |     27.4 |
| Seasonal naive (baseline)     |  47.2 |   57   |     28.1 |
| Linear trend + seasonality    |  57.8 |   65.8 |     41.1 |

**Best model:** Random forest (lag features)

## Forecast (titles added per calendar year)

|      |   forecast |
|-----:|-----------:|
| 2021 |        637 |
| 2022 |       1845 |
| 2023 |       1215 |

## Interpretation

- Monthly additions grew fast between 2016 and 2019, then levelled off and fell slightly in 2020-2021.
- The TV-show share of additions fell from 41% in 2016 to 34% in 2021 (peak 41% in 2016).
- The forecast is 1847 titles over the next 12 months, against 1973 in the last 12 observed (-6%).
- With only about 5 years of monthly history and a clear break around 2020, the forecast is a rough guide: treat it as a range, not an exact number.
