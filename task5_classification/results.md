# Task 5 - Classification results

**Target:** audience maturity group (Kids / Family / Teens / Adults), 8704 titles, 70 features, 80/20 stratified split, 5-fold CV.

|                         |   CV macro-F1 |   Test accuracy |   Test macro-F1 |
|:------------------------|--------------:|----------------:|----------------:|
| Random Forest           |         0.63  |           0.614 |           0.594 |
| Hist Gradient Boosting  |         0.607 |           0.608 |           0.59  |
| Logistic Regression     |         0.608 |           0.594 |           0.589 |
| Decision Tree           |         0.572 |           0.58  |           0.576 |
| Majority-class baseline |       nan     |           0.46  |           0.158 |

**Best model:** Random Forest

```
              precision    recall  f1-score   support

        Kids       0.78      0.74      0.76       181
      Family       0.41      0.38      0.40       230
       Teens       0.52      0.51      0.51       529
      Adults       0.69      0.72      0.71       801

    accuracy                           0.61      1741
   macro avg       0.60      0.59      0.59      1741
weighted avg       0.61      0.61      0.61      1741

```

## Top features (Random Forest)

|                                 |      0 |
|:--------------------------------|-------:|
| duration_min                    | 0.1321 |
| genre: Kids' TV                 | 0.116  |
| genre: Children & Family Movies | 0.0993 |
| release_year                    | 0.0771 |
| years_to_netflix                | 0.0651 |
| year_added                      | 0.0518 |
| n_genres                        | 0.048  |
| seasons                         | 0.0226 |
| genre: International TV Shows   | 0.0193 |
| genre: International Movies     | 0.0187 |
| genre: Dramas                   | 0.0171 |
| country: India                  | 0.0162 |
| country: United States          | 0.0161 |
| genre: Documentaries            | 0.0158 |
| genre: Independent Movies       | 0.0154 |
