# Result tables (generated from reports/*.csv; do not edit by hand)

## 3.1 Model comparison (CV = 5-fold GroupKFold on 80 train trips; test = 10 student-test trips)

| model | family | config | cv_macro_f1_mean | cv_macro_f1_std | cv_top1_mean | test_top1_all | test_macro_f1 | test_log_loss | test_route_continuity | train_fit_top1 | train_time_s | inference_s_per_1k_candidates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| logreg | Linear | C=1.0, L2, scaled | 0.8755 | 0.0219 | 0.7481 | 0.8297 | 0.8912 | 0.1250 | 0.9099 | 0.7318 | 1.5297 | 0.0000 |
| gaussian_nb | Probabilistic | var_smoothing=1e-9 | 0.7582 | 0.0180 | 0.7309 | 0.7902 | 0.7661 | 0.5327 | 0.9048 | 0.7153 | 0.2962 | 0.0001 |
| decision_tree | Tree | max_depth=12, min_samples_leaf=50 | 0.8907 | 0.0205 | 0.7555 | 0.8400 | 0.9027 | 0.1018 | 0.9192 | 0.7522 | 10.9159 | 0.0000 |
| random_forest | Ensemble (bagging) | n_estimators=60, max_depth=14, min_samples_leaf=30 | 0.8937 | 0.0194 | 0.7590 | 0.8400 | 0.9058 | 0.0958 | 0.9193 | 0.7541 | 32.1122 | 0.0006 |
| hist_gb | Boosting | max_iter=200, lr=0.1, max_leaf_nodes=31 | 0.8939 | 0.0204 | 0.7583 | 0.8370 | 0.9028 | 0.0966 | 0.9181 | 0.7556 | 4.5407 | 0.0006 |

## 4. Bayesian optimisation

Trials: 12 (budget 600 s), sampler Optuna TPE, objective: mean candidate-level macro-F1, CV: GroupKFold(5) by trip, 80 train trips

Best params: `{"learning_rate": 0.021146536167031597, "max_iter": 400, "max_leaf_nodes": 109, "min_samples_leaf": 39, "l2_regularization": 0.0008111941985431928}`


| model | cv_macro_f1 | cv_top1_all | cv_log_loss | test_top1_all | test_macro_f1 | test_log_loss | test_route_continuity | train_time_s | inference_s_per_1k_candidates |
|---|---|---|---|---|---|---|---|---|---|
| untuned | 0.8939 | 0.7416 | 0.1157 | 0.8370 | 0.9028 | 0.0966 | 0.9181 | 4.0654 | 0.0006 |
| tuned_bayes | 0.8962 | 0.7437 | 0.1127 | 0.8398 | 0.9051 | 0.0942 | 0.9210 | 24.7814 | 0.0039 |

### All trials

| number | value | params_l2_regularization | params_learning_rate | params_max_iter | params_max_leaf_nodes | params_min_samples_leaf | duration |
|---|---|---|---|---|---|---|---|
| 0 | 0.8936 | 0.0001 | 0.1000 | 200 | 31 | 20 | 0 days 00:00:21.835868 |
| 1 | 0.8955 | 0.0006 | 0.0551 | 400 | 97 | 136 | 0 days 00:01:44.903939 |
| 2 | 0.8943 | 0.3470 | 0.0305 | 100 | 112 | 137 | 0 days 00:00:32.009797 |
| 3 | 0.8962 | 0.0008 | 0.0211 | 400 | 109 | 39 | 0 days 00:02:02.364709 |
| 4 | 0.8954 | 0.0029 | 0.0329 | 200 | 74 | 79 | 0 days 00:00:54.359962 |
| 5 | 0.8947 | 0.0191 | 0.1049 | 100 | 48 | 64 | 0 days 00:00:23.449267 |
| 6 | 0.8946 | 0.0002 | 0.1677 | 150 | 73 | 133 | 0 days 00:00:20.558302 |
| 7 | 0.8952 | 6.7322 | 0.1036 | 150 | 22 | 424 | 0 days 00:00:32.601175 |
| 8 | 0.8946 | 0.0159 | 0.1786 | 200 | 26 | 180 | 0 days 00:00:24.952644 |
| 9 | 0.8948 | 0.0020 | 0.0278 | 250 | 18 | 373 | 0 days 00:00:51.370445 |
| 10 | 0.8962 | 0.0002 | 0.0258 | 350 | 116 | 20 | 0 days 00:01:39.855274 |
| 11 | 0.8961 | 0.0023 | 0.0240 | 350 | 113 | 28 | 0 days 00:01:40.047530 |

## 5. HMM vs classifier-only

| partition | method | top1_all | top1_covered | route_continuity | decode_s |
|---|---|---|---|---|---|
| train_oof_80_trips | classifier_only | 0.7437 | 0.8380 | 0.9243 | nan |
| train_oof_80_trips | classifier+HMM | 0.7899 | 0.8901 | 0.9810 | 2.3590 |
| student_test_10_trips | classifier_only | 0.8398 | 0.8413 | 0.9210 | nan |
| student_test_10_trips | classifier+HMM | 0.8740 | 0.8755 | 0.9804 | 0.2444 |

### Decoder grid (top 8 by top1_all, tuning trips only)

| beta | lam | w | top1_all | top1_covered | route_continuity |
|---|---|---|---|---|---|
| 10.0000 | 8.0000 | 0.5000 | 0.8056 | 0.8670 | 0.9760 |
| 10.0000 | 16.0000 | 1.0000 | 0.8055 | 0.8669 | 0.9746 |
| 20.0000 | 8.0000 | 0.5000 | 0.8055 | 0.8669 | 0.9746 |
| 20.0000 | 8.0000 | 0.3000 | 0.8055 | 0.8669 | 0.9762 |
| 10.0000 | 4.0000 | 1.0000 | 0.8054 | 0.8668 | 0.9630 |
| 20.0000 | 4.0000 | 0.5000 | 0.8053 | 0.8667 | 0.9727 |
| 10.0000 | 8.0000 | 1.0000 | 0.8053 | 0.8667 | 0.9727 |
| 6.0000 | 16.0000 | 1.0000 | 0.8052 | 0.8666 | 0.9758 |

## Candidate miss rate, train

median 0.0000, max 0.8849, trips above 5%: 5

## Candidate miss rate, student_test

median 0.0000, max 0.0071, trips above 5%: 0
