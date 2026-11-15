# NISAR training comparison

Trained from scratch on 30 examples across 3 volcanoes. Reported on 6 later windows excluded from all fitting and selection.

| Model | MAE (mm) | RMSE (mm) | MAE skill vs persistence |
| --- | ---: | ---: | ---: |
| persistence | 23.9508 | 55.2629 | +0.0% |
| ema30 | 26.1315 | 50.3718 | -9.1% |
| smoothing | 25.9614 | 48.9106 | -8.4% |
| robust | 27.1197 | 49.3599 | -13.2% |
| four_branch | 26.1375 | 50.4259 | -9.1% |
| cvae | 34.0956 | 60.2583 | -42.4% |

Positive skill means lower MAE than persistence. Scores weight volcanoes equally and windows equally within each volcano. Identical scoring masks are used for all models.

CVAE duration selected from training-only folds: 49 epochs. Ensemble weights use training-only out-of-fold branch predictions.

Persistence, EMA30, smoothing and robust low-rank are local baselines with no global neural weights. Four-branch and CVAE learn new parameters. No hackathon checkpoints were loaded.

Targets are actual 24-day observations in these archives. Six windows and three volcanoes provide a small temporal comparison, not an independent performance guarantee. CVAE uncertainty is uncalibrated.

CVAE empirical CRPS: 31.4953 mm; 90% interval coverage: 9.4%; mean interval width: 15.6510 mm.
