# NISAR training comparison

Trained from scratch on 33 examples across 4 volcanoes. Reported on the 6 original Peru reporting windows excluded from all fitting and selection.

| Model | MAE (mm) | RMSE (mm) | MAE skill vs persistence |
| --- | ---: | ---: | ---: |
| persistence | 23.9508 | 55.2629 | +0.0% |
| ema30 | 26.1315 | 50.3718 | -9.1% |
| smoothing | 25.9614 | 48.9106 | -8.4% |
| robust | 27.1197 | 49.3599 | -13.2% |
| four_branch | 25.8739 | 50.5853 | -8.0% |
| cvae | 32.2768 | 57.8588 | -34.8% |

Positive skill means lower MAE than persistence. Scores weight volcanoes equally and windows equally within each volcano. Identical scoring masks are used for all models.

CVAE duration selected from training-only folds: 44 epochs. Ensemble weights use training-only out-of-fold branch predictions.

Persistence, EMA30, smoothing and robust low-rank are local baselines with no global neural weights. Four-branch and CVAE learn new parameters. No hackathon checkpoints were loaded.

Targets are actual 24-day observations in these archives. The reporting windows cover the original Peru volcanoes. CVAE uncertainty is uncalibrated. A reused reporting set is not a new independent test.

CVAE empirical CRPS: 29.6435 mm; 90% interval coverage: 9.4%; mean interval width: 15.8073 mm.

Training counts by physical volcano: {'misti': 10, 'mount_asama': 3, 'sabancaya': 10, 'ubinas': 10}. Mount Asama viewing directions remain separate histories but share one physical-volcano grouping. All Japan examples are training data; no held-out Japan performance is reported.

## Comparison with the original model set

The same previously evaluated Peru windows and scoring masks are reused. No reporting targets were used in fitting or internal configuration selection.

| Model | Original MAE | Combined MAE | MAE change | Original RMSE | Combined RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| persistence | 23.9508 | 23.9508 | +0.0000 | 55.2629 | 55.2629 |
| ema30 | 26.1315 | 26.1315 | +0.0000 | 50.3718 | 50.3718 |
| smoothing | 25.9614 | 25.9614 | +0.0000 | 48.9106 | 48.9106 |
| robust | 27.1197 | 27.1197 | +0.0000 | 49.3599 | 49.3599 |
| four_branch | 26.1375 | 25.8739 | -0.2636 | 50.4259 | 50.5853 |
| cvae | 34.0956 | 32.2768 | -1.8188 | 60.2583 | 57.8588 |

**Temporal limit:** Japan training targets extend through 2026-10-21, later than the Peru reporting forecast origins. This reused Peru comparison is retrospective, not a strict forward-time test. All Japan examples are used for training; no held-out Japan performance is measured.
