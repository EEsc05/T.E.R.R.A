# NISAR training comparison

Trained from scratch on 53 examples across 6 volcanoes. Reported on 12 reporting windows excluded from all fitting and selection.

| Model | MAE (mm) | RMSE (mm) | MAE skill vs persistence |
| --- | ---: | ---: | ---: |
| persistence | 23.9511 | 47.5035 | +0.0% |
| ema30 | 23.8441 | 42.2345 | +0.4% |
| smoothing | 23.8171 | 41.2270 | +0.6% |
| robust | 24.4130 | 41.3746 | -1.9% |
| four_branch | 23.7383 | 41.8327 | +0.9% |
| cvae | 25.9449 | 45.1134 | -8.3% |

Positive skill means lower MAE than persistence. Scores weight volcanoes equally and windows equally within each volcano. Identical scoring masks are used for all models.

CVAE duration selected from training-only folds: 43 epochs. Ensemble weights use training-only out-of-fold branch predictions.

Persistence, EMA30, smoothing and robust low-rank are local baselines with no global neural weights. Four-branch and CVAE learn new parameters. No hackathon checkpoints were loaded.

Observed target horizons in this run: [24.0] days. Reporting volcanoes: ['dabbahu', 'misti', 'popocatepetl', 'sabancaya', 'ubinas']. CVAE uncertainty is uncalibrated. A reused reporting set is not a new independent test.

CVAE empirical CRPS: 21.8994 mm; 90% interval coverage: 29.8%; mean interval width: 30.4765 mm.

**Retrospective comparison:** Some training targets occur after the reporting forecast origins. The latest training target is 2026-10-21T00:00:00.000000000. These scores are not a strict global forward-time test and cannot establish prospective improvement.

Training counts by physical volcano: {'dabbahu': 10, 'misti': 10, 'mount_asama': 3, 'popocatepetl': 11, 'sabancaya': 10, 'ubinas': 9}. Mount Asama viewing directions remain separate histories but share one physical-volcano grouping. All Japan examples are training data; no held-out Japan performance is reported.

## Comparison with the previous Peru and Japan model set

Reference saved models were rescored on the current reporting windows and identical masks. No fitting was performed on the reporting rows.

| Model | Original MAE | Combined MAE | MAE change | Original RMSE | Combined RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| persistence | 23.9511 | 23.9511 | +0.0000 | 47.5035 | 47.5035 |
| ema30 | 23.8441 | 23.8441 | +0.0000 | 42.2345 | 42.2345 |
| smoothing | 23.8171 | 23.8171 | +0.0000 | 41.2270 | 41.2270 |
| robust | 24.4130 | 24.4130 | +0.0000 | 41.3746 | 41.3746 |
| four_branch | 23.8103 | 23.7383 | -0.0719 | 42.4578 | 41.8327 |
| cvae | 30.1492 | 25.9449 | -4.2043 | 50.4295 | 45.1134 |

## Input provenance and limitations

The previous Peru and Mount Asama inputs were combined with newly prepared
Popocatepetl (Mexico) and Dabbahu (Ethiopia) inputs. Ubinas was rebuilt from
existing subsets excluding two BETA pairs. Original data and model artifacts
remain unchanged, verified against their saved SHA256 hashes. See
`input_provenance.json` and `artifact_sha256.json`.

Some Peru reporting periods were evaluated previously; the enlarged reporting
set is not wholly independent. Adding countries and cleaning Ubinas both changed
the experiment, so paired score differences cannot isolate the effect of extra
data alone. The CVAE's nominal 90% interval covers only 29.8% of observed values;
its uncertainty intervals are not reliable probability statements.

Six targeted tests passed. Independent recalculation from saved predictions
reproduced MAE/RMSE for all current and reference models and verified identical
scoring masks. The four local baseline predictions match the reference exactly.
