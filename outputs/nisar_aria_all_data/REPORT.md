# NISAR training comparison

Trained from scratch on 254 examples across 14 volcanoes. Reported on 62 reporting windows excluded from all fitting and selection.

| Model | MAE (mm) | RMSE (mm) | MAE skill vs persistence |
| --- | ---: | ---: | ---: |
| persistence | 13.8868 | 31.5808 | +0.0% |
| ema30 | 13.2082 | 27.9331 | +4.9% |
| smoothing | 13.1869 | 27.2983 | +5.0% |
| robust | 13.4824 | 27.4017 | +2.9% |
| four_branch | 13.1718 | 27.7821 | +5.1% |
| cvae | 14.0221 | 28.9737 | -1.0% |

Positive skill means lower MAE than persistence. Scores weight volcanoes equally and windows equally within each volcano. Identical scoring masks are used for all models.

CVAE duration selected from training-only folds: 27 epochs. Ensemble weights use training-only out-of-fold branch predictions.

Persistence, EMA30, smoothing and robust low-rank are local baselines with no global neural weights. Four-branch and CVAE learn new parameters. No hackathon checkpoints were loaded.

Observed target horizons in this run: [18.0, 24.0, 30.0, 36.0] days. Reporting volcanoes: ['coropuna', 'cotopaxi', 'dabbahu', 'erta_ale', 'kilauea', 'misti', 'popocatepetl', 'reykjanes', 'ruiz_nevado_del', 'sabancaya', 'taupo', 'ubinas']. CVAE uncertainty is uncalibrated. A reused reporting set is not a new independent test.

CVAE empirical CRPS: 11.6266 mm; 90% interval coverage: 48.8%; mean interval width: 20.2523 mm.

**Retrospective comparison:** Some training targets occur after the reporting forecast origins. The latest training target is 2026-10-21T00:00:00.000000000. These scores are not a strict global forward-time test and cannot establish prospective improvement.

Training counts by physical volcano: {'ambrym': 3, 'coropuna': 36, 'cotopaxi': 34, 'dabbahu': 10, 'erta_ale': 18, 'kilauea': 30, 'misti': 10, 'mount_asama': 3, 'popocatepetl': 11, 'reykjanes': 32, 'ruiz_nevado_del': 36, 'sabancaya': 10, 'taupo': 12, 'ubinas': 9}. Training-only volcanoes without a reporting score: ['ambrym', 'mount_asama']. Different viewing directions and sensors remain separate histories and share their physical-volcano grouping.

## Comparison with the previous model set

Reference saved models were rescored on the current reporting windows and identical masks. No fitting was performed on the reporting rows.

| Model | Previous MAE | New MAE | MAE change | Previous RMSE | New RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| persistence | 13.8868 | 13.8868 | +0.0000 | 31.5808 | 31.5808 |
| ema30 | 13.2082 | 13.2082 | +0.0000 | 27.9331 | 27.9331 |
| smoothing | 13.1869 | 13.1869 | +0.0000 | 27.2983 | 27.2983 |
| robust | 13.4824 | 13.4824 | +0.0000 | 27.4017 | 27.4017 |
| four_branch | 13.3420 | 13.1718 | -0.1702 | 27.7645 | 27.7821 |
| cvae | 14.7428 | 14.0221 | -0.7207 | 30.0092 | 28.9737 |

## Data intake and dashboard

The new pool contains 254 training examples from 14 physical volcanoes in 10
countries, represented by 15 independent sensor/orbit histories. Reporting uses
62 windows across 12 volcanoes. All three Mount Asama examples and all three
Ambrym examples are training only. The ARIA batch contributes 198 training
examples with reporting partitions, plus the three Ambrym examples; the previous
NISAR pool contributes 53 training examples. The cleaned Ubinas inputs exclude
the two earlier BETA pairs. All previous model artifacts remain unchanged.

The ARIA batch processes up to 60 connected acquisitions per selected stack,
not the whole archive. Manam and Lastarria failed the usable-history/coverage
checks. Ambrym uses a smaller 20 km AOI with unchanged quality thresholds.
See `../../data/aria/REPORT.md` for conversion details and exclusions, and
`input_manifest.json` for all inputs and hashes. Sensor histories were kept
separate and were not spliced into one time series.

The standalone dashboard is `../volcano_watch/index.html`, with a detailed page
at `../volcano_watch/details.html`. It includes locally bundled globe geometry,
country/region/date controls, model selection, real observed/forecast/error maps,
CVAE spread, pixel histories and the reporting benchmark. It makes no external
network requests. Keep its accompanying assets and data folders together.

## Final verification

All 41 tests passed. Independent recalculation from saved prediction arrays
reproduced MAE/RMSE for all six new models and all six reference summaries.
Current/reference scoring masks match; the four local baseline predictions are
unchanged. Browser checks covered all 15 histories, globe rotation and country
selection, model/layer controls, paired error maps and responsive desktop/mobile
layouts, with no script errors or external requests. Exported data hashes and
all six model hashes were verified. See `verification.json` and the dashboard's
`browser_verification.json`.

The internal folds omit redundant four-branch training diagnostics. A regression
test verified identical fitted support vectors and predictions, and the first
real-data CVAE fold exactly matched the earlier calculation. Model fitting,
selection targets and the reporting protocol are unchanged by this optimization.
