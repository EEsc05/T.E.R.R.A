# T.E.R.R.A website v3 — visible forecast sequences

Terrain Evolution & Radar-based Research Analytics.

Extract the whole `TERRA_website_v3.zip` into a folder, keep `assets/` and `data/` with the HTML files, and open `index.html` in a current browser. No model server or Python installation is needed. Previous versions and the forecasting engine/checkpoints have not been overwritten.

## What Generate forecast maps does

1. Choose a country, region and satellite history. The newest map in the separately exported causal input window appears as the read-only starting input date. The browsing date does not change that input window.
2. Select a future target date and a map interval (7, 12 or 30 days). Generate runs the saved four-branch model locally for each interval and always includes the exact target date. For a 30-day target at 7-day intervals, it produces five maps at 7, 14, 21, 28 and 30 days.
3. The view moves to the map panel beside the globe and selects Forecast. Dated thumbnails show the real starting observation followed by the predicted maps. Play and the slider visit these populated dates only; a shared colour scale makes temporal changes comparable. Playback stops on the final target. With reduced motion enabled, the final map is shown immediately and playback remains available manually.

Each future date is predicted directly from the same six-slot window by default. It is not an interpolated image, a new radar measurement, a newly fitted model or an activity probability. The initial saved forecast can resemble a newly computed forecast for the same target because the checkpoint and inputs are identical. Repeating the same calculation should return the same values.

For experimental recursive forecasting, explicitly enable predicted inputs, select the final prediction, and choose Use this prediction in the next window. The next sequence uses that updated window. Observed/predicted/missing slot counts and generation depth remain labelled. Reset restores the latest genuine prepared observations. Sparse windows retain real endpoints and insert masked interior slots; they do not invent measurements. The original actual observation remains the reference for cumulative change. Feedback can compound errors; it does not create calibrated confidence.

The former timeline mixed observation-only dates with forecast dates. In Forecast mode those dates produced an empty map. v3 separates the layer timelines, brings results into view, and shows progress and validation errors both at the generator and beside the map. Observed and Error retain their appropriate data availability; nonexistent observations or error maps are not fabricated.

## Activity options

- Ground deformation summarizes the currently selected displacement/error map, including valid-cell count, signed range and spatial mean. These are map statistics, not hazard scores.
- Reported eruption opens the preserved retrospective Stage 2 case where available. It does not run a new activity model or use your newly generated maps. Its high false-positive rate and uncalibrated probability remain disclosed.
- Gas emission, magma movement, slope movement/landslides, land subsidence, fault movement and infrastructure movement explain potential applications and the additional evidence needed. No trained predictor, probability, live alert or engineering-safety judgment is supplied for these applications.

The main disaster selector still identifies unsupported hazard categories. General applicability of deformation measurements is not evidence that this trained volcanic-landscape collection generalizes to other hazards.

## Scientific provenance and limits

Numeric browser weights come unchanged from `outputs/nisar_aria_sakurajima_20261115/models/four_branch.joblib`. The browser implementation and weight asset are identical to v2. See `engine_README.md` and its verification JSON files for the numerical port and Python-checkpoint agreement. No retraining was performed.

Included data comprise 16 separate sensor/orbit histories, 15 volcanoes and 10 countries, from NISAR and Sentinel-1 ARIA GUNW. See `dataset.json`, `causal_input_provenance.json` and `activity_sources.json`. New forecast inputs use the causal history tensors; the full retrospective observation cube is only for browsing and matching actual targets. The current model uses signed 32×32 relative line-of-sight displacement maps in mm with masks and consistent geometry/grid metadata. The graphical site uses included histories, and does not read or process raw satellite products.

Numerical forecast horizons of 1–180 days are permitted. Saved reporting horizons are 18, 24, 30 and 36 days. Shorter/longer horizons, sparse inputs and feedback chains have not demonstrated calibrated skill. A sequence is a set of point forecasts, not independent evidence or a confidence interval. The saved reporting cohort has been reused in comparisons and is not a new independent test.

The change panel reports 100 × sum(abs(selected − original)) / sum(abs(original)) over common valid cells. The percentage depends on displacement reference and is withheld when mean absolute starting displacement is below 1 mm. Absolute and signed changes in mm are also shown. Starting observations show 0 mm temporal change. These values are not percentage land-height change, model accuracy or danger.

Official-resource buttons identify government sources, with public monitoring institutions labelled separately. Region-specific evacuation maps are included where verified; no live shelter opening/capacity or evacuation instruction is inferred. See `government_sources.md` for dated documents and verification limits. Creator names/bios remain blank pending supplied information.

## Checks

`browser_verification_v3.json` records actual generated sequences across all 16 histories, populated playback, common colour scales, original-date feedback provenance, sparse masks, independence from the retrospective browsing cube, visible date errors, all eight activity choices, mobile navigation and four pages at five widths. It checks that earlier version files and engine files match their pre-edit hashes.

`site_sha256.json` identifies packaged source/data assets; `package_manifest.json` lists archive entries. Runtime forecast-engine and model asset hashes match v2. Screenshots and reproducibility helpers remain in the repository version folder, outside the lightweight download.
