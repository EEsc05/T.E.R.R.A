# Browser four-branch inference

Load `assets/model-four-branch.js`, then `forecast-engine.js`. The API is:

```js
const result = await TerraFourBranch.predict({
  maps,           // six or more flat1024 or32×32 signed LOS displacement maps
  dates,          // matching strictly increasing UTC dates; last six slots used
  valid_masks,    // matching boolean masks; omit only to derive from finite values
  metadata: {
    units: 'mm',
    spacing_km: [rowSpacing, columnSpacing],
    grid_convention: 'aoi_block_mean32',
    geometry_available: false
  },
  horizon_days: 30 //1–180days beyond the latest supplied map
});
```

Returned `prediction_mm`, `change_mm` and `support_mask` are flat1024 arrays. Unsupported cells are null in numeric outputs. Origin/target, checkpoint SHA256, source model run and actual horizon-bin fusion weights are returned. `confidence_calibrated` is always false. This API does not invent a probability or interval.

This is the current trained model from `outputs/nisar_aria_sakurajima_20261115/models/four_branch.joblib`, exported read-only; it is not a newly trained model. The browser implementation reproduces the temporal EMA, robust quality smoothing, block16 spatial basis with regularized linear temporal forecast, and StandardScaler/RBF-SVR context model. It performs no model fitting. Inputs and branch arithmetic preserve relevant float32 rounding; double precision is used for eigendecomposition, spatial projection and SVR kernel summation.

The trained context fusion coefficient is zero in the first horizon bin and approximately6.5×10^-17 in the second. For those bins only, the expensive SVR branch is skipped when its fusion coefficient has magnitude below10^-14. The numerical verification below includes both those bins and the long-horizon bin with25% context weight. The API identifies whether context was evaluated. This optimization does not alter the stored weights.

`engine_reference.py` exports real Python-checkpoint reference cases; `engine_verify.py` evaluates them in Chromium. Sixteen cases span Dabbahu, El Misti, Kilauea and Sakurajima, at7/30/45/120-day horizons and varied natural masks. Maximum absolute point-prediction difference is approximately**0.00001487mm**, below the specified0.001mm tolerance. `engine_verification.json` is authoritative. The initial failed numerical trial is retained as `engine_verification_failed_initial.json`; its quality-branch MAD factor was corrected before final verification.

`engine_extended_reference.py` / `engine_extended_verify.py` verify explicit missing-history slots and three recursive forecast-feedback steps. All five additional cases pass the same tolerance, with maximum error approximately0.000000833mm. The Python source emits its expected all-NaN quantile warning when the first history slot is entirely missing; both implementations use the same scale fallback.

These checks establish numerical agreement with the saved engine, not accuracy of an arbitrary new forecast. Recursive predictions remain predictions and should be tracked separately from observations. Reusing model outputs as inputs can compound bias, reduce independent observational support and move the model outside its evaluated regime. It does not yield a validated formula for lowering a confidence percentage. The UI owns recursion provenance and qualitative reliability notices.

Known geometry must include a verified `los_vector` of three finite values. The engine does not align grids, unwrap phase, convert raw radar intensity to displacement, infer unknown geometry or validate upstream acquisition/publication times. A1–180-day permitted numerical horizon is not evidence of evaluated skill at every such horizon.

The assets work without a model server or a Python installation in the viewer's browser. The SVR computation yields periodically to the event loop to keep the interface responsive. Export/test helpers require the existing local Python environments; they are reproducibility tools, not browser runtime dependencies. No checkpoint or current src file was modified.

`engine_sparse_recursive_verification.json` adds seven targeted checks: four real maps plus two interior missing slots, five real maps plus one interior missing slot, each at 30 and 120 days; and three 30-day recursive steps. All pass, with maximum error 0.000000833 mm. `engine_recursive_browser_feedback_verification.json` separately confirms the actual JavaScript outputs can be fed into successive JavaScript histories and stay within the same tolerance against the independent Python recursive trajectory (third-step maximum error 0.000000143 mm). No additional engine changes were needed.

`engine_contract_verification.json` confirms measured zero maps remain zero, four invalid input contracts are rejected, and calibrated confidence is not claimed.
