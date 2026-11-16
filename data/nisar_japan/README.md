# Mount Asama, Japan

Use `mount_asama_ascending/` and `mount_asama_descending/` for the summit-centered datasets.
Both carry `volcano_id=mount_asama`, `volcano_name=Mount Asama (Asamayama)`, and `country=Japan`.
The histories remain separate LOS stacks: 139/A/20 and 46/D/70.

Each contains `history.npz`, `series.npz`, `examples.npz`, metadata, the source catalogue,
and cached native AOI subsets. Ascending has 9 dates and 1 supervised example;
descending has 10 dates and 2 examples. Targets are actual 24-day observations.
No standalone train/validation split is available. History coverage is sparse,
approximately 7% ascending and 18% descending; validity masks must be respected.

`mount_asama_validation_report.json` records actual counts, masks and input validation.
`mount_asama_sites.json` and `site_definitions.json` define the same final summit AOIs.
`exploratory_pixel_windows.json` and the older `japan_*` survey directories are
exploratory raster-window references, not the summit-centered Mount Asama inputs.

Reproduce from the project root:

```bash
python -m src prepare-data --sites-config data/nisar_japan/mount_asama_sites.json \
  --output-dir data/nisar_japan --max-baseline-days 36 \
  --min-pair-valid-fraction 0.05 --token-file src/token.txt
```

The 5% pair-retention threshold preserves partial observations. Per-pixel coherence
>=0.4 and block-valid fraction >=0.5 remain unchanged. No observations are filled
in to replace missing measurements. Existing Peru datasets/models are unchanged.
