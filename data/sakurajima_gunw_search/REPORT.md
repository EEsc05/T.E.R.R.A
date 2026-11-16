# Sakurajima GUNW search and sample-quality screening

Catalogue query timestamp: 2026-11-15T15:47:40.297773+00:00. Quality screening completed: 2026-11-15T15:55:36.297087+00:00.

## Location and scope

Sakurajima (Aira), Japan; centre 31.5772° N, 130.6589° E, from the [Smithsonian Global Volcanism Program](https://volcano.si.edu/volcano.cfm?vn=282080). The AOI is a 15 km × 15 km square centred on that coordinate. It includes ocean and is not a land-only volcano mask.

Searched the ASF catalogue for NISAR GUNW from 2025-01-01 and Sentinel-1 ARIA GUNW_STD from 2014-01-01, both through 2026-11-15. Each query had a 5,000-result limit; neither was truncated.

Found 41 NISAR HH products (40 PROVISIONAL_V1 and one NISAR_UR_L2 record), representing 24 distinct acquisition pairs in three separate viewing geometries. Reprocessed products are deduplicated by acquisition endpoints, retaining the latest processing date/filename. There were no ARIA GUNW matches in this query; this is not a claim that Sentinel-1 raw imagery is unavailable.

## Candidate stacks

| Stack (track/direction/frame) | Connected dates | Unique pairs | Observation span | Sample valid-grid fractions | Pass all three samples? |
|---|---:|---:|---|---|---|
| [125, 'A', 17] | 9 | 8 | 2026-06-22 – 2026-10-20 | 34.9%, 38.6%, 42.6% | True |
| [24, 'A', 18] | 9 | 8 | 2026-06-27 – 2026-10-13 | 1.1%, 42.4%, 45.4% | False |
| [3, 'D', 73] | 9 | 8 | 2026-06-26 – 2026-10-12 | 0.0%, 0.9%, 44.9% | False |

## Quality method and interpretation

Used the current engine readers without changing thresholds: HH unwrapped phase converted to line-of-sight displacement in mm, finite phase/coherence, coherence ≥0.4, connected-component label >0, ≥50% valid native samples per output block, ≥90% geometric AOI coverage, and ≥20% valid blocks for pair retention. The output is 32×32, approximately 468.75 m per cell over this AOI. Exact native spacing and mean coherence over valid output blocks are recorded per sample in quality_probes.json.

Screened the earliest, middle and latest pair in each deduplicated largest connected acquisition component. Baselines are positive and ≤36 days; BETA collections are excluded. Passing this screen establishes usable masked pixels in three sampled interferograms. It does not validate every pair, phase unwrapping, atmospheric correction quality, summit deformation sensitivity, or a complete forecasting window. Full-stack pixel screening and temporal-network preparation remain to be done before training.

## Recommendation and training limitations

1 of the three stacks pass the three-sample coverage screen. The highest minimum sampled valid fraction is in sakurajima_125_a17; this is a coverage ranking, not a deformation-accuracy ranking.

Each stack has nine connected acquisition dates. Under the existing six-history, nominal 30-day forecast (±12-day target tolerance, excluding the zero datum), catalogue connectivity suggests only one candidate supervised window per stack. This has not been confirmed after full pixel-quality filtering. These short histories cannot support a convincing independent Sakurajima train/validation/test assessment. If added to the global corpus, group all three stacks under physical volcano_id=sakurajima, preserve separate orbit/frame histories and line-of-sight geometry, and apply the existing temporal purge rules. Do not concatenate ascending and descending maps into one displacement series.

No training was run, no checkpoint was modified and no existing training dataset was replaced. Only this dedicated search directory contains the new catalogue and compact AOI raster subsets.

## Sample provenance

### sakurajima_125_a17

- [NISAR_L2_PR_GUNW_023_125_A_017_024_4000_SH_20260622T203702_20260622T203722_20260704T203702_20260704T203721_P05023_N_P_J_001.h5](https://nisar.asf.earthdatacloud.nasa.gov/NISAR/NISAR_L2_GUNW_PROVISIONAL_V1/NISAR_L2_PR_GUNW_023_125_A_017_024_4000_SH_20260622T203702_20260622T203722_20260704T203702_20260704T203721_P05023_N_P_J_001/NISAR_L2_PR_GUNW_023_125_A_017_024_4000_SH_20260622T203702_20260622T203722_20260704T203702_20260704T203721_P05023_N_P_J_001.h5): 357/1024 valid blocks (34.86%); native spacing [80.0, -80.0] m; valid-block mean coherence 0.6405759510955541.
- [NISAR_L2_PR_GUNW_029_125_A_017_030_4000_SH_20260902T203659_20260902T203719_20260914T203659_20260914T203719_P05023_N_P_J_001.h5](https://nisar.asf.earthdatacloud.nasa.gov/NISAR/NISAR_L2_GUNW_PROVISIONAL_V1/NISAR_L2_PR_GUNW_029_125_A_017_030_4000_SH_20260902T203659_20260902T203719_20260914T203659_20260914T203719_P05023_N_P_J_001/NISAR_L2_PR_GUNW_029_125_A_017_030_4000_SH_20260902T203659_20260902T203719_20260914T203659_20260914T203719_P05023_N_P_J_001.h5): 395/1024 valid blocks (38.57%); native spacing [80.0, -80.0] m; valid-block mean coherence 0.5990051231999308.
- [NISAR_L2_PR_GUNW_032_125_A_017_033_2000_SH_20261008T203644_20261008T203719_20261020T203644_20261020T203719_P05023_N_F_J_001.h5](https://nisar.asf.earthdatacloud.nasa.gov/NISAR/NISAR_L2_GUNW_PROVISIONAL_V1/NISAR_L2_PR_GUNW_032_125_A_017_033_2000_SH_20261008T203644_20261008T203719_20261020T203644_20261020T203719_P05023_N_F_J_001/NISAR_L2_PR_GUNW_032_125_A_017_033_2000_SH_20261008T203644_20261008T203719_20261020T203644_20261020T203719_P05023_N_F_J_001.h5): 436/1024 valid blocks (42.58%); native spacing [80.0, -80.0] m; valid-block mean coherence 0.6399210925075351.

### sakurajima_24_a18

- [NISAR_L2_PR_GUNW_032_024_A_018_033_4000_SH_20261001T202858_20261001T202933_20261013T202858_20261013T202933_P05023_N_F_J_001.h5](https://nisar.asf.earthdatacloud.nasa.gov/NISAR/NISAR_L2_GUNW_PROVISIONAL_V1/NISAR_L2_PR_GUNW_032_024_A_018_033_4000_SH_20261001T202858_20261001T202933_20261013T202858_20261013T202933_P05023_N_F_J_001/NISAR_L2_PR_GUNW_032_024_A_018_033_4000_SH_20261001T202858_20261001T202933_20261013T202858_20261013T202933_P05023_N_F_J_001.h5): 465/1024 valid blocks (45.41%); native spacing [80.0, -80.0] m; valid-block mean coherence 0.7015350873238141.
- [NISAR_L2_PR_GUNW_024_024_A_018_025_4000_SH_20260627T202901_20260627T202936_20260709T202900_20260709T202935_P05023_N_F_J_001.h5](https://nisar.asf.earthdatacloud.nasa.gov/NISAR/NISAR_L2_GUNW_PROVISIONAL_V1/NISAR_L2_PR_GUNW_024_024_A_018_025_4000_SH_20260627T202901_20260627T202936_20260709T202900_20260709T202935_P05023_N_F_J_001/NISAR_L2_PR_GUNW_024_024_A_018_025_4000_SH_20260627T202901_20260627T202936_20260709T202900_20260709T202935_P05023_N_F_J_001.h5): 434/1024 valid blocks (42.38%); native spacing [80.0, -80.0] m; valid-block mean coherence 0.6096154324650447.
- [NISAR_L2_PR_GUNW_029_024_A_018_030_4000_SH_20260826T202858_20260826T202933_20260907T202858_20260907T202933_P05023_N_F_J_001.h5](https://nisar.asf.earthdatacloud.nasa.gov/NISAR/NISAR_L2_GUNW_PROVISIONAL_V1/NISAR_L2_PR_GUNW_029_024_A_018_030_4000_SH_20260826T202858_20260826T202933_20260907T202858_20260907T202933_P05023_N_F_J_001/NISAR_L2_PR_GUNW_029_024_A_018_030_4000_SH_20260826T202858_20260826T202933_20260907T202858_20260907T202933_P05023_N_F_J_001.h5): 11/1024 valid blocks (1.07%); native spacing [80.0, -80.0] m; valid-block mean coherence 0.6165228825271698.

### sakurajima_3_d73

- [NISAR_L2_PR_GUNW_029_003_D_073_030_2000_SH_20260825T100250_20260825T100325_20260906T100250_20260906T100325_P05023_N_F_J_002.h5](https://nisar.asf.earthdatacloud.nasa.gov/NISAR/NISAR_L2_GUNW_PROVISIONAL_V1/NISAR_L2_PR_GUNW_029_003_D_073_030_2000_SH_20260825T100250_20260825T100325_20260906T100250_20260906T100325_P05023_N_F_J_002/NISAR_L2_PR_GUNW_029_003_D_073_030_2000_SH_20260825T100250_20260825T100325_20260906T100250_20260906T100325_P05023_N_F_J_002.h5): 9/1024 valid blocks (0.88%); native spacing [80.0, -80.0] m; valid-block mean coherence 0.5729459565810047.
- [NISAR_L2_PR_GUNW_032_003_D_073_033_2000_SH_20260930T100250_20260930T100325_20261012T100250_20261012T100325_P05023_N_F_J_001.h5](https://nisar.asf.earthdatacloud.nasa.gov/NISAR/NISAR_L2_GUNW_PROVISIONAL_V1/NISAR_L2_PR_GUNW_032_003_D_073_033_2000_SH_20260930T100250_20260930T100325_20261012T100250_20261012T100325_P05023_N_F_J_001/NISAR_L2_PR_GUNW_032_003_D_073_033_2000_SH_20260930T100250_20260930T100325_20261012T100250_20261012T100325_P05023_N_F_J_001.h5): 460/1024 valid blocks (44.92%); native spacing [80.0, -80.0] m; valid-block mean coherence 0.694680037691052.
- [NISAR_L2_PR_GUNW_024_003_D_073_025_2000_SH_20260626T100253_20260626T100327_20260708T100252_20260708T100327_P05023_N_F_J_001.h5](https://nisar.asf.earthdatacloud.nasa.gov/NISAR/NISAR_L2_GUNW_PROVISIONAL_V1/NISAR_L2_PR_GUNW_024_003_D_073_025_2000_SH_20260626T100253_20260626T100327_20260708T100252_20260708T100327_P05023_N_F_J_001/NISAR_L2_PR_GUNW_024_003_D_073_025_2000_SH_20260626T100253_20260626T100327_20260708T100252_20260708T100327_P05023_N_F_J_001.h5): 0/1024 valid blocks (0.00%); native spacing [80.0, -80.0] m; valid-block mean coherence None.

## Reproducibility files

- sites_config.json: geographic AOI and source coordinate.
- search_summary.json, nisar_survey.json, aria_survey.json: query timestamp and connectivity counts.
- catalogues/sakurajima.json and aria_catalogues/sakurajima.json: catalogue records.
- product_links.json: all 41 catalogue product download URLs and collection/processing metadata.
- probe_sites.json: three stack configurations used for screening.
- quality_probes.json: exact sample IDs, quality measurements, source URLs and bytes fetched.
- rasters/<stack>/subsets/*.h5: compact AOI derivatives, retaining original source product/URL attributes.

To rerun the raster screen from the repository root (Earthdata token required locally):

```bash
PYTHONDONTWRITEBYTECODE=1 /home/emily/miniforge3/envs/terra/bin/python -m src.probe_gunw \
  --sites-config data/sakurajima_gunw_search/probe_sites.json \
  --catalogue-root data/sakurajima_gunw_search \
  --data-root data/sakurajima_gunw_search/rasters \
  --token-file src/token.txt --workers 3
```

This command repeats the basic coverage screen; the added native-spacing and coherence fields in this report were measured locally from the resulting subsets with src.nisar_processing.read_pair. Existing subsets are reused by the downloader.
