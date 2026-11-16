# Expanded GUNW discovery

Earthdata authentication succeeded. Queried 63 volcanoes in 23 GVP country/territory labels; 62 had NISAR GUNW catalogue matches. The survey contains 1913 distinct NISAR GUNW product IDs and 224 stack candidates. This is a geographically diverse sample, not an exhaustive global inventory.

Coordinates and country labels come from the [Smithsonian GVP WFS](https://volcano.si.edu/database/webservices.cfm). NISAR discovery is restricted to the GUNW processing level, with a maximum 36-day pair baseline for candidate stacks. Candidate stacks exclude BETA; raw catalogues retain maturity labels for audit.

## Prepared NISAR inputs

| Volcano | Country | Dates | All supervised windows | Training | Validation | Training valid pixels |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Popocatepetl | Mexico | 29 | 21 | 11 | 3 | 94.1% |
| Dabbahu | Ethiopia | 28 | 20 | 10 | 3 | 99.8% |

Files are under `data/nisar_global/SITE/`: `train.npz`, `validation.npz`, `examples.npz`, `history.npz`, `series.npz` and metadata. All archives passed the real dataset reader, fixed-grid model preparation and temporal split checks. Seven windows per site were purged between partitions. Targets are actual 24- or 36-day observations; no temporal interpolation. All selected products are `NISAR_L2_GUNW_PROVISIONAL_V1`. These are calibrated provisional products, not a claim of fully validated satellite accuracy.

## Additional NISAR sites screened

Three products per site (early, middle and late) were read through authenticated HTTP ranges. The coverage check retains the current coherence >=0.4 and block-valid fraction >=0.5 thresholds. This screen is not a full-stack connectivity or model-performance assessment.

| Volcano | Country | Connected dates | Sample valid fraction range | Three-sample check |
| --- | --- | ---: | ---: | --- |
| Popocatepetl | Mexico | 29 | 93.3%–97.1% | pass |
| Dabbahu | Ethiopia | 28 | 99.8%–100.0% | pass |
| Hekla | Iceland | 27 | 28.8%–67.7% | pass |
| Askja | Iceland | 28 | 0.0%–51.7% | fail |
| Kilauea | United States | 28 | 56.2%–98.0% | pass |
| Tongariro | New Zealand | 27 | 80.5%–95.6% | pass |
| Lascar | Chile | 26 | 96.4%–98.9% | pass |

## Longer historical ARIA GUNW inventory

Separately queried existing Sentinel-1 ARIA GUNW for one candidate per country label. It is stored in `aria_survey.json` and `aria_catalogues/`; it is not loaded by the current NISAR HDF5 reader. The [ARIA GUNW product guide](https://hyp3-docs.asf.alaska.edu/guides/gunw_product_guide/) describes this separate product family. Phase/reference sign, radar wavelength, geolocation, masks and file variables must be adapted before producing matching model inputs. Dates below count catalogue connectivity only.

| Volcano | Country | GUNW products returned | Best connected dates | Date range |
| --- | --- | ---: | ---: | --- |
| Ruiz, Nevado del | Colombia | 2825 | 306 | 2016-08-05–2023-04-07 |
| Cotopaxi | Ecuador | 1856 | 268 | 2015-06-12–2022-05-18 |
| Ambrym | Vanuatu | 1221 | 252 | 2016-10-18–2022-05-14 |
| Reykjanes | Iceland | 5000+ | 248 | 2018-02-05–2022-05-21 |
| Taupo | New Zealand | 1046 | 185 | 2014-11-04–2021-12-15 |
| Coropuna | Peru | 937 | 150 | 2016-10-26–2021-12-11 |
| Kilauea | United States | 2689 | 140 | 2015-05-30–2018-11-28 |
| Erta Ale | Ethiopia | 1380 | 48 | 2018-11-20–2020-06-18 |
| Lastarria | Chile-Argentina | 1287 | 26 | 2021-05-13–2021-11-03 |
| Manam | Papua New Guinea | 3 | 3 | 2022-08-20–2022-09-13 |

The Iceland query hit the 5,000-product limit, so its inventory is incomplete. ARIA date endpoints are sorted for graph counting only, not interpreted as a physical phase/reference convention.

## Maturity audit

Two BETA pairs were found in the existing Ubinas selection. They are recorded in `legacy_beta_audit.json`; the earlier archives and both trained model sets were preserved. New preparation excludes BETA by default (including offline subsets with provenance), with explicit `--include-beta` available for deliberate research use. [ASF maturity definitions](https://nisar-docs.asf.alaska.edu/asf-search/#search-by-data-maturity) distinguish pre-calibration BETA from calibrated provisional products.

No new models were trained during this search. Prepared data and catalogue-only candidates are distinguished above.
