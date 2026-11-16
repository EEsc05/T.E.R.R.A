# ARIA GUNW preparation

Converted the saved Sentinel-1 ARIA inventory into the same NumPy training schema as the current NISAR inputs. Seven sites have clean temporal training/reporting splits; Ambrym has three training-only examples. No NISAR histories or existing model artifacts were changed.

| Volcano | Country | Connected dates retained | Train | Report | Mean pair valid cells |
| --- | --- | ---: | ---: | ---: | ---: |
| coropuna | Peru | 60 | 36 | 9 | 29.2% |
| cotopaxi | Ecuador | 60 | 34 | 9 | 90.8% |
| erta_ale | Ethiopia | 48 | 18 | 4 | 24.9% |
| kilauea | United States | 60 | 30 | 8 | 89.7% |
| reykjanes | Iceland | 60 | 32 | 8 | 21.7% |
| ruiz_nevado_del | Colombia | 60 | 36 | 9 | 47.6% |
| taupo | New Zealand | 30 | 12 | 3 | 37.5% |
| ambrym | Vanuatu | 43 | 3 (training only) | 0 | 35.3% |

Total ARIA: 201 training examples (198 with reporting partitions plus 3 Ambrym), and 50 reporting examples. Combined with the prior NISAR inputs: 254 training examples, 62 reporting windows, 14 physical volcanoes, and 10 countries.

## Scope and excluded sites

The 22-site survey had 10 sites with catalogued products. Twelve had no products in that survey. Manam had only three dates and no coherent pixels in the sampled AOI. Lastarria sampled products covered too little of the configured AOI. These outcomes are not proof that no other usable products exist. Up to three nominal stacks were screened per site, using three representative products before full processing.

The primary 30 km Ambrym AOI failed the coverage threshold. A separate 20 km AOI around the same volcano center retained 69 pairs, of which 44 form the retained connected component. Only three causal supervised examples survived; all are training only. The original thresholds were retained. Data are under data/aria_island/ambrym.

Each stack uses at most 60 connected acquisitions from the existing inventory. Quality filtering can shorten its connected component further. This is not a complete download of every catalogued product. The Reykjanes catalogue hit its 5,000-product search cap.

## Representation and limitations

Each history contains six dated 32x32 maps of relative LOS displacement in mm and validity masks. Targets use actual future acquisitions within 12 days of a requested 30-day horizon; observed training/reporting horizons span 18–36 days. No labels are interpolated in time. Each history is inverted only from pairs acquired by its forecast origin. Product publication latency is not modeled.

ARIA phase uses the product wavelength and a +1 conversion sign for chronological displacement positive toward the sensor. Native NISAR uses opposite phase/date order, so its existing -1 conversion is consistent. See sign_and_schema_audit.json, the [ASF guide](https://hyp3-docs.asf.alaska.edu/guides/gunw_product_guide/) and [ARIA-tools documentation](https://github.com/aria-tools/ARIA-tools#readme).

Native pixel centers are projected onto UTM physical grids and block-averaged. Coherence >=0.4, cell valid fraction >=0.5 and pair valid fraction >=0.2 are enforced. Only the dominant coherent unwrapping component is retained. ARIA and NISAR have distinct wavelengths, geometries and noise. Matching units and dimensions does not make their measurements interchangeable.

Each pair is referenced to its retained-pixel median; reference coverage can change, and spatially uniform displacement is removed. No additional atmospheric, ionospheric or tide corrections are applied. This remains experimental relative deformation data.

## Verification

All seven train/report pairs passed the real TrainingDataset reader, causal metadata checks, fixed-shape checks, nonempty reporting mask checks and temporal split validation. Pair/source provenance is recorded per site. No token or credential is included in these outputs. Original whole product files are not persisted; reduced pair arrays are cached. Modest products may be fetched into bounded memory for faster conversion; larger products use HTTP ranges.
