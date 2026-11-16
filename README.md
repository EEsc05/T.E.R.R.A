# T.E.R.R.A

**Terrain Evolution & Radar-based Research Analytics**

T.E.R.R.A uses satellite radar observations to study ground movement and forecast future deformation maps. The project includes data preparation, model training, evaluation and an interactive website for viewing observations and predictions.

The forecasting models include persistence, EMA, smoothing, robust low-rank, a four-branch hybrid and a conditional variational autoencoder (CVAE). Persistence provides the baseline for comparison.

The dataset described here covers **14 volcanoes in 10 countries**. It combines NISAR GUNW and Sentinel-1 ARIA GUNW products obtained through NASA’s Alaska Satellite Facility (ASF).

| Country | Volcanoes / regions |
| --- | --- |
| Colombia | Nevado del Ruiz |
| Ecuador | Cotopaxi |
| Ethiopia | Dabbahu, Erta Ale |
| Iceland | Reykjanes |
| Japan | Mount Asama |
| Mexico | Popocatépetl |
| New Zealand | Taupō |
| Peru | El Misti, Sabancaya, Ubinas, Coropuna |
| United States | Kīlauea |
| Vanuatu | Ambrym |

See the [readable data-source list](outputs/nisar_aria_all_data/DATA_SOURCES_READABLE.txt) for each location’s satellite source, or the [full inventory](outputs/nisar_aria_all_data/DATA_SOURCES.txt) for product links and dates.

To set up the Python environment, install Miniforge or Conda, then run these commands from the project folder:

```bash
conda env create -f environment.yml
conda activate terra
python -m src --help
```

If the environment already exists, update it with `conda env update -n terra -f environment.yml`.

- `src/` contains the current preparation, training and forecasting code.
- `data/` contains downloaded and prepared datasets. Model inputs use NumPy `.npz` files rather than CSV.
- `outputs/` contains saved models, evaluation results, figures and website versions.

For more instructions, see [src/README.md](src/README.md). Results for the dataset above are in the [evaluation report](outputs/nisar_aria_all_data/REPORT.md).

Forecast maps represent relative ground movement along the satellite’s line of sight. This is a research project; deformation forecasts are not eruption predictions or official hazard warnings.
