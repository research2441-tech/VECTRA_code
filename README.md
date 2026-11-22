# VECTRA_code

Code, results and figures for the manuscript

**VECTRA: Thermally Anchored Renewal Forecasting with Intervention-Conditioned Scenarios and Region-Adaptive Calibration for Outage-Tolerant Dengue Vector-Control Allocation over an Edge–Fog–Cloud Continuum**

VECTRA forecasts weekly dengue cases one to eight weeks ahead with a renewal model whose generation interval and thermal suitability are fixed by published *Aedes aegypti* data, corrected by bounded GRU closures. Larviciding, adulticiding and source reduction act on the reproduction number through the vector traits they change. The forecasts feed an excess-burden allocation linear program, region-and-horizon adaptive conformal intervals and a simulated edge-fog-cloud placement with DVFS and a state-level fallback.

Everything sits in one flat folder.

## Quick start

Google Colab (free tier, CPU only): open `VECTRA_Colab.ipynb` and run all cells.

Local:

```bash
pip install -r requirements.txt
python run_all.py main transfer belem     # add --quick for a short smoke test
python analyze.py
python make_schematics.py
python make_figures.py
```

The full run took about 25 minutes on a 2-core x86 CPU without a GPU.

## Data

Nothing needs to be downloaded by hand. On first use, `vectra_data.py` fetches the public files from the Hugging Face Hub and stores them next to the code as `data_meso_*.parquet` and `data_belem.csv`.

| Dataset | Hugging Face repository | Content |
|---|---|---|
| Primary | `thiagorfreitas/dengue-tcc2-data` | SINAN probable dengue cases and INMET station climate for 43 mesoregions (MG, ES, RJ, SP, GO, DF), weekly, 2019 to 2023 |
| External | `JeanDias/InfoDengue_Belem` | InfoDengue weekly notified cases and climate for Belém, 2010 to 2024 |

Splits are chronological. Primary data: train 2019 to 2021, validation 2022, test 2023. Belém: train 2010 to 2020, validation 2021, test 2022 to 2024.

## Files

| File | Purpose |
|---|---|
| `vectra_data.py` | Data download, weekly panels, climate gap filling, coupling graph |
| `vectra_mech.py` | Thermal suitability, EIP, generation interval kernel, intervention shifts |
| `vectra_models.py` | VECTRA, GRU-NB, Graph-GRU-NB and the feature window |
| `vectra_train.py` | Training, refit protocol, persistence, NB-GLM, LightGBM-Q and LSTM-GBM baselines |
| `vectra_metrics.py` | WIS, coverage, alert metrics |
| `vectra_decision.py` | Averted-excess matrix, expected-value and CVaR allocation, regret, SPO+ |
| `vectra_conformal.py` | Model-native, split and adaptive conformal calibration |
| `vectra_offload.py` | Edge-fog-cloud simulator with DVFS and outage fallbacks |
| `run_all.py` | All experiments; writes `results_*.json` and `cache_*.pt` |
| `analyze.py` | Wilcoxon/Holm, Friedman/Nemenyi, bootstrap; writes `results_analysis.json` |
| `make_schematics.py` | Fig. 1 and Fig. 2 (system and model diagrams) |
| `make_figures.py` | Result figures, grayscale |
| `VECTRA_Colab.ipynb` | Self-contained notebook that writes every module and reruns the study |
| `results_*.json` | Results reported in the paper |
| `fig*.png` | Figures of the paper |
| `cache_SE-Brazil_*_s0_m1.pt` | Seed-0 weights of VECTRA and GRU-NB used for the fan chart |

## Main results (2023 test year, five seeds)

| Model | WIS | Cov80 | Allocation regret |
|---|---|---|---|
| Reactive rule | n/a | n/a | 33.8% |
| NB-GLM | 66.30 | 0.542 | 32.3% |
| LightGBM-Q | 74.10 ± 0.46 | 0.662 | 25.8% |
| GRU-NB | 65.79 ± 3.38 | 0.788 | 21.5% |
| Graph-GRU-NB | 68.25 ± 10.75 | 0.800 | 19.8% |
| **VECTRA** | **65.61 ± 1.17** | **0.796** | **22.1%** |

The differences in WIS between VECTRA and the learned baselines are not statistically significant. See the paper for the transfer, external validation, calibration and deployment results, including the negative results for the coupling graph, CVaR allocation and SPO+ fine-tuning.

## Reproducibility notes

* Seeds 0 to 4; 200 scenario paths per forecast.
* Hyperparameters were chosen on the 2022 validation year only, and all learned models were refitted on 2019 to 2022 before testing.
* Intervention efficacies and costs are assumptions documented in `vectra_mech.py` and `vectra_decision.py`.
* Deployment results come from a simulator with assumed network and hardware parameters (`vectra_offload.py`); only the workloads are measured.

## License

MIT, see `LICENSE`. The datasets keep the terms stated on their Hugging Face pages.
