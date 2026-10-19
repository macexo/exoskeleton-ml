# Reproducing the 2025-26 Baseline

> **Status:** Returning members keep this updated to track progress. Sections marked TODO aren't verified yet.

## 1. Setup

```bash
cd archive/2025-26
uv sync
uv run pytest
```

uv installs Python 3.10 for this folder automatically.

## 2. Data access

Training uses the `MacExo/exoData` dataset on Hugging Face (about 10 GB once downloaded).

TODO: who grants access, what token type is needed (read access should be enough), and how to log in (`uv run scripts/login_hf.py` or `HF_TOKEN` in a local `.env`).

## 3. Train

```bash
uv run scripts/train_tcn.py
```

TODO: confirm this works with the default config. Last year's docs estimate ~2-4 hours on a GPU and ~8-12 hours on CPU. Outputs go to `outputs/<date>/<time>/` (ignored by Git).

## 4. Evaluate

TODO: the command to evaluate a trained model and where the metrics are printed or saved.

## 5. Expected results

Last year's docs list these targets from the original paper. They are not results last year's team measured:

| Metric | Target |
|---|---|
| Overall RMSE | 0.2-0.4 Nm/kg |
| Hip RMSE | 0.3-0.5 Nm/kg |
| Knee RMSE | 0.15-0.3 Nm/kg |
| R² | 0.80-0.95 |

TODO: replace with the baseline results from this year's first successful reproduction.

## Known issues

TODO: anything that differs from these steps or needs a workaround.