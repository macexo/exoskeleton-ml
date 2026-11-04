# Reproducing the 2025-26 Baseline

> **Status:** Returning members keep this updated to track progress. Sections marked TODO aren't verified yet.

## 1. Setup

```bash
cd archive/2025-26
uv sync --locked
uv run pytest
```

uv installs Python 3.10 for this folder automatically.

The default test command runs 44 offline tests. The 11 integration tests download the public Hugging Face dataset and run separately:

```bash
uv run pytest -m integration
```

## 2. Data access

Training uses the [`MacExo/exoData` dataset on Hugging Face](https://huggingface.co/datasets/MacExo/exoData) (about 10 GB once downloaded)

No login needed, dataset downloads automatically on first use.

## 3. Train

```bash
uv run scripts/train_tcn.py
```

Outputs are saved to `outputs/<date>/<time>/` (ignored by Git). Last year's docs estimate 2-4 hours on a GPU and 8-12 hours on a CPU. For a quick check, add `training.num_epochs=5`.

## 4. Evaluate

Evaluation happens automatically after training. The script reloads `best_model.pt`, evaluates it on the configured test split, prints the metrics to the terminal, and saves them in `outputs/<date>/<time>/training_results.pt`.

Currently no standalone command for evaluating an existing checkpoint 

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
