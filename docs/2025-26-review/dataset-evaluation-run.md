# Five-epoch baseline run and dataset audit

**Date:** November 12, 2026
**Purpose:** Provide a traceable short run for [issue #19](https://github.com/macexo/exoskeleton-ml/issues/19), retaining the archived implementation and documenting its behavior.
**Execution:** Assistant-assisted run requested by Krishiv; Saharsh's review remains pending. This does not certify both members have reviewed or independently executed it.

## Provenance

- Source revision: `5747037e203c0e69e853f8f070ca7fcf10dfe5ea` on `docs/review-dataset-evaluation`; archived files unchanged from this revision.
- Dataset: [`MacExo/exoData`, revision `d3640796d9b8697153f36c9154ced1555cd9230d`](https://huggingface.co/datasets/MacExo/exoData/tree/d3640796d9b8697153f36c9154ced1555cd9230d).
- Download: six Parquet shards, checked against the Hub file sizes and SHA-256/LFS object hashes. [Inspection and hashes](dataset-evaluation-data-audit.json)
- Runtime: Python 3.10, PyTorch 2.10.0, Apple MPS. Exact runtime and result hashes appear in the [machine-readable record](dataset-evaluation-results.json).
- Environment workaround: the existing archive environment's SciPy 1.15.3 failed to load `scipy.sparse.linalg._propack._spropack` on macOS (`__thread_bss` Mach-O section error). A separate SciPy 1.14.1 wheel was installed in the chat's working folder and selected with `PYTHONPATH`; neither the repository dependencies nor the archive environment were modified. This run is therefore **not an exact locked-environment reproduction**.

The data cache was populated from the pinned Parquet shards using Hugging Face `Dataset.from_parquet`, `save_to_disk`, and metadata in the format the original downloader expects. The archive then used its existing local-cache path with networking disabled. No participant/task truncation or synthetic data was used. [Download/cache implementation](../../archive/2025-26/src/exoskeleton_ml/data/download.py)

## Configuration and execution

The [resolved configuration](dataset-evaluation-config.yaml) records the original Hydra settings with these overrides; local folder prefixes are generalized in the committed copy:

```text
training.num_epochs=5
training.batch_size=8
data.num_workers=0
data.cache_dir=<chat-work>/data/archive_cache
training.output_dir=<chat-work>/run/five_epoch
hydra.run.dir=<chat-work>/run/five_epoch
```

The batch size was reduced from 32 to 8 for local memory use; it is not a tuned hyperparameter result. The model remains the default five-block, 25-channel TCN with kernel 7 and dropout 0.2; optimizer Adam, learning rate 0.001, weight decay 0.0001. Label/input normalization, zero substitution, participant split, checkpoint selection and test scoring are unchanged. [Original model config](../../archive/2025-26/configs/model/tcn.yaml) · [Original training code](../../archive/2025-26/scripts/train_tcn.py)

The command was executed from `archive/2025-26/`:

```sh
PYTHONPATH="<chat-work>/runtime:<repo>/archive/2025-26/src" \
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 \
<archive-python> scripts/train_tcn.py \
  training.num_epochs=5 training.batch_size=8 data.num_workers=0 \
  data.cache_dir=<chat-work>/data/archive_cache \
  training.output_dir=<chat-work>/run/five_epoch \
  hydra.run.dir=<chat-work>/run/five_epoch
```

`<chat-work>` and `<repo>` are path placeholders for portability; the committed configuration uses the same path placeholders. The original configuration with exact local paths is retained in the local evidence bundle and identified by `config_sha256` in the results record. `seed: 42` is present in the configuration, but the original trainer does not initialize random generators from it. No external seeding was applied, so an exact numerical rerun is not promised. [Trainer](../../archive/2025-26/scripts/train_tcn.py)

## Dataset inventory and missingness

Counting `sequence_length` per Parquet record gives 1,669 trials and 9,369,823 timesteps, with lengths from 216 to 43,201. For each split, nested target arrays were flattened to `(timesteps, 4)` in the defined hip L/R, knee L/R order; `~isfinite` was counted before any zero substitution. Arrow nulls become NumPy NaNs during this inspection. These are counts of scalar target values, not recordings. [Machine-readable audit](dataset-evaluation-data-audit.json)

| Split | Trials | Timesteps | Missing/non-finite joint values | All joint values | Missing fraction |
|---|---:|---:|---:|---:|---:|
| train | 1,375 | 7,536,065 | 5,751,494 | 30,144,260 | 19.08% |
| val | 73 | 466,355 | 359,756 | 1,865,420 | 19.29% |
| test | 221 | 1,367,403 | 1,089,972 | 5,469,612 | 19.93% |


The measured missingness shows why the archived score needs qualification: the loader fills missing labels with zero and its mask excludes padding rather than original missing labels. [Loader](../../archive/2025-26/src/exoskeleton_ml/data/datasets.py)

## Training and official archived test metrics

| Item | Value |
|---|---:|
| Epochs completed | 5 |
| Selected checkpoint epoch | 3 |
| Best validation loss | 0.23206183612346648 |
| Test loss (equal-weight mean of batch MSEs) | 0.2670237249029534 |
| `rmse_overall` | 0.4986379297 |
| `rmse_hip_l` | 0.4971362740 |
| `rmse_hip_r` | 0.5049592487 |
| `rmse_knee_l` | 0.5220316502 |
| `rmse_knee_r` | 0.4689496591 |
| `mae_overall` | 0.3502026834 |
| `r2_overall` | 0.6410168747 |
| `nrmse_overall` | 0.0311556013 |

Training loss per epoch: 0.558289, 0.359853, 0.311699, 0.296499, 0.277481.
Validation loss per epoch: 0.310490, 0.262302, 0.232062, 0.253839, 0.232274.


The script selected the checkpoint by the lowest mean validation batch loss, reloaded it and executed its final test step on all 221 test recordings. These are **standardized-target metrics produced by the archived code**. Its “Nm/kg” suffix is not valid with normalization enabled. [Trainer](../../archive/2025-26/scripts/train_tcn.py) · [Target standardization](../../archive/2025-26/src/exoskeleton_ml/data/datasets.py)

## Read-only diagnostic rescoring

| Scoring method | RMSE | MAE | R² | NRMSE | Scalar references |
|---|---:|---:|---:|---:|---:|
| Archived standardized | 0.498638 | 0.350203 | 0.641017 | 0.031156 | 5,469,612 |
| Physical, zero-filled | 0.198915 | 0.139451 | 0.642298 | 0.030524 | 5,469,612 |
| Physical, valid references only | 0.209252 | 0.147437 | 0.682420 | 0.032110 | 4,379,640 |

RMSE and MAE in the two physical rows are **Nm/kg**. NRMSE and R² are dimensionless. The diagnostic pass reproduced every official archived metric exactly.

| Valid-reference physical RMSE | Hip L | Hip R | Knee L | Knee R |
|---|---:|---:|---:|---:|
| Nm/kg | 0.211883 | 0.238486 | 0.196444 | 0.186555 |

| Participant (valid-reference physical metrics) | RMSE (Nm/kg) | MAE (Nm/kg) | Pooled R² |
|---|---:|---:|---:|
| BT15 | 0.226220 | 0.158235 | 0.634302 |
| BT16 | 0.212440 | 0.147815 | 0.718544 |
| BT17 | 0.187463 | 0.136354 | 0.684662 |


This additional pass does not retrain or change the checkpoint. It reconstructs physical predictions channel-wise using the saved training means and `(std + 1e−8)` scales, and compares them with original cached target tensors. Three summaries are kept separate:

1. **Archived standardized:** recompute exactly the original score to verify the saved results.
2. **Physical, zero-filled:** undo standardization but retain the archive's fabricated zero references.
3. **Physical, valid references only:** undo standardization and exclude original non-finite references. This answers a different scoring question; it does not repair the model's earlier training on zero-filled labels.

The original `RunningMetrics` accumulator is used for all three summaries. Each source target is read from its cached tensor before `__getitem__` replaces NaNs. Padding remains masked. This is a diagnostic of unit/missing-label effects, not a claim to reproduce the paper's task/participant averaging, delay alignment, sensors or online controller. [Metric formulas](../../archive/2025-26/src/exoskeleton_ml/utils/metrics.py) · [Original cache and zero substitution](../../archive/2025-26/src/exoskeleton_ml/data/datasets.py)

## Evidence and limitations

The committed JSON contains epochs, validation/training losses, best checkpoint epoch, official and diagnostic test scores, scored-value counts, and checkpoint/results/config hashes. The companion evidence bundle retained locally includes the original log, `best_model.pt`, `training_results.pt`, normalization statistics and the helper scripts used for preparation/audit/rescoring. Data and model binaries are not committed, following [CONTRIBUTING.md](../../CONTRIBUTING.md).

This is one short run with modified batch size and a runtime compatibility workaround. It establishes that this archived pipeline was trained and tested under the recorded conditions; it does not establish convergence, a statistically reliable improvement, equal performance to last year, or readiness for hardware use. The documented historical values remain separate from this run. Both assigned members' report review is still required before requesting a lead review. [Issue #19](https://github.com/macexo/exoskeleton-ml/issues/19)
