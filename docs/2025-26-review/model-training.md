# 2025–26 Model and Training Review

## Summary

The pipeline trains a causal temporal convolutional network (TCN) to predict four joint moments from 28 IMU and joint-angle inputs at each timestep. It batches padded trials, uses masked MSE and Adam, and selects the best checkpoint by validation loss. Several config options and receptive-field comments do not match the code. ([training script](../../archive/2025-26/scripts/train_tcn.py#L198), [TCN](../../archive/2025-26/src/exoskeleton_ml/models/tcn.py#L246), [config](../../archive/2025-26/configs/config.yaml#L3))

## How it works

### Model and data path

| Step | Behavior | Source |
|---|---|---|
| Data | Loads 24 IMU channels and four joint angles, normalizes features and targets, pads trials, and marks real timesteps in a mask. | [data config](../../archive/2025-26/configs/data/phase1.yaml), [dataset](../../archive/2025-26/src/exoskeleton_ml/data/datasets.py#L295) |
| TCN | Two causal Conv1d layers per block; dilations grow by powers of two. Chomping removes future padding. ReLU, dropout, weight normalization, and residual skips follow each block. | [blocks](../../archive/2025-26/src/exoskeleton_ml/models/tcn.py#L17), [stack](../../archive/2025-26/src/exoskeleton_ml/models/tcn.py#L179) |
| Output | A linear layer predicts four continuous moment values per timestep. | [output layer](../../archive/2025-26/src/exoskeleton_ml/models/tcn.py#L284) |
| Training | Computes masked squared error, clips gradients, updates Adam, validates each epoch, saves checkpoints, and tests the best validation-loss checkpoint. | [loss](../../archive/2025-26/scripts/train_tcn.py#L41), [optimizer and scheduler](../../archive/2025-26/scripts/train_tcn.py#L259), [loop](../../archive/2025-26/scripts/train_tcn.py#L294) |

### Causality and receptive field

Causal convolutions use only current and past inputs. Dilation spaces convolution taps farther apart in successive blocks, growing the lookback exponentially. Residual skips provide a direct path through each block; a 1×1 convolution matches channel widths when needed. ([block implementation](../../archive/2025-26/src/exoskeleton_ml/models/tcn.py#L17), [dilations](../../archive/2025-26/src/exoskeleton_ml/models/tcn.py#L208))

The default has five blocks, kernel size 7, and dilations 1, 2, 4, 8, 16. With two convolutions per block:

```text
R = 1 + 2 × (7 − 1) × (1 + 2 + 4 + 8 + 16) = 373 samples
```

At 100 Hz, this is 3.73 seconds of context. The often-cited 187 samples counts one convolution per block. The factory reports that incorrect value; the actual stack uses two. Actual fields are 121 samples for small and 1009 for large, twice the config comments. ([factory](../../archive/2025-26/src/exoskeleton_ml/models/model_factory.py#L33), [block](../../archive/2025-26/src/exoskeleton_ml/models/tcn.py#L82), [default config](../../archive/2025-26/configs/model/tcn.yaml#L13), [small](../../archive/2025-26/configs/model/tcn_small.yaml#L13), [large](../../archive/2025-26/configs/model/tcn_large.yaml#L13))

### Why a TCN; baseline and sizes

The team says it chose a TCN after the prior season’s LSTM underperformed, but no controlled comparison or metrics were preserved. TCNs support causal predictions and process sequence positions in parallel; LSTMs and GRUs process recurrent states step by step. Transformers are not implemented or compared here. The TCN paper compares models on other datasets, not MacExo’s. ([team findings](https://docs.google.com/document/d/1kilBxeR_R3_EPtA63CWHcPSqFdR6qLgfnSk4lgX6Gxw/edit?usp=sharing), [TCN paper](https://arxiv.org/abs/1803.01271), [model configs](../../archive/2025-26/configs/model/))

`BaselineModel` has no layers and its `forward` is empty. Its config describes a 10-input, five-class LSTM classifier, unlike this 28-input/four-output regression task. It is not a usable comparison model. ([baseline code](../../archive/2025-26/src/exoskeleton_ml/models/baseline.py#L7), [baseline config](../../archive/2025-26/configs/model/baseline.yaml), [team findings](https://docs.google.com/document/d/1kilBxeR_R3_EPtA63CWHcPSqFdR6qLgfnSk4lgX6Gxw/edit?usp=sharing))

| Variant | Blocks × channels | Kernel | Dropout | Actual / stated field* |
|---|---:|---:|---:|---:|
| Small | 4 × 15 | 5 | 0.2 | 121 / 61 |
| Default | 5 × 25 | 7 | 0.2 | 373 / 187 |
| Large | 6 × 50 | 9 | 0.3 | 1009 / 505 |

*Samples; actual values account for both convolutions per block. All use 28 inputs and four outputs. ([small](../../archive/2025-26/configs/model/tcn_small.yaml#L8), [default](../../archive/2025-26/configs/model/tcn.yaml#L8), [large](../../archive/2025-26/configs/model/tcn_large.yaml#L8), [block construction](../../archive/2025-26/src/exoskeleton_ml/models/tcn.py#L211))

### Loss, optimizer, schedule and stopping

The loss averages squared error over valid timesteps and all four targets. Padding is excluded by the mask. The `criterion` argument is unused; the function directly computes squared error. ([loss](../../archive/2025-26/scripts/train_tcn.py#L41), [mask](../../archive/2025-26/src/exoskeleton_ml/data/datasets.py#L315))

The config requests Adam, learning rate 0.001, and weight decay 0.0001. Code always uses Adam but reads the learning rate and weight decay. It hard-codes `ReduceLROnPlateau` (factor 0.5, patience 10, minimum LR 1e-6) and gradient clipping at 1.0; corresponding config fields do not control these values. The scheduler steps on validation loss. ([config](../../archive/2025-26/configs/config.yaml#L9), [training setup](../../archive/2025-26/scripts/train_tcn.py#L259), [clipping](../../archive/2025-26/scripts/train_tcn.py#L122))

Early stopping monitors validation loss and stops after 20 epochs without improvement (default patience; `min_delta=0`). The best checkpoint is saved on improvement and reloaded for test evaluation. Periodic checkpoints store scheduler state; `best_model.pt` does not. ([early stopping](../../archive/2025-26/src/exoskeleton_ml/utils/early_stopping.py#L11), [loop](../../archive/2025-26/scripts/train_tcn.py#L325), [checkpointing](../../archive/2025-26/src/exoskeleton_ml/utils/checkpointing.py#L11))

### Hydra and run evidence

Hydra composes `configs/config.yaml` with the `data/phase1` and `model/tcn` groups. Overrides such as `model=tcn_small` or `training.num_epochs=5` select a config or replace a value; `-m` runs a multirun sweep. The script saves the composed config. Some accepted overrides have no effect: optimizer choice, scheduler settings, clip norm, seed, and device are ignored or hard-coded in the training code. ([main config](../../archive/2025-26/configs/config.yaml#L3), [entry point](../../archive/2025-26/scripts/train_tcn.py#L198), [quick start](../../archive/2025-26/docs/tcn_quick_start.md#L65), [Hydra docs](https://hydra.cc/docs/1.3/intro/), [training setup](../../archive/2025-26/scripts/train_tcn.py#L263))

The findings doc records the pair’s five-epoch Colab T4 run: `uv run scripts/train_tcn.py training.num_epochs=5`, test split BT15–BT17, and validation RMSE falling from 0.625 to 0.500. It reports these test metrics alongside the historical result: ([findings and log reference](https://docs.google.com/document/d/1kilBxeR_R3_EPtA63CWHcPSqFdR6qLgfnSk4lgX6Gxw/edit?usp=sharing), [split config](../../archive/2025-26/configs/data/phase1.yaml))

| Metric | Pair run, 5 epochs | Historical figure |
|---|---:|---:|
| RMSE (reported Nm/kg) | 0.500 | 0.4316 |
| R² | 0.639 | 0.7311 |
| MAE (reported Nm/kg) | 0.348 | 0.3014 |
| Hip L / R RMSE | 0.499 / 0.488 | 0.44 average |
| Knee L / R RMSE | 0.520 / 0.493 | 0.42 average |

**Metric caveat:** targets are standardized in the default data path, and metrics run on those values without inverse transformation. The script labels RMSE and MAE as Nm/kg, but the archived path appears to produce standardized-unit errors. Verify the Colab config and log before treating these as physical-unit metrics. ([normalization](../../archive/2025-26/configs/data/phase1.yaml#L103), [target scaling](../../archive/2025-26/src/exoskeleton_ml/data/datasets.py#L303), [metric path](../../archive/2025-26/scripts/train_tcn.py#L164), [metric implementation](../../archive/2025-26/src/exoskeleton_ml/utils/metrics.py#L37))

The optimization plan has checkmarks beside proposed work, but its results table has only historical baseline values; planned experiment rows are blank. The evidence does not show that any planned optimization was run. The five-epoch pair run is the only additional run found. ([plan](../../archive/2025-26/docs/optimization_plan.md#L268), [results](../../archive/2025-26/docs/optimization_plan.md#L357), [pair findings](https://docs.google.com/document/d/1kilBxeR_R3_EPtA63CWHcPSqFdR6qLgfnSk4lgX6Gxw/edit?usp=sharing))

## Advantages

- Causal predictions use past sensor data only, which fits real-time control. ([implementation](../../archive/2025-26/src/exoskeleton_ml/models/tcn.py#L17), [TCN paper](https://arxiv.org/abs/1803.01271))
- Convolutions process sequence positions in parallel, unlike recurrent LSTMs/GRUs. Actual controller latency still needs measurement. ([TCN stack](../../archive/2025-26/src/exoskeleton_ml/models/tcn.py#L179), [TCN paper](https://vladlen.info/papers/TCN.pdf))
- Block count, width, kernel, and dropout provide simple capacity/context options. ([factory](../../archive/2025-26/src/exoskeleton_ml/models/model_factory.py#L33), [configs](../../archive/2025-26/configs/model/))
- Validation selects the checkpoint used for final testing. ([training loop](../../archive/2025-26/scripts/train_tcn.py#L328))

## Disadvantages

- Factory and config comments undercount receptive field by half. ([factory](../../archive/2025-26/src/exoskeleton_ml/models/model_factory.py#L33), [config](../../archive/2025-26/configs/model/tcn.yaml#L20), [plan](../../archive/2025-26/docs/tcn_implementation_plan.md#L102))
- The baseline is an empty stub, so there is no valid comparison against an LSTM, GRU, or simpler model. ([baseline](../../archive/2025-26/src/exoskeleton_ml/models/baseline.py#L7))
- Metrics may be standardized units despite the Nm/kg label. Neither the pair result nor historical 0.4316 should be interpreted as physical error until verified. ([target scaling](../../archive/2025-26/src/exoskeleton_ml/data/datasets.py#L303), [metrics](../../archive/2025-26/src/exoskeleton_ml/utils/metrics.py#L59), [labels](../../archive/2025-26/scripts/train_tcn.py#L308))
- Several Hydra training overrides are ignored, reducing reproducibility. ([config](../../archive/2025-26/configs/config.yaml#L15), [training setup](../../archive/2025-26/scripts/train_tcn.py#L263))
- Dataset preprocessing changes NaNs to zero, making missing torques zero-valued labels. The loss’s NaN mask would also fail if NaNs reached it because `NaN × 0` remains NaN. ([dataset](../../archive/2025-26/src/exoskeleton_ml/data/datasets.py#L298), [loss](../../archive/2025-26/scripts/train_tcn.py#L61))
- Causality does not prove real-time suitability. No target-device latency or memory results were found; default context is 3.73 seconds. ([default config](../../archive/2025-26/configs/model/tcn.yaml#L13), [team findings](https://docs.google.com/document/d/1kilBxeR_R3_EPtA63CWHcPSqFdR6qLgfnSk4lgX6Gxw/edit?usp=sharing))
- The pair reports using Colab because the archived Python/SciPy setup failed on recent macOS. ([team findings](https://docs.google.com/document/d/1kilBxeR_R3_EPtA63CWHcPSqFdR6qLgfnSk4lgX6Gxw/edit?usp=sharing))

## Open questions

- What code, data split, and metric pipeline produced the historical 0.4316 result? No log or checkpoint was found. ([optimization results](../../archive/2025-26/docs/optimization_plan.md#L357))
- Did the Colab run use the archived normalization and metric code, or inverse-transform predictions? Check its `run_log.txt`. ([data config](../../archive/2025-26/configs/data/phase1.yaml#L103), [findings/log reference](https://docs.google.com/document/d/1kilBxeR_R3_EPtA63CWHcPSqFdR6qLgfnSk4lgX6Gxw/edit?usp=sharing))
- Are prior LSTM results recorded elsewhere? The findings mention underperformance but provide no numbers or protocol. ([team findings](https://docs.google.com/document/d/1kilBxeR_R3_EPtA63CWHcPSqFdR6qLgfnSk4lgXw/edit?usp=sharing))
- What latency and memory use does the model achieve on the controller at its target frequency? ([quick start](../../archive/2025-26/docs/tcn_quick_start.md#L23))
- Should missing target values be masked instead of changed to zero? ([dataset preprocessing](../../archive/2025-26/src/exoskeleton_ml/data/datasets.py#L295))
