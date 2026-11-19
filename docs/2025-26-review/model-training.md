# 2025–26 Model and Training Review

## Summary

The 2025–26 pipeline trains a causal temporal convolutional network (TCN) to map 28 normalized IMU and joint-angle inputs to four joint-moment predictions at each timestep. Training uses padded variable-length trial batches, masked mean-squared error, Adam, validation-loss scheduling and early stopping; the main config is composed with Hydra. ([training script](../../archive/2025-26/scripts/train_tcn.py#L198), [TCN](../../archive/2025-26/src/exoskeleton_ml/models/tcn.py#L246), [default config](../../archive/2025-26/configs/config.yaml#L3))

## How it works

### Model path

| Step | What happens | Source |
|---|---|---|
| Inputs | The data loader provides 24 IMU channels and four joint angles. It standardizes inputs and targets using dataset statistics and pads each batch to its longest trial, returning a boolean mask for real timesteps. | [dataset config](../../archive/2025-26/configs/data/phase1.yaml), [dataset loading and collation](../../archive/2025-26/src/exoskeleton_ml/data/datasets.py#L295) |
| Causal temporal blocks | Features are transposed to channels-first. Each block applies two Conv1d layers with the same dilation, symmetric padding followed by a right-side “chomp,” ReLU, dropout, weight normalization, and a residual skip (1×1 convolution when channel widths differ). This retains one output position per input timestep while preventing access to future samples. | [TCN block](../../archive/2025-26/src/exoskeleton_ml/models/tcn.py#L46), [TCN stack and forward path](../../archive/2025-26/src/exoskeleton_ml/models/tcn.py#L179) |
| Predictions | A linear layer maps the final channels to four continuous moment estimates for every timestep; no output activation. | [TCN output layer](../../archive/2025-26/src/exoskeleton_ml/models/tcn.py#L284) |
| Loss and update | The training loop calculates masked squared error, backpropagates, clips the global gradient norm, and updates Adam. It validates each epoch, steps the scheduler, saves the best validation-loss model and periodically saves checkpoints. After training, it reloads the best model for test evaluation and writes the metric histories. | [loss and training step](../../archive/2025-26/scripts/train_tcn.py#L41), [optimizer and scheduler](../../archive/2025-26/scripts/train_tcn.py#L259), [epoch loop and checkpointing](../../archive/2025-26/scripts/train_tcn.py#L294) |

### Causality and receptive field

A causal convolution at time *t* must use inputs only through *t*. The implementation pads each convolution and removes the padded right tail with `Chomp1d`, leaving the left padding to represent the unavailable past at the beginning of a sequence. Dilation spaces kernel taps apart, so a sequence of blocks with dilations 1, 2, 4, 8, 16 expands the look-back range exponentially. Residual additions give each block a direct path around the convolutions, with a 1×1 projection when channel counts differ. ([`Chomp1d` and `TemporalBlock`](../../archive/2025-26/src/exoskeleton_ml/models/tcn.py#L17), [dilation schedule](../../archive/2025-26/src/exoskeleton_ml/models/tcn.py#L208))

The active default config has five blocks, kernel size 7, and dilations 1, 2, 4, 8, 16. Each block contains **two** convolutions, so the receptive field is:

```text
R = 1 + 2 × (kernel_size − 1) × Σ(dilation_i)
  = 1 + 2 × 6 × (1 + 2 + 4 + 8 + 16)
  = 373 samples = 3.73 s at 100 Hz
```

***NOTE***: The often-repeated 187-sample (1.87 s) figure counts one convolution per block. The factory reports that value because it computes `(kernel_size − 1) × (2^num_levels − 1) + 1`; the configured blocks actually have two convolutions each. The small config’s actual field is 121 samples and the large config’s is 1009 samples, versus their config comments’ 61 and 505. The model uses the calculated value only as a reported `eff_hist`; the convolution stack itself determines the true field. ([factory formula](../../archive/2025-26/src/exoskeleton_ml/models/model_factory.py#L33), [two convolutions per block](../../archive/2025-26/src/exoskeleton_ml/models/tcn.py#L82), [default config](../../archive/2025-26/configs/model/tcn.yaml#L13), [small config](../../archive/2025-26/configs/model/tcn_small.yaml#L13), [large config](../../archive/2025-26/configs/model/tcn_large.yaml#L13); independent calculation also recorded in the [team findings doc](https://docs.google.com/document/d/1kilBxeR_R3_EPtA63CWHcPSqFdR6qLgfnSk4lgX6Gxw/edit?usp=sharing))

### Why a TCN

The TCN was selected at the start of 2025–26 after the previous season’s LSTM underperformed.

TCN architecture is better suited for the task: causal convolutions support prediction from past sensor readings only, and convolution operations can process positions in parallel, unlike recurrent models that pass a hidden state step by step. The TCN paper reports TCN-versus-LSTM/GRU results on its own benchmarks. ([team findings doc](https://docs.google.com/document/d/1kilBxeR_R3_EPtA63CWHcPSqFdR6qLgfnSk4lgX6Gxw/edit?usp=sharing), [Bai, Kolter & Koltun, *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*](https://arxiv.org/abs/1803.01271), [repository model configs](../../archive/2025-26/configs/model/))

### Baseline and model sizes

The baseline model is not implemented. ([baseline implementation](../../archive/2025-26/src/exoskeleton_ml/models/baseline.py#L7))

### Loss, padding and missing values

The training script selects MSE. `collate_fn` right-pads each trial to the longest trial in that batch and sets the mask to true only for observed timesteps; `masked_loss` expands this mask over the four targets and divides the sum of valid squared errors by the number of valid elements. Thus padding does not contribute to the loss. The `criterion` argument passed to `masked_loss` is not used; the function computes squared error directly. ([loss function](../../archive/2025-26/scripts/train_tcn.py#L41), [criterion selection](../../archive/2025-26/scripts/train_tcn.py#L259), [padding and mask](../../archive/2025-26/src/exoskeleton_ml/data/datasets.py#L315))

There is a NaN-handling mismatch. `masked_loss` builds a validity mask for NaN predictions and targets, but computes `(outputs - targets) ** 2` before multiplying by that mask; IEEE arithmetic gives `NaN × 0 = NaN`, so this does not safely exclude NaNs. In the active dataset path, `__getitem__` first changes NaN inputs and targets to zero, which means missing target torques become zero-valued training labels rather than masked-out observations. Padding is handled; missing sensor/target data is not preserved as missing through this pipeline. ([NaN masking order](../../archive/2025-26/scripts/train_tcn.py#L61), [NaN replacement](../../archive/2025-26/src/exoskeleton_ml/data/datasets.py#L295))

### Optimizer, learning-rate schedule, clipping and early stopping

The default configuration requests Adam with learning rate 0.001 and weight decay 0.0001. The script always constructs Adam, reading the learning rate and weight decay values but not the `optimizer` name. After each validation epoch it calls `ReduceLROnPlateau` on validation loss, with factor 0.5, patience 10 and minimum learning rate 1e-6. Those scheduler settings are hard-coded: `training.scheduler.type`, factor, patience and min_lr in Hydra do not control the live scheduler. After backpropagation, the code clips the total gradient norm to 1.0; the configured `gradient_clip_norm` is likewise not read. ([training config](../../archive/2025-26/configs/config.yaml#L9), [optimizer/scheduler implementation](../../archive/2025-26/scripts/train_tcn.py#L259), [gradient clipping](../../archive/2025-26/scripts/train_tcn.py#L122))

**Metric-unit caveat:** the default data config enables standard normalization, and `__getitem__` applies the training-set mean and standard deviation to the four target moments. Validation then passes those normalized targets and model outputs directly into `RunningMetrics`; no inverse transform is applied before RMSE and MAE are printed with an `Nm/kg` label. Thus the archived training path appears to report standardized-unit errors, not physical Nm/kg. R² is also aggregated over the standardized joint channels. ([normalization enabled](../../archive/2025-26/configs/data/phase1.yaml#L103), [target normalization](../../archive/2025-26/src/exoskeleton_ml/data/datasets.py#L303), [metric calculation path](../../archive/2025-26/scripts/train_tcn.py#L164), [metrics use supplied tensors directly](../../archive/2025-26/src/exoskeleton_ml/utils/metrics.py#L37))

`EarlyStopping` watches validation loss in “min” mode, resets its counter on any strict decrease, and stops when the configured patience is reached; the default patience is 20 epochs and `min_delta` is zero. It runs after the scheduler step and checkpointing. The best model is saved independently whenever validation loss improves, then reloaded for test evaluation. Periodic checkpoints include scheduler state, but `best_model.pt` is written without it, so the best-model file is suitable for inference but is not a complete resume point for scheduler training state. ([early stopping utility](../../archive/2025-26/src/exoskeleton_ml/utils/early_stopping.py#L11), [training loop](../../archive/2025-26/scripts/train_tcn.py#L325), [checkpoint calls](../../archive/2025-26/scripts/train_tcn.py#L328), [checkpoint contents](../../archive/2025-26/src/exoskeleton_ml/utils/checkpointing.py#L11))

### Hydra configuration

`@hydra.main` loads `configs/config.yaml`; its `defaults` list composes `data/phase1.yaml` and `model/tcn.yaml`, then `_self_` applies the main training settings. Hydra passes the resulting OmegaConf `DictConfig` to `train(cfg)`. A config-group override such as `model=tcn_small` selects a different model file; dotted overrides such as `training.num_epochs=5` replace values for that invocation. `-m` asks Hydra to launch a multirun sweep. The script prints and saves the composed config into each output directory. Hydra’s documentation describes this defaults-list composition, command-line override syntax and multirun behavior. ([main config](../../archive/2025-26/configs/config.yaml#L3), [Hydra entry point and config save](../../archive/2025-26/scripts/train_tcn.py#L198), [quick-start override examples](../../archive/2025-26/docs/tcn_quick_start.md#L65), [Hydra 1.3 getting started](https://hydra.cc/docs/1.3/intro/))

***IMPORTANT***: Hydra can accept overrides for `training.optimizer`, `training.scheduler.*`, `training.gradient_clip_norm`, `seed` and `device`, but accepting an override does not make it effective. In this script the optimizer kind, scheduler kind/settings, clipping threshold, and random-seed setup are not read from those config fields; scheduler and clipping are **hard-coded**, and the code does not apply `cfg.seed` or `cfg.device`. ([config fields](../../archive/2025-26/configs/config.yaml#L15), [hard-coded training controls](../../archive/2025-26/scripts/train_tcn.py#L263))

### Training run evidence and optimization-plan status

For a five-epoch run on Google Colab with a T4 GPU, using `uv run scripts/train_tcn.py training.num_epochs=5`: test set is BT15–BT17; validation RMSE fell from 0.625 to 0.500, with learning rate 1e-3. ([output reference](https://docs.google.com/document/d/1kilBxeR_R3_EPtA63CWHcPSqFdR6qLgfnSk4lgX6Gxw/edit?usp=sharing), [split config](../../archive/2025-26/configs/data/phase1.yaml))

| Held-out test metric | Pair run, 5 epochs (reported value) | Historical figure in optimization plan |
|---|---:|---:|
| RMSE (Nm/kg) | 0.500 | 0.4316 |
| R² | 0.639 | 0.7311 |
| MAE (Nm/kg) | 0.348 | 0.3014 |
| Hip L / Hip R RMSE (Nm/kg) | 0.499 / 0.488 | 0.44 average |
| Knee L / Knee R RMSE (Nm/kg) | 0.520 / 0.493 | 0.42 average |

The optimization plan marks several experiments with checkmarks, but its results table contains numbers only for the historical baseline.  ([optimization phases](../../archive/2025-26/docs/optimization_plan.md#L268), [results table](../../archive/2025-26/docs/optimization_plan.md#L357))

## Advantages

- **Causal per-timestep prediction:** the chomped convolutions prevent use of future inputs, which fits a controller that must act from sensor history. ([causal padding implementation](../../archive/2025-26/src/exoskeleton_ml/models/tcn.py#L17), [TCN paper](https://arxiv.org/abs/1803.01271))
- **Parallel sequence processing:** the model applies convolutions across the time axis in parallel, avoiding the step-by-step recurrent dependency of an LSTM/GRU. This can improve throughput, though the team still needs to measure end-to-end latency on its target computer. ([TCN stack](../../archive/2025-26/src/exoskeleton_ml/models/tcn.py#L179), [TCN paper discussion](https://vladlen.info/papers/TCN.pdf))
- **Configurable context and capacity:** changing block count, channel width, kernel size and dropout gives a clear way to trade model capacity and temporal context; the factory constructs each TCN from its selected config. ([model factory](../../archive/2025-26/src/exoskeleton_ml/models/model_factory.py#L33), [model variants](../../archive/2025-26/configs/model/))
- **Best-checkpoint evaluation:** validation loss selects a checkpoint during training, and final test metrics are evaluated after loading that checkpoint. The pair’s short run shows steadily falling train loss and validation RMSE across five epochs. ([checkpoint selection and test evaluation](../../archive/2025-26/scripts/train_tcn.py#L328), [pair’s five-epoch results](https://docs.google.com/document/d/1kilBxeR_R3_EPtA63CWHcPSqFdR6qLgfnSk4lgX6Gxw/edit?usp=sharing))

## Disadvantages

- **Context is misreported:** the factory, all three TCN config comments and the implementation plan undercount receptive field by a factor of two because they count one convolution per residual block even though the code runs two. This can mislead embedded buffer sizing and any real-time history analysis. ([factory](../../archive/2025-26/src/exoskeleton_ml/models/model_factory.py#L33), [default config](../../archive/2025-26/configs/model/tcn.yaml#L20), [implementation plan formula](../../archive/2025-26/docs/tcn_implementation_plan.md#L102))
- **No baseline comparison:** the baseline model is not implemented. ([baseline implementation](../../archive/2025-26/src/exoskeleton_ml/models/baseline.py#L7))
- **Reported RMSE/MAE units are suspect:** default targets are standardized before training, and the metric implementation never converts predictions or labels back to Nm/kg, even though the training script prints that unit. The 0.500 pair-run value and historical 0.4316 value should not be treated as physical torque errors or compared with raw-unit scores until the exact run config and metric pipeline are confirmed. ([target normalization](../../archive/2025-26/src/exoskeleton_ml/data/datasets.py#L303), [metric computation](../../archive/2025-26/src/exoskeleton_ml/utils/metrics.py#L59), [metric output labels](../../archive/2025-26/scripts/train_tcn.py#L308))
- **Several tuning knobs are misleading:** Hydra overrides for scheduler settings and gradient clipping appear valid but have no effect; optimizer selection, seed and device config are also not applied by the shown training code. This makes experiment commands harder to interpret and reproduce. ([main config](../../archive/2025-26/configs/config.yaml#L15), [training setup](../../archive/2025-26/scripts/train_tcn.py#L263))
- **NaN targets are converted to zeros:** the dataset discards the missing-value signal before loss masking, while the mask’s multiplication would still propagate NaNs if any reached it. Either path risks learning incorrect values for missing torques. ([dataset NaN replacement](../../archive/2025-26/src/exoskeleton_ml/data/datasets.py#L298), [masked loss](../../archive/2025-26/scripts/train_tcn.py#L61))
