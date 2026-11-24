# 2025–26 Data Pipeline Review
 
Review for [Issue #20](https://github.com/macexo/exoskeleton-ml/issues/20).
 
**Scope:** `archive/2025-26/`  
**Contributors:** Manasvi Bandi and AadiCodes4
 
## Summary
Last year's data pipeline takes raw exoskeleton recordings, converts them into a Hugging Face dataset, and prepares them for PyTorch training. It handles downloading, caching, normalization, and batching trials of different lengths. Overall, the pipeline has a good structure, but we found a few areas that could be improved, especially around missing values, cache management, and data validation.
 
## How it works
 
 
### Data preparation and preprocessing
 
Data preparation happens once, offline, in [`prepare_hf_dataset.py`](../../archive/2025-26/scripts/prepare_hf_dataset.py). Preprocessing happens every time we load the data for training, in [`datasets.py`](../../archive/2025-26/src/exoskeleton_ml/data/datasets.py).
 
**CSV conversion.** Each trial in the raw Phase 1 data has three CSV files.
 
- `*_exo.csv` contains the IMU readings.
- `*_angle_filt.csv` contains the filtered joint angles.
- `*_moment_filt_bio.csv` contains the biological joint moments, already normalized by body mass (Nm/kg).
The [`process_trial()`](../../archive/2025-26/scripts/prepare_hf_dataset.py#L87) function loads all three files and skips the trial if any file is missing, if the three files have different lengths, or if a required column is missing. It then keeps 24 IMU features (accelerometer and gyroscope for the left and right thigh and shank), 4 joint angles, and 4 joint moments. Foot IMUs and insole data are left out. Each trial is also tagged with the participant's body mass, which is hardcoded from the published paper.
 
**Hugging Face upload.** The processed trials are combined into a Hugging Face dataset and uploaded to [`MacExo/exoData`](https://huggingface.co/datasets/MacExo/exoData). This means team members can download a clean dataset instead of reprocessing the raw CSVs themselves.
 
**Preprocessing.** When the dataset is loaded, [`_preprocess_and_cache()`](../../archive/2025-26/src/exoskeleton_ml/data/datasets.py#L142) converts each trial to `float32` tensors and joins the IMU features and joint angles into a single `(T, 28)` input tensor. The 4 moments become the `(T, 4)` target tensor. Each trial is then saved as a `.pt` file (see Downloading and caching below).
 
Note that [`preprocessing.py`](../../archive/2025-26/src/exoskeleton_ml/data/preprocessing.py) only contains empty placeholder functions. All of the real preprocessing happens inside `datasets.py`.
 
**NaN handling.** NaN values are handled in two places, and neither one removes bad data.
 
- In `prepare_hf_dataset.py`, NaN checks only run when the `--validate` flag is passed, and even then they [only print a warning](../../archive/2025-26/scripts/prepare_hf_dataset.py#L164). Trials with NaNs are still uploaded.
- In `datasets.py`, [`__getitem__()`](../../archive/2025-26/src/exoskeleton_ml/data/datasets.py#L298) replaces every NaN in both the inputs and the targets with 0 using `torch.nan_to_num`.
This conflicts with [`phase1.yaml`](../../archive/2025-26/configs/data/phase1.yaml), which sets `remove_invalid: true` (remove trials with NaN values). That setting is never used in the code.
 
**Normalization.** Each of the 28 input channels and 4 target channels is standardized as `(x - mean) / std`. The statistics are computed in [`_compute_normalization_stats()`](../../archive/2025-26/src/exoskeleton_ml/data/datasets.py#L196) using `nanmean` and a NaN-aware std, so missing values don't affect the statistics. They are computed from the training participants only, saved to `normalization_stats.json`, and [reused for validation and test](../../archive/2025-26/src/exoskeleton_ml/data/datasets.py#L464).
 
One detail is that the statistics ignore NaNs, but the NaNs are then replaced with 0 *before* normalization. After normalization, a missing value becomes `-mean / std` instead of 0, so it can look like a real reading to the model.
 
### Downloading and caching
 
The [`download.py`](../../archive/2025-26/src/exoskeleton_ml/data/download.py) file handles downloading the dataset from Hugging Face. Before downloading, it checks whether a local copy already exists. If the cache is available, it loads the dataset from disk instead of downloading it again.
 
The pipeline uses two levels of caching:
 
1. **Hugging Face dataset cache:** Stores the downloaded dataset in `data/processed/phase1/hf_cache/`, along with a `metadata.json` file containing information such as participant IDs, trial counts, and download date.
2. **Preprocessed tensor cache:** The [`datasets.py`](../../archive/2025-26/src/exoskeleton_ml/data/datasets.py#L123) file converts individual trials into PyTorch tensors and saves them as `.pt` files. An `index.pkl` file keeps track of the saved trials.
The tensor cache uses a hash based on the selected participant IDs. If the corresponding cache already exists, the pipeline reuses it unless reprocessing is explicitly requested.
 
This helps avoid downloading and converting the same data every time we run an experiment.
 
### Dataset loading and batching
 
The [`ExoskeletonDataset`](../../archive/2025-26/src/exoskeleton_ml/data/datasets.py) class converts each trial into PyTorch tensors.
 
For a trial with `T` timesteps:
 
- Inputs have shape `(T, 28)`.
- Targets have shape `(T, 4)`.
The dataset is split by participant, with 11 participants used for training, one for validation, and three for testing. These splits are defined in [`phase1.yaml`](../../archive/2025-26/configs/data/phase1.yaml).
 
Since trials have different lengths, the DataLoader uses a custom [`collate_fn()`](../../archive/2025-26/src/exoskeleton_ml/data/datasets.py#L315) to make batching possible.
 
It finds the longest trial in each batch and pads the shorter ones with zeros. It also creates a Boolean mask to identify which timesteps contain real data and which ones are padding.
 
The training code uses this mask when calculating the loss, so padded timesteps are excluded. However, the mask does not identify missing target values that were already replaced with zeros.
 
### Testing
 
The repository includes [`test_dataloader.py`](../../archive/2025-26/scripts/test_dataloader.py) and additional [automated tests](../../archive/2025-26/tests/test_data/test_dataloader.py).
 
These cover areas such as dataset loading, batch dimensions, padding, masking, and metadata.
 
We found that some tests are more thorough than others. For example, the padding tests check that padded positions contain zeros and that the masks are correct. However, the shuffling and normalization tests don't fully verify the behaviors they're named after.
 
## Advantages
 
- **Two-level caching:** Saving both the Hugging Face dataset and preprocessed tensors avoids repeating downloads and conversions during experiments. ([download.py](../../archive/2025-26/src/exoskeleton_ml/data/download.py), [datasets.py](../../archive/2025-26/src/exoskeleton_ml/data/datasets.py))
- **Participant-based splits:** Separating participants between training, validation, and testing helps us evaluate how well the model works on people it hasn't seen before. ([phase1.yaml](../../archive/2025-26/configs/data/phase1.yaml))
- **Variable-length support:** Dynamic padding lets us work with trials of different lengths without having to cut every trial down to a fixed size. ([datasets.py](../../archive/2025-26/src/exoskeleton_ml/data/datasets.py#L315))
- **Consistent normalization:** The same training statistics are used for validation and testing, reducing the risk of data leakage. ([datasets.py](../../archive/2025-26/src/exoskeleton_ml/data/datasets.py#L456))
- **Existing tests:** The repository already has tests for padding, masking, tensor shapes, and dataset loading, giving us a useful starting point for future improvements. ([DataLoader tests](../../archive/2025-26/tests/test_data/test_dataloader.py))
## Disadvantages
 
- **Cache invalidation:** The preprocessed cache key only depends on participant IDs. If the original dataset or preprocessing code changes, we could end up reusing outdated tensors. The configuration also mentions invalidating the cache when settings change, but this isn't reflected in the cache key. ([datasets.py](../../archive/2025-26/src/exoskeleton_ml/data/datasets.py#L123), [phase1.yaml](../../archive/2025-26/configs/data/phase1.yaml))
- **Limited integrity checks:** The download code checks that the required fields exist, but it only performs detailed feature-dimension checks on the first trial. This means issues in later trials could go unnoticed. ([download.py](../../archive/2025-26/src/exoskeleton_ml/data/download.py))
- **Missing values:** Replacing NaN targets with zeros could cause the model to learn from values that weren't actually recorded. It would be better to distinguish missing labels from genuine zeros. ([datasets.py](../../archive/2025-26/src/exoskeleton_ml/data/datasets.py#L295))
- **Padding efficiency:** Since batches are padded to their longest trial, combining very short and very long trials can waste memory and computation. Grouping trials with similar lengths could make batching more efficient. ([datasets.py](../../archive/2025-26/src/exoskeleton_ml/data/datasets.py#L315))
- **Configuration inconsistencies:** Some options in `phase1.yaml`, such as configurable padding values and normalization methods, aren't reflected in the dataset implementation we reviewed. We should confirm which settings are actually being used. ([phase1.yaml](../../archive/2025-26/configs/data/phase1.yaml), [datasets.py](../../archive/2025-26/src/exoskeleton_ml/data/datasets.py))
- **Sampling rate and units:** The original data is recorded at 200 Hz, while parts of the interface documentation assume 100 Hz. There are also inconsistencies around whether joint angles are represented in degrees or radians. These need to be clarified before integrating the pipeline with this year's hardware. ([Issue #20](https://github.com/macexo/exoskeleton-ml/issues/20), [interface specification](../../archive/2025-26/docs/ml_control_interface.md))
- **Validation only warns:** NaN checks during data preparation are opt-in and never remove trials, and the `remove_invalid` option in the config is not implemented. ([prepare_hf_dataset.py](../../archive/2025-26/scripts/prepare_hf_dataset.py#L164), [phase1.yaml](../../archive/2025-26/configs/data/phase1.yaml))
## Open questions
 
- How common are NaNs across the dataset, and are certain trials or participants affected more than others?
- Should missing target values be masked out instead of being replaced with zeros?
- Are the saved normalization statistics consistently applied during inference?
- Which configuration settings in `phase1.yaml` are actually used by the training scripts?
- Would grouping trials of similar lengths improve training efficiency?
- How should we invalidate cached data when the Hugging Face dataset or preprocessing code changes?
- What sampling rate and angle units should we standardize on for this year's hardware?
- Should NaNs be filled after normalization (so they become 0) or masked, instead of filled before?
## Training results
 
We ran the archived training script for 1 epoch (`uv run scripts/train_tcn.py training.num_epochs=1`) on Nov 24, 2026, using the default participant split (11 train, 1 validation, 3 test).
 
| Metric | Result |
|---|---|
| Model | TCN (45,604 parameters) |
| Epochs | 1 |
| Test RMSE | 0.6360 |
| Test MAE | 0.4425 |
| Test R² | 0.4160 |
 
Per-joint test RMSE was 0.6212 (hip L), 0.6223 (hip R), 0.6732 (knee L) and 0.6260 (knee R).
 
The script labels RMSE and MAE as Nm/kg, but the targets are standardized before training and the metrics are never converted back, so these values are in standardized units, not Nm/kg. R² is not affected.
 
For comparison, last year's run of about 80 epochs reached a validation R² of 0.7311. Our lower score is expected after only 1 epoch.
 
