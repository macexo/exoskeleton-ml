# McMaster Exoskeleton - Machine Learning

Machine learning code for the McMaster Exoskeleton project. This repository contains the data pipeline, models, training, and real-time inference code for predicting joint torque in a soft exoskeleton, targeting a Raspberry Pi 5 that communicates with the STM32 boards, IMUs, and motors over CAN.

Last year's code (2025/26) is kept for reference in [`archive/2025-26/`](archive/2025-26/).

## Getting Started

This project uses [uv](https://docs.astral.sh/uv/) to manage Python and dependencies, so you don't need to install Python separately.

1. Install uv by following the [official instructions](https://docs.astral.sh/uv/getting-started/installation/).
2. Clone the repo and install dependencies:

   ```bash
   git clone https://github.com/macexo/exoskeleton-ml.git
   cd exoskeleton-ml
   uv sync
   ```

3. Run the tests to check your setup:

   ```bash
   uv run pytest
   ```

Run any script with `uv run <script>`. You don't need to activate a virtual environment.

## Project Structure

The structure below is planned and will be filled in as work progresses.

```
exoskeleton-ml/
├── src/exo_ml/                  <-- Main Python package
│   ├── data/                    # Dataset loading and preprocessing
│   ├── models/                  # Model architectures
│   ├── training/                # Training loop and utilities
│   ├── evaluation/              # Metrics and evaluation
│   └── inference/               # Real-time inference on the Raspberry Pi
│
├── configs/                     # Training and experiment configuration
├── scripts/                     # Entry points (train, evaluate, export, run inference)
├── tests/                       # Automated tests
├── docs/                        # Design docs, interface specs, research notes
├── sandbox/                     <-- Onboarding practice projects (one folder per member)
├── archive/
│   └── 2025-26/                 <-- Last year's code, kept for reference (not actively developed)
│
├── CONTRIBUTING.md              # Contribution guidelines and coding standards
└── README.md                    # This file
```

## Active Projects

### sandbox -- Onboarding

Practice projects to introduce members to PyTorch, Git, and uv. Each new member adds a folder with a small PyTorch training script and a short README.

**Quick start:**

1. Create your own folder: `sandbox/<your-name>/`.
2. Follow your onboarding issue on the project board.
3. Open a pull request following [CONTRIBUTING.md](CONTRIBUTING.md).

### archive/2025-26 -- Reproducing Last Year's Baseline

Last year's TCN torque prediction pipeline, trained on the Georgia Tech dataset. Before starting this year's model, all members will get the old repo running, reproducing its results as a baseline:

- **Returning members** fix the archive so it runs (setup, bug fixes, docs). These are the only changes made to `archive/2025-26/`.
- **Everyone reproducing the baseline** runs the archive code as-is and records their results in `sandbox/<your-name>/`.

New work goes in `src/exo_ml/`.

**Quick start (reproducing):**

1. Go to the folder and install its dependencies: `cd archive/2025-26 && uv sync`.
2. Follow `archive/2025-26/REPRODUCE.md` to train and evaluate the baseline.
3. Record your results in `sandbox/<your-name>/reproduction.md`, following your reproduction issue.

### torque-prediction -- Real-Time Torque Model (upcoming)

This year's model for predicting joint torque from IMU and motor data in real time on the Raspberry Pi 5. Work starts in `src/exo_ml/` once the baseline is reproduced, reusing what worked from last year.

## Prerequisites

- **uv** -- for Python and dependency management
- **Git** -- for version control ([Learn Git Branching](https://learngitbranching.js.org/) if you're new)
- **NVIDIA GPU or Apple Silicon Mac** -- optional, speeds up training

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.
