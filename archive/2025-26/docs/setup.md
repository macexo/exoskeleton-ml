# Setup Guide

This guide sets up the archived Exoskeleton AI project with uv.

## Prerequisites

Install the following first:

- [uv](https://docs.astral.sh/uv/)
- Git

## Step-by-Step Setup

### 1. Open the archived project

```bash
git clone https://github.com/macexo/exoskeleton-ml.git
cd exoskeleton-ml/archive/2025-26
```

### 2. Install Python and dependencies

```bash
uv sync --locked
```

uv downloads the required Python 3.10 interpreter, creates `.venv`, and installs both the package
and its development dependencies. There is no virtual environment to activate; prefix project
commands with `uv run`.

### 3. Verify the setup

```bash
uv run pytest
```

The default suite is offline. After obtaining access to `MacExo/exoData` and authenticating with
Hugging Face, run the private-data integration tests explicitly:

```bash
uv run pytest -m integration
```

See [REPRODUCE.md](../REPRODUCE.md) for the training and evaluation workflow.
