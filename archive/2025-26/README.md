> Snapshot of [McMaster-Exoskeleton/exoskeleton-ai](https://github.com/McMaster-Exoskeleton/exoskeleton-ai/tree/530135aa13c779df916afd5a1f9c8640b3281f12), kept to reproduce last year's baseline. Only changes needed to run it are made here (see [CONTRIBUTING.md](../../CONTRIBUTING.md)).

# Exoskeleton AI

Machine learning system for task-agnostic movement recognition for the McMaster Exoskeleton Team [MacExo](macexo.com)

## Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/)
- Git

### Installation

```bash
cd archive/2025-26
uv sync --locked
uv run pytest
```

uv downloads the required Python 3.10 interpreter and creates an isolated `.venv` automatically.
The default test command runs the offline suite. Tests that require authenticated access to the
private Hugging Face dataset are opt-in:

```bash
uv run pytest -m integration
```

Continue with [REPRODUCE.md](REPRODUCE.md) to download the data, train the TCN, and evaluate it.
