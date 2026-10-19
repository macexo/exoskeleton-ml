>Snapshot of [McMaster-Exoskeleton/exoskeleton-ai](https://github.com/McMaster-Exoskeleton/exoskeleton-ai/tree/530135aa13c779df916afd5a1f9c8640b3281f12), kept to reproduce last year's baseline. Only changes needed to run it are made here (see [CONTRIBUTING.md](../../CONTRIBUTING.md)). The setup steps below are outdated: run `uv sync` and follow [REPRODUCE.md](REPRODUCE.md).
# Exoskeleton AI

Machine learning system for task-agnostic movement recognition for the McMaster Exoskeleton Team [MacExo](macexo.com)
## Setup

### Prerequisites

- Python 3.10 or higher
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/exoskeleton-ai.git
cd exoskeleton-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package with development dependencies
make install-dev
```