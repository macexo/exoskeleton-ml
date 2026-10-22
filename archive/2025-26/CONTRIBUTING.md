# Contributing to Exoskeleton AI

## Getting Started

1. Clone the repository `git clone git@github.com:macexo/exoskeleton-ml.git`
2. Enter the archive and install the locked dependencies with uv:
   ```bash
   cd exoskeleton-ml/archive/2025-26
   uv sync --locked
   ```

## Development Workflow

### 1. Create a Branch

Create a feature branch for your work:

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write clean, readable code
- Follow the code style guidelines (see below)
- Add type hints to all functions
- Write docstrings for all public functions and classes
- Add tests for new functionality

### 3. Test Your Changes

Before committing, ensure all tests pass:

```bash
# Run the offline tests
uv run pytest

# Run private-data integration tests after authenticating with Hugging Face
uv run pytest -m integration

# Check code formatting
uv run black --check src/ tests/

# Run linters
uv run ruff check src/ tests/
uv run mypy src/
```

### 4. Commit Your Changes

- use an IDE or `git commit -m "commit message"`
- feel free to read this for details on how to write more structured and clean commit messages. [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear title and description
- Reference to any related issues
- Summary of changes
- Testing performed

## Code Style Guidelines

### Python Code Style

- **Line length**: 100 characters
- **Formatting**: Use Black for code formatting
- **Import sorting**: Handled by Ruff
- **Type hints**: Required for all function signatures
- **Docstrings**: Google style format

Example:

```python
def process_signals(
    signals: np.ndarray,
    normalize: bool = True,
    remove_mean: bool = True,
) -> torch.Tensor:
    """Preprocess signal data.

    Args:
        signals: Input signals of shape (n_samples, n_features).
        normalize: Whether to normalize signals to unit variance.
        remove_mean: Whether to remove mean from signals.

    Returns:
        Preprocessed signals as PyTorch tensor.

    Raises:
        ValueError: If signals have invalid shape.
    """
    # Implementation here
    pass
```

### Testing

- Write tests for all new features when applicable

Example:

```python
def test_preprocess_signals_normalization():
    """Test that signal normalization works correctly."""
    # Arrange
    signals = np.random.randn(100, 10)

    # Act
    processed = preprocess_signals(signals, normalize=True)

    # Assert
    assert torch.allclose(processed.std(dim=0), torch.ones(10), atol=1e-6)
```
