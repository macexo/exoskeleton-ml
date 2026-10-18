# Contributing to McMaster Exoskeleton - Machine Learning

This guide covers how we track work, write code, and get it merged.

## Before you start

- Complete the setup steps in the [README](README.md).
- New to GitHub? Read the team's GitHub guide (pinned in Discord) for key terms and how the project board works.
- New to Git? Work through [Learn Git Branching](https://learngitbranching.js.org/) first.
- Every piece of work starts from an **issue** on the project board. If there's no issue for what you want to do, ask a lead to create one or create it yourself.

## Where your code goes

| Folder | What goes there |
|---|---|
| `src/exo_ml/` | This year's code: data, models, training, evaluation, inference |
| `sandbox/<your-name>/` | Onboarding practice projects and baseline reproduction results. Only edit your own folder. |
| `archive/2025-26/` | Last year's code, fixed up by returning members. **Only** changes needed to run and reproduce it (setup, bug fixes, docs). No new features. |

Sandbox code uses the main project's environment. Run it with `uv run` (e.g., `uv run sandbox/your-name/train.py`) and don't create a separate uv project in your folder.

The archive is its own uv project. Run commands for it from inside that folder (e.g., `cd archive/2025-26 && uv run pytest`).

If you want to reuse something from the archive, copy or rewrite it into `src/exo_ml/` in a separate PR rather than importing from `archive/`.

## Workflow

### 1. Pick up an issue

Find an issue on the project board, assign yourself, and move it to **In Progress**.

### 2. Create a branch

Branch from the latest version of `main`:

```bash
git checkout main
git pull
git checkout -b feature/your-feature-name
```

Name your branch after what you're doing, using one of these prefixes:

| Prefix | Use for | Example |
|---|---|---|
| `feature/` | New functionality | `feature/imu-dataloader` |
| `fix/` | Bug fixes | `fix/gyro-units` |
| `docs/` | Documentation only | `docs/setup-guide` |
| `research/` | Experiments, notebooks, exploratory work | `research/graph-nn` |
| `chore/` | Setup, dependencies, config, and other housekeeping | `chore/add-ci` |
| `sandbox/` | Onboarding practice projects | `sandbox/your-name` |

If none of these fit (e.g., a test-only or refactor PR), use the closest one.

### 3. Make your changes

Keep your changes focused on your issue. Small pull requests are easier to review and get merged faster.

If you need a new package, add it with uv (this updates `pyproject.toml` and `uv.lock`, which you should commit):

```bash
uv add package-name
```

Add type hints to all function signatures and docstrings to public functions (Google style, see below). Add tests for new functionality where it makes sense.

### 4. Check your work

Run these before committing:

```bash
uv run ruff format .     # Format code
uv run ruff check .      # Lint
uv run pytest            # Run tests
```

### 5. Commit

```bash
git add .
git status
git commit -m "feat: add IMU dataloader"
```

Before committing, check that `git status` only lists the files you meant to change under "Changes to be committed".

We use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/): start each message with a type, a colon, and a short description in lowercase.

| Type | Use it for | Example |
|---|---|---|
| `feat` | New code or functionality | `feat: add IMU dataloader` |
| `fix` | Fixing a bug | `fix: correct gyro units in preprocessing` |
| `docs` | Documentation only | `docs: add CAN message spec` |
| `test` | Adding or updating tests | `test: add windowing shape tests` |
| `refactor` | Restructuring code without changing what it does | `refactor: split trainer into smaller functions` |
| `research` | Experiments, notebooks, exploratory work | `research: compare TCN and LSTM on hip torque` |
| `chore` | Setup, dependencies, config, and other housekeeping | `chore: add ruff to dev dependencies` |

Commit as often as you like; commits are combined into one when your PR is merged.

You can also commit from the **Source Control** panel in VS Code instead of the terminal.

### 6. Open a pull request

```bash
git push -u origin feature/your-feature-name
```

Then go to the repo on GitHub. You'll see a prompt to **Compare & pull request**; click it and fill in the template:

- Give the PR a Conventional Commits title (e.g., `feat: add IMU dataloader`). When it's squash merged, this title becomes the commit message on `main`.
- Write `Closes #12` (using your issue number) so the issue closes automatically when your PR is merged.
- Describe what you changed and how you tested it.

Not finished but want feedback? Open it as a **draft pull request**.

### 7. Review and merge

- Every PR needs **one approval from a lead** before it can be merged.
- Respond to every review comment, either with a change or a reply.
- Once approved and CI passes, merge with **Squash and merge** and delete your branch.

### 8. Clean up

Switch back to `main` and get the latest changes before starting your next task:

```bash
git checkout main
git pull
```

## Code style

- **Formatting and linting:** Ruff, line length 100 (configured in `pyproject.toml`).
- **Type hints:** required on all function signatures, including tests.
- **Docstrings:** Google style.

```python
def window_signals(
    signals: np.ndarray,
    window_size: int,
    stride: int = 1,
) -> np.ndarray:
    """Split a signal into overlapping windows.

    Args:
        signals: Input array of shape (n_samples, n_features).
        window_size: Number of samples per window.
        stride: Step between consecutive windows.

    Returns:
        Array of shape (n_windows, window_size, n_features).

    Raises:
        ValueError: If window_size is larger than n_samples.
    """
```

## Tests

Use pytest and keep tests in `tests/`, mirroring the `src/` layout. Structure each test as arrange, act, assert:

```python
def test_window_signals_shape() -> None:
    """Windowing produces the expected output shape."""
    signals = np.random.randn(100, 6)

    windows = window_signals(signals, window_size=10, stride=5)

    assert windows.shape == (19, 10, 6)
```

## Rules that keep the repo healthy

- **Never commit data, model weights, or large files.** Datasets live on Hugging Face; checkpoints and outputs stay local or in shared storage. `data/`, `outputs/`, and weight files are already in `.gitignore`.
- **Never commit secrets** such as Hugging Face or API tokens. Use environment variables or a local `.env` file (ignored by Git).
- **Never push directly to `main`.** All changes go through a pull request.
- **Code that commands motors needs review from embedded/controls** before it runs on hardware.

## Need help?

- If you're stuck for **30 minutes**, ask for help in the issue or the Discord chat. Include what you tried and the full error message.
- Post a short weekly update on your issue: **done**, **next**, **blocked**.
- Unsure about Git? Ask a lead or returning member. Almost everything in Git can be undone or fixed; that's the whole purpose of version control.
