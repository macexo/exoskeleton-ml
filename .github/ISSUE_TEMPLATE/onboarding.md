---
name: Onboarding
about: Practice project for a new member (PyTorch, Git, and uv)
title: "Onboarding: <your-name>"
labels: []
assignees: []
---

Welcome! This issue walks you through your first contribution. Post questions here as you go.

## Steps

- [ ] Complete the setup in the [README](https://github.com/macexo/exoskeleton-ml/blob/main/README.md) (`uv sync`, then `uv run pytest`).
- [ ] Create a branch: `git checkout -b sandbox/<your-name>`.
- [ ] Create your folder: `sandbox/<your-name>/`.
- [ ] Write a small PyTorch training script (e.g., a classifier on a toy dataset).
- [ ] Add a short `README.md` in your folder: what it does and how to run it.
- [ ] Run `uv run ruff format .`, `uv run ruff check .`, and `uv run pytest`.
- [ ] Open a pull request following [CONTRIBUTING.md](https://github.com/macexo/exoskeleton-ml/blob/main/CONTRIBUTING.md) and request a review.

## Notes

- Only edit files inside your own folder.
- PyTorch, torchvision, NumPy, and Matplotlib are already installed. Ask if you need other packages installed, before running `uv add`.