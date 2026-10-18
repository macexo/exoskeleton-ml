---
name: Baseline reproduction
about: Reproduce last year's TCN baseline and record results in your sandbox folder
title: "Reproduce baseline: <your-name>"
labels: []
assignees: []
---

Run last year's code as-is and record your results in your sandbox folder. If something there is broken, open a **Bug report** issue so a returning member can fix it.

## Steps

- [ ] Create a branch: `git checkout -b sandbox/<your-name>-reproduction`.
- [ ] Install the archive's dependencies: `cd archive/2025-26 && uv sync`.
- [ ] Follow `archive/2025-26/REPRODUCE.md` to train and evaluate the baseline.
- [ ] Create `sandbox/<your-name>/reproduction.md` using the format below.
- [ ] Open a pull request following [CONTRIBUTING.md](https://github.com/macexo/exoskeleton-ml/blob/main/CONTRIBUTING.md).

## Format for `reproduction.md`

    ## Results

    | Metric | My result | Last year's result |
    |---|---|---|
    | | | |

    ## Environment

    - OS:
    - GPU / Apple Silicon / CPU:
    - Training time:

    ## Differences from the instructions

    Anything you had to change to get it running.

## Questions

<!-- Post questions here or in comments as you go. -->