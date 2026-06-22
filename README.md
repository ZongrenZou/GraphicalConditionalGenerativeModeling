<!-- PAPER_LINK: update this URL when the paper is published -->
[PAPER_LINK]: https://arxiv.org/abs/2606.16219

# Graphical Conditional Generative Modeling

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![JAX](https://img.shields.io/badge/JAX-0.6.2-FFA500.svg)](https://github.com/google/jax)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-EE4C2C.svg)](https://pytorch.org)
[![Paper](https://img.shields.io/badge/Paper-arXiv-B31B1B?logo=arxiv&logoColor=white)][PAPER_LINK]

Source code for the paper [*Graphical Conditional Generative Modeling for Digital Twin Modeling*][PAPER_LINK].

## Quick start

```bash
pip install -e .
cd examples/ex1_lorenz_63
python flow_distill_prune.py
```

Run scripts from within each example directory so relative paths to `data/`, `checkpoints/`, and `figs/` resolve correctly.

To run every primary example (ex1–ex7):

```bash
python scripts/run_all_examples.py
```

These are full reproduction runs and may take hours or days.

## Installation

```bash
pip install -e .
```

All runtime dependencies are declared in `pyproject.toml`.

For the Reacher example (ex6) with GPU support, install the pinned CUDA PyTorch wheel after the editable install:

```bash
pip install torch==2.8.0.dev20250506+cu128 --index-url https://download.pytorch.org/whl/nightly/cu128
```

## Package layout

| Path | Description |
|------|-------------|
| [`src/gcm/`](src/gcm/) | Core library: CFM training, kernel pruning, checkpoint I/O |
| [`examples/`](examples/) | Paper experiments (ex1–ex7) |

## Examples

| Example | Domain | Entry script |
|---------|--------|--------------|
| [ex1_lorenz_63](examples/ex1_lorenz_63/) | Stochastic Lorenz-63 | `flow_distill_prune.py` |
| [ex2_multiscale](examples/ex2_multiscale/) | Multiscale Lorenz-96 | `flow_distill_prune.py` |
| [ex3_pendulum](examples/ex3_pendulum/) | Stochastic pendulum + SAC | `flow_distill.py` |
| [ex4_heat_equation](examples/ex4_heat_equation/) | 1D heat equation + MPC | `flow_distill_prune.py` |
| [ex5_lunar_lander](examples/ex5_lunar_lander/) | Lunar lander impulses + SAC | `flow_distill_prune.py` |
| [ex6_reacher](examples/ex6_reacher/) | Reacher robot + PETS | `flow_distill_prune.py` |
| [ex7_gpr_oil](examples/ex7_gpr_oil/) | GPR + WTI oil prices | `flow_distill_prune_all.py` |


## Cite us
```
@article{zou2026graphical,
  title={Graphical conditional generative modeling for digital twin modeling},
  author={Zou, Zongren and Bourdais, Th{\'e}o and Baptista, Ricardo and Owhadi, Houman},
  journal={arXiv preprint arXiv:2606.16219},
  year={2026}
}
```