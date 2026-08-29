# BaCP — Backbone Contrastive Pruning

A static-pruning research framework: prune a **pretrained** network to extreme
sparsity while preserving its representations with a contrastive objective
(PrC / SnC / FiC + CE, after CAP — Xu et al., AAAI 2022, arXiv 2112.07198 —
applied to vision backbones), and compare against iterative-pruning baselines
(magnitude, SNIP-it, WANDA) at **0.95 / 0.97 / 0.99** sparsity.

Every equation the code computes is documented with its source in
[`docs/foundation.md`](docs/foundation.md). The full citation audit is
[`docs/citations.md`](docs/citations.md).

## Layout

```
project/
├── models/               ResNet (He et al. 2016), VGG (Simonyan & Zisserman 2015)
├── model_factory.py      registry: resnet34/50/101, vgg11/19, vit-tiny/small,
│                         distilbert, roberta; records init_source provenance
├── dataset_factory.py    CIFAR-10 (+ registered others), SST-2
├── pruning_factory.py    magnitude / SNIP-it / WANDA (+ RigL/EAST kept as
│                         cited, unused references); masks, scope, schedules
├── loss_functions.py     CAP contrastive functional, SupCon, NT-Xent, KD
├── bacp.py               BaCPTrainer: the four-term objective
├── trainer.py            plain Trainer (dense + iterative pruning)
├── runner.py             cell -> argv -> subprocess -> run record
├── results.py            run records, provenance, checkpoint resolution
├── scripts/              baseline_script / pruning_script / bacp_script (CLI)
├── tests/                the invariant suite (pytest)
└── test_notebooks/       what you actually run -- see its README
```

## Install

Python **3.10–3.12**. The ceiling is `torchvision==0.20.1`: it publishes cp39–
cp312 wheels and no sdist, so on 3.13 pip fails with an error naming
torchvision, not Python. The floor is `scipy==1.14.1` (test suite only,
`Requires-Python >=3.10`); training alone would run back to 3.9. The reference
environment is 3.12.10.

The two requirement files are alternatives, not layers — they pin different
torch builds, so use one per virtualenv.

**Training** — `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

PyPI's torch wheel only carries CUDA on Linux/x86_64, where it pulls the
`nvidia-*-cu12` 12.4 runtime as a dependency. The Windows wheel is CPU-only and
macOS has no CUDA build at all, so a Windows GPU box takes torch from PyTorch's
own index first and the rest afterwards:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

`nb_common.setup()` prints `cuda=` and the GPU name; check it before starting a
real cell, because a CPU-only torch does not fail — it just trains on the CPU.

**Tests only** — `requirements-dev.txt`. No GPU, no training extras (no wandb,
no fvcore); it pins the `+cpu` torch builds and adds pytest, pyflakes and scipy:

```bash
pip install -r requirements-dev.txt
```

## Running

Everything runs from `project/test_notebooks/` (see its README): a smoke
notebook that pushes every model through dense → prune → BaCP in minutes, a
dataset-staging notebook for DBFS, and one notebook per model under
`resnet/ vgg/ vit/ bert/`. Training output streams live into each cell; a run
is complete iff its record exists under `results/runs/`, so everything is
resumable and idempotent.

Protocols follow the paper's appendix (Tables 1–3); they live in one place,
`project/test_notebooks/nb_common.py`.

## Verifying

```bash
python -m pytest -q
```
