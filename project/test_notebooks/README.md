# The notebooks — static pruning, one model per notebook

```
00_smoke_test_all.ipynb    every model x (dense, prune, bacp), 2-epoch/2-batch
                           smoke; ends in a PASS/FAIL table. Run this FIRST on
                           any new cluster or machine.
01_datasets_to_dbfs.ipynb  DATABRICKS ONLY -- stage CIFAR-10, SST-2, tokenizers
                           and the ImageNet weights into /dbfs so nothing
                           re-downloads. Elsewhere see "Off Databricks" below;
                           only its ImageNet-weights cell matters, and only for
                           ResNet/VGG.
resnet/  resnet34/  resnet34.ipynb
         resnet50/  resnet50.ipynb
vgg/     vgg11/     vgg11.ipynb
         vgg19/     vgg19.ipynb        (BaCP LR 0.05)
vit/     vit_tiny/  vit_tiny.ipynb     (224px, hub weights)
         vit_small/ vit_small.ipynb
bert/    distilbert/ distilbert.ipynb  (SST-2)
         roberta/    roberta.ipynb     (SST-2, BaCP LR 1e-5)
nb_common.py               all protocols (FAMILIES), preflight, streaming
                           runner, results tables. The single place settings
                           live.
```

## Order

1. `01_datasets_to_dbfs` once per workspace — **Databricks only**; see below
   for what a non-Databricks machine needs instead.
2. `00_smoke_test_all` once per machine — every pipeline must PASS, on a
   brand-new machine too. Its cells are smoke cells, so they chain from the
   smoke dense checkpoint they just produced. A smoke checkpoint can seed
   another smoke cell and nothing else: it can never satisfy a real cell (see
   *Idempotence*).
3. That model's **dense** cell, before any sparse cell of the same model: the
   I.P. and BaCP cells resolve their starting checkpoint from the dense run's
   record (`results.resolve_checkpoint`).
4. Then its I.P. and BaCP cells, any pruner, any sparsity, any order.

### Off Databricks: what `01_datasets_to_dbfs` is and is not needed for

It stages CIFAR-10, SST-2, tokenizers and the ImageNet weights so a cluster
whose `/dbfs` survives restarts never waits on a download. Elsewhere:

- **Datasets: not needed.** The loaders pass `download=True`, and off
  Databricks `training_utils` sets `cache_dir='./cache'` while the runner
  launches the scripts from `project/scripts` — so CIFAR-10 downloads itself
  into `project/scripts/cache` on first run. The notebook's CIFAR cell stages
  to that same directory when `DATABRICKS_RUNTIME_VERSION` is unset; running it
  only front-loads the download.
- **ViT and the BERTs: not needed.** `MODELS['vit-tiny'].weight` is `''` by
  design — the HF builder arrives pretrained, `init_source` is `hf_hub`, and
  the hub fetch happens when `preflight()` builds the model.
- **ResNet/VGG: needed.** Their registry entries are absolute
  `/dbfs/research/bacp/*.pth` paths on every platform, `load_weights` fails
  soft, and preflight then halts on `init_source='random'`. On Windows that
  path resolves against the current drive, so `fetch_imagenet_weights` plants
  `<drive>:\dbfs\research\bacp\` at the root. Either accept that and run just
  that one cell for the models you want —
  `nb.fetch_imagenet_weights('resnet34')` — or point `model_factory.MODELS` at
  a path of your own first.

## The rules the notebooks enforce

- **Preflight**: a run whose model is not actually pretrained
  (`init_source != imagenet_checkpoint / hf_hub`) halts before training.
- **Idempotence**: a cell with an existing record prints `SKIP`; delete the
  record JSON under `results/runs/` to re-run it. Smoke runs are marked twice —
  `is_smoke` in the record and a `.smoke` key suffix from `make_cell` — and
  `resolve_checkpoint` skips a record carrying either, so a real sparse cell
  cannot start from a 2-batch smoke checkpoint; it raises instead, saying how
  many smoke records it ignored. Run a model's own dense cell before any sparse
  cell of that model: that is what puts a real checkpoint where the sparse
  cells look.
- **Live output**: every run streams epoch / acc / loss / sparsity / ETA into
  its cell and tees to `results/logs/<key>.log`; loss going NaN prints a
  banner.

## Running on your own machine

The reference runs used a single A100 80GB with 24 vCPU; a CNN cell peaks near
4 GB of GPU memory on it, which is why `nb_common`'s parallel runner exists at
all. Adjusting for a smaller machine:

- `num_workers=24` is baked into the base protocols in `nb_common.FAMILIES`.
  Override it per cell — `nb.make_cell(..., num_workers=8)` — rather than
  oversubscribing the CPU with dataloader workers.
- The CNN cells are CIFAR-scale: `image_size=32`, `batch_size=512`. That is the
  ~4 GB figure, so they fit a normal workstation card.
- ViT is a different order of cost: `image_size=224` at `batch_size=256`, ~49x
  the pixels per image at half the batch. Expect to lower `batch_size` and
  expect a much longer wall clock. Measure it before committing — every record
  carries `peak_gpu_mem_gb` and `s_per_epoch_mean`.
- Records go to `BACP_RESULTS_DIR`, which off Databricks defaults to
  `<repo>/results`.

## Protocols and sources

Protocols are the paper's (appendix Tables 1–3) and live in
`nb_common.FAMILIES`. Every equation the framework computes is documented with
its citation in `docs/foundation.md`. Published comparators (paper Table 1) are
embedded in `nb_common.PAPER` and printed by each notebook's results cell —
ResNet-34 has no published row and is reported standalone.

Static pruners: magnitude (Han et al. 2015), SNIP-it (Lee et al. 2019 /
Verdenius et al. 2020), WANDA (Sun et al. 2023), on Zhu & Gupta's (2017) cubic
schedule. Dynamic methods (RigL, EAST) are cited, not run. WikiText-2 MLM is
excluded (defect D1: broken loader).
