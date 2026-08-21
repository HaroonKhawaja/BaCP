# The notebooks — static pruning, one model per notebook

```
00_smoke_test_all.ipynb    every model x (dense, prune, bacp), 2-epoch/2-batch
                           smoke; ends in a PASS/FAIL table. Run this FIRST on
                           any new cluster.
01_datasets_to_dbfs.ipynb  stage CIFAR-10, SST-2, tokenizers and the ImageNet
                           weights into /dbfs so nothing re-downloads.
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

1. `01_datasets_to_dbfs` once per workspace — everything lands in `/dbfs`.
2. `00_smoke_test_all` once per cluster — every pipeline must PASS.
3. Any model notebook, any cell, in any order you like. Each run cell is
   independent: dense first (sparse cells start from its checkpoint), then
   I.P. and BaCP cells per pruner.

## The rules the notebooks enforce

- **Preflight**: a run whose model is not actually pretrained
  (`init_source != imagenet_checkpoint / hf_hub`) halts before training.
- **Idempotence**: a cell with an existing record prints `SKIP`; delete the
  record JSON under `results/runs/` to re-run it. Smoke records carry a
  `.smoke` suffix and never satisfy real cells.
- **Live output**: every run streams epoch / acc / loss / sparsity / ETA into
  its cell and tees to `results/logs/<key>.log`; loss going NaN prints a
  banner.

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
