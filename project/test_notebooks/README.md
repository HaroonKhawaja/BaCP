# Test notebooks — one per model family, static pruning protocol

One notebook per model. Every test (or grouped test) is one cell, and every run
streams its training output live into that cell — epoch, accuracy, loss,
sparsity, s/epoch, ETA, and a loud banner if the loss goes NaN.

```
vision_models/        resnet_static.ipynb   (resnet34 | resnet50 | resnet101)
                      vgg_static.ipynb      (vgg19 | vgg11)
vision_transformers/  vit_static.ipynb      (vit-tiny | vit-small)
llms/                 distilbert_sst2.ipynb
                      roberta_sst2.ipynb
nb_common.py          shared helpers; all protocols live here in FAMILIES
```

## The protocol

The original paper's static pipeline (Paper/BaCP__Appendix.pdf Tables 1–3):
**pretrained** model → dense finetune on the task → static pruning
(magnitude / SNIP / WANDA) at **0.95 / 0.97 / 0.99**, 5 pruning epochs +
10 recovery → BaCP → AdamW finetune. Dynamic methods (RigL, EAST) are cited
from their papers, not re-run.

## Order inside a notebook

1. **setup** — paths, hardware banner.
2. **config** — MODEL / SEED / GPU / SPARSITIES / PRUNERS / SMOKE / OVERRIDES.
3. **weights + preflight** — puts real ImageNet weights where the registry
   looks, then **refuses to continue if the model is not actually pretrained**.
   (`load_weights` fails soft; this cell is what makes silent random-init
   training impossible.)
4. **dense baseline** — required first: every sparse cell starts from the dense
   checkpoint of the same seed.
5. **I.P. cells** (one per pruner) and **BaCP cells** (one per pruner), each
   looping over the three sparsities.
6. **results table** — our numbers next to the paper's Table 1.
7. **health** — every static record on disk.

What you run is up to you: each cell is independent. A cell whose record exists
is **skipped** (`SKIP <key> — already recorded`); delete the record JSON under
`results/runs/` to re-arm it. Set `SMOKE = True` to push the whole pipeline
through 2-batch runs first — smoke records carry a `.smoke` key suffix and can
never satisfy a real cell.

## Where things land

| what | where |
|---|---|
| run records (the results) | `$BACP_RESULTS_DIR/runs/*.json` |
| live logs (also tee'd)    | `$BACP_RESULTS_DIR/logs/<key>.log` |
| checkpoints               | `project/scripts/research/bacp/...` (locally) or `/dbfs/research/bacp/...` (Databricks) |

On Databricks (`DATABRICKS_RUNTIME_VERSION` set), `BACP_RESULTS_DIR` defaults
to `/dbfs/research/bacp_results` and checkpoints go under `/dbfs`, so both
survive the cluster.

## Known exclusions and fixes

- **WikiText-2 MLM is excluded** — its loader is broken (defect D1,
  `docs/research_overview.md` §8.7). The LLM notebooks run SST-2 only, and
  GLUE's test split is unlabelled, so reported "test" accuracy **is the
  validation set**.
- **VGG defect M2 is fixed** in this commit: the pruning scope no longer
  exempts VGG's 119M-param MLP head. Every earlier VGG sparse number is void.
- ViT and the LLMs had never been run under the current codebase — do a
  `SMOKE = True` pass before committing GPU hours there.
