# The headline grid: CIFAR-10, four backbones

Records for exactly the cells in Table 1 -- resnet34/50 and vgg11/19 on
CIFAR-10 -- so the summary.* macros (cell counts, min/max delta, per-model
and per-sparsity means) describe that table and nothing else.

Excluded on purpose:
  * CIFAR-100 -- the generator keys results as <model>.<arm>.<pruner>.<sparsity>
    with no dataset field, so cifar100 records COLLIDE with cifar10 on 27 keys
    and silently overwrite them.
  * MobileNetV2 -- a separate table; folding it in moved summary.delta.ncells
    from 48 to 57 and summary.delta.min to -9.92, which is the WANDA-at-0.99
    collapse rather than an effect size.

Both are carried instead as hand-entered c100.* and mobilenet.* keys in the
MEASURED block of make_results_macros.py.

    python Paper/neurips2026/tools/make_results_macros.py results/runs_grid
