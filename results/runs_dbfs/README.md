# Run records exported from DBFS

Exported from `/dbfs/research/bacp_results/runs` on the Databricks A100
workspace, where the sweeps actually ran. They live here rather than in
`results/runs/` because that directory holds only local smoke runs, and
pointing the macro generator at it would blank every value in the paper.

Each file carries the subset of fields `make_results_macros.py` reads:
experiment_group, model_name, method, sparsity_requested, status,
is_smoke, eval_samples_dropped, seed, phase, ended_at, and both accuracy
metrics -- plus provenance fields (duration_s, param counts, learning
rate, epochs, objective settings) kept so a record can be audited.
Training logs and checkpoints stay on DBFS.

Regenerate the paper's macros with:

    python Paper/neurips2026/tools/make_results_macros.py results/runs_dbfs
