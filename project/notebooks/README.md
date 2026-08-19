# Ladder notebooks

Run these in order. Everything heavy lives in `ladder_nb.py`; the notebooks are thin.

| Notebook | What it runs | Gates |
|---|---|---|
| `00_setup.ipynb` | environment, pytest, CIFAR-10 prefetch | - |
| `01_stage_a_substrate.ipynb` | A0, A2-A6 - the sparse substrate | - |
| `02_stage_b_architecture.ipynb` | B1-B3 - DyReLU, phasing, weight sharing | - |
| `03_stage_c_controls.ipynb` | C-null, C0, C0b, C1, C2, C3 | **G1, G3, G4** |
| `04_stage_d_teachers.ipynb` | D1, D2 (= full BaCP) | **G5** |
| `05_stage_d4_backward.ipynb` | D4 leave-one-out | - |
| `06_report.ipynb` | aggregate table, gates, invariants | all |
| `watch_run.ipynb` | one cell, streamed epoch by epoch | - |

## Order is not advisory

`01` must complete before anything else: every sparse rung resolves a dense
checkpoint **of the same seed** at execution time. Launching sparse cells before
A0 exists is how 21 of 22 cells failed on the first attempt.

Within a stage, order does not matter - the pool claims cells atomically
(`O_CREAT|O_EXCL`), so two notebooks running at once cannot double-run a cell.

## These notebooks cannot invalidate a run

`code_fingerprint` hashes `project/**/*.py` but excludes `tests/`, `notebooks/` and
`test_notebooks/`. So `ladder_nb.py` and every notebook here are outside the hash.
Editing them is always safe mid-sweep. Editing anything else in `project/` is not:
it splits the fingerprint and `test_all_comparable_runs_share_one_code_fingerprint`
will fail, because two arms built from different source cannot be compared.

## Known open defects

- **Sparse training diverges to NaN at 0.999.** Dense is clean on all 5 seeds;
  C-null went NaN on 5 of 5 from ~epoch 10 and C0 on 2 of 3 from ~epoch 64. Cause not
  yet established. Check the `nan` column in `nb.progress()` before trusting any table.
- **G1 has no collapse floor.** It tests only `abs(C-null - C0) <= 0.5`, so two arms
  collapsed to chance give `|0.00| <= 0.5` and a **pass**.
- **The ResNet stem keeps its maxpool.** A 32x32 input reaches layer1 at 16x16
  instead of 32x32. Removing it costs zero parameters and ~4x the MACs.
