"""nb_common.iter_progress: one printed line per EPOCH, not per tqdm repaint.

tqdm repaints its bar many times a second, and every repaint carries the same
"Epoch [k/N]" text. The stream loop used to print one row per matching line,
which buried the cell in dozens of identical rows per epoch and made the
per-epoch timing meaningless -- it measured the gap between two repaints, so
it reported "0.2s/ep eta 0.0m" for an epoch that actually took a minute.

These tests feed the real shapes the trainers emit.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'test_notebooks'))

from nb_common import iter_progress                                # noqa: E402


def _epochs(events):
    return [e for e in events if e[0] == 'epoch']


def _lines(events):
    return [e[1] for e in events if e[0] == 'line']


def test_tqdm_repaints_collapse_to_one_event_per_epoch():
    """Twenty repaints of one epoch must yield exactly one event."""
    raw = [
        f"Training Fine-tuning Epoch [37/50]:  {p}%| | {p}/97 "
        f"[00:0{p % 10}<00:05, 9.5it/s, accuracy: 54.{p:02d}, loss: 2.4{p:02d}]\n"
        for p in range(1, 21)
    ]
    ev = _epochs(list(iter_progress(raw)))
    assert len(ev) == 1, f'expected 1 epoch event, got {len(ev)}'
    _, phase, ep, tot, acc, loss, _sp = ev[0]
    assert (phase, ep, tot) == ('Training Fine-tuning', 37, 50)
    # the LAST repaint holds the finished-epoch metrics
    assert acc == pytest.approx(54.20)
    assert loss == pytest.approx(2.420)


def test_distinct_epochs_and_phases_each_emit_once():
    raw = []
    for ep in (1, 2, 3):
        raw += [f"Training Epoch [{ep}/3]: 50%| [accuracy: 10.0, loss: 9.0]\n",
                f"Training Epoch [{ep}/3]: 99%| [accuracy: 11.0, loss: 8.0]\n"]
    raw += [f"Training Fine-tuning Epoch [1/2]: 99%| [accuracy: 80.0, loss: 0.5]\n"]
    ev = _epochs(list(iter_progress(raw)))
    assert [(e[1], e[2]) for e in ev] == [
        ('Training', 1), ('Training', 2), ('Training', 3),
        ('Training Fine-tuning', 1)]


def test_carriage_returns_in_one_chunk_are_split():
    """tqdm repaints with \\r; a buffered chunk must not count as one line."""
    chunk = ("Training Epoch [5/9]: 10%| [loss: 3.0]\r"
             "Training Epoch [5/9]: 60%| [loss: 2.0]\r"
             "Training Epoch [5/9]: 99%| [loss: 1.0]\n")
    ev = _epochs(list(iter_progress([chunk])))
    assert len(ev) == 1
    assert ev[0][5] == pytest.approx(1.0)


def test_a_field_missing_from_a_repaint_does_not_blank_it():
    """Sparsity is printed only on mask-update steps; it must not be lost."""
    raw = ["Training Epoch [4/10]: 20%| [accuracy: 50.0, loss: 2.0, sparsity: 0.9500]\n",
           "Training Epoch [4/10]: 99%| [accuracy: 55.0, loss: 1.5]\n"]
    ev = _epochs(list(iter_progress(raw)))
    assert len(ev) == 1
    _, _, _, _, acc, loss, sp = ev[0]
    assert (acc, loss) == (pytest.approx(55.0), pytest.approx(1.5))
    assert sp == pytest.approx(0.95), 'sparsity was blanked by a later repaint'


def test_non_epoch_lines_pass_through_untouched():
    raw = ["[SPARSITY SCOPE] Prunable: backbone, task head\n",
           "Training Epoch [1/1]: 99%| [loss: 1.0]\n",
           "[TRAINER] Checkpoint saved.\n"]
    events = list(iter_progress(raw))
    assert _lines(events) == ["[SPARSITY SCOPE] Prunable: backbone, task head",
                              "[TRAINER] Checkpoint saved."]
    assert len(_epochs(events)) == 1


def test_last_epoch_is_flushed_at_end_of_stream():
    """The final epoch has no successor to trigger its flush."""
    ev = _epochs(list(iter_progress(["Training Epoch [9/9]: 99%| [loss: 0.1]\n"])))
    assert len(ev) == 1 and ev[0][2] == 9


def test_nan_loss_survives_to_the_event():
    ev = _epochs(list(iter_progress(["Training Epoch [2/5]: 99%| [loss: nan]\n"])))
    loss = ev[0][5]
    assert loss != loss, 'NaN must reach the consumer so the banner can fire'


def test_realistic_burst_prints_one_row_per_epoch():
    """End-to-end shape check against the flood seen in the vgg19 probe."""
    raw = []
    for ep in range(35, 39):
        raw += [f"Training Fine-tuning Epoch [{ep}/50]: {b}/97 "
                f"[accuracy: 90.{b:02d}, loss: 0.5{b:02d}]\n"
                for b in range(1, 26)]
    ev = _epochs(list(iter_progress(raw)))
    assert len(ev) == 4, f'100 raw lines -> {len(ev)} rows, expected 4'
    assert [e[2] for e in ev] == [35, 36, 37, 38]
