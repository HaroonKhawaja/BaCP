"""L0 -- static guards. No model, no data, no GPU. Runs in well under a second.

These are the cheapest tests in the suite and they catch the most expensive
class of bug in this codebase: a name that does not exist on a code path nobody
has run recently. Five of the eight blocking crashes on `main` are exactly that,
and every one of them is caught here.
"""

import ast
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path

import pytest
from pyflakes import api as pyflakes_api
from pyflakes import reporter as pyflakes_reporter

PROJECT = Path(__file__).resolve().parents[1]

# compute_metrics is the one module needing fvcore, and nothing in the repo
# imports it (git history: added with the FLOPs work, unhooked again when
# inference timing came out of the training pipeline). Keeping fvcore -- an
# unmaintained package that also forces numpy<2 -- out of requirements-dev.txt
# is worth more than importing a dead module, so skip rather than install.
_HAS_FVCORE = importlib.util.find_spec("fvcore") is not None

MODULES = [
    "bacp",
    pytest.param("compute_metrics", marks=pytest.mark.skipif(
        not _HAS_FVCORE, reason="compute_metrics needs fvcore; nothing imports it")),
    "dataset_factory",
    "dyrelu_adapter",
    "logger",
    "loss_functions",
    "model_factory",
    "pruning_factory",
    "results",
    "runner",
    "trainer",
    "training_utils",
    "utils",
    "weight_sharing",
    "models.resnet",
    "models.vgg",
]

# Files where an undefined name or an unused local is a real defect rather than
HOT_FILES = [
    PROJECT / "bacp.py",
    PROJECT / "dyrelu_adapter.py",
    PROJECT / "loss_functions.py",
    PROJECT / "model_factory.py",
    PROJECT / "pruning_factory.py",
    PROJECT / "trainer.py",
    PROJECT / "training_utils.py",
    PROJECT / "utils.py",
    PROJECT / "weight_sharing.py",
    PROJECT / "models" / "resnet.py",
]


class _Collector:
    """Minimal pyflakes reporter that keeps messages instead of printing them."""

    def __init__(self):
        self.messages = []
        self.errors = []

    def unexpectedError(self, filename, msg):
        self.errors.append(f"{filename}: {msg}")

    def syntaxError(self, filename, msg, lineno, offset, text):
        self.errors.append(f"{filename}:{lineno}: {msg}")

    def flake(self, message):
        self.messages.append(message)


def _flake(paths):
    collector = _Collector()
    for path in paths:
        pyflakes_api.checkPath(str(path), reporter=collector)
    assert not collector.errors, "pyflakes could not parse: " + "; ".join(collector.errors)
    return collector.messages


def _of_class(messages, class_name):
    return [f"{m.filename}:{m.lineno}: {m.message % m.message_args}"
            for m in messages if type(m).__name__ == class_name]


def _unresolved_star_names(messages):
    """Resolve pyflakes' 'may be undefined, or defined from star imports' messages.

    This step is why the test works at all. bacp.py does
    ``from training_utils import *`` (plus pruning_factory and utils), and in the
    presence of a star import pyflakes will NOT emit UndefinedName -- it
    downgrades to the far weaker ImportStarUsage, which merely says the name
    *might* come from one of those modules.

    That downgrade is exactly why three undefined names survived on the BaCP
    path for months. So do the resolution pyflakes declines to do: import each
    named star module and check whether the name is actually there.
    """
    unresolved = []
    for m in messages:
        if type(m).__name__ != "ImportStarUsage":
            continue
        name, from_list = m.message_args[0], m.message_args[1]
        modules = [part.strip() for part in str(from_list).split(",")]
        if any(hasattr(importlib.import_module(mod), name) for mod in modules):
            continue
        unresolved.append(f"{m.filename}:{m.lineno}: {name} (not in {from_list})")
    return unresolved


@pytest.mark.parametrize("module_name", MODULES)
def test_all_modules_import(module_name):
    """Every module imports cleanly.

    A module-scope failure here means no test below can run, so this is the
    first thing to look at when the suite goes dark.
    """
    importlib.import_module(module_name)


def test_no_undefined_names():
    """No module references a name that does not exist.

    This is the test for the three undefined names in bacp.py:
    _handle_wanda_hooks, _handle_optimizer_and_pruning, and PRUNING_REGISTRY
    (the real registry is PRUNER_REGISTRY). All three sit on the BaCP training
    path, which is why they survived: nothing ran it.

    Covers both message classes -- plain UndefinedName, and the star-import
    downgrade that hid these three in the first place (see
    _unresolved_star_names).
    """
    messages = _flake(HOT_FILES)
    found = _of_class(messages, "UndefinedName") + _unresolved_star_names(messages)
    assert not found, "Undefined names:\n  " + "\n  ".join(found)


def test_star_imports_are_not_used_in_hot_files():
    """Flag `from x import *` in the files that carry the training loops.

    Kept separate from test_no_undefined_names, and expected to fail until the
    star imports in bacp.py are replaced with explicit ones. It is not style
    pedantry: a star import defeats pyflakes' undefined-name analysis outright,
    which is the mechanism that let three broken call sites ship. Fixing this
    makes the whole class of bug statically detectable for free.
    """
    offenders = []
    for path in HOT_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
                offenders.append(f"{path.name}:{node.lineno}: from {node.module} import *")
    assert not offenders, (
        "Star imports disable undefined-name detection:\n  " + "\n  ".join(offenders)
    )


def test_no_unused_locals_in_hot_files():
    """No assigned-but-never-read local variables.

    This is the test for `thera = x` in dyrelu_adapter.py, a typo for `theta`
    that makes DyReLUB raise UnboundLocalError on any 2-D input -- which is the
    VGG classifier path, where DyReLUB(4096) sees [B, 4096] activations.
    """
    found = _of_class(_flake(HOT_FILES), "UnusedVariable")
    assert not found, "Unused locals (often a typo'd assignment):\n  " + "\n  ".join(found)


def _imported_module_names(tree):
    """Every module name the file imports, star or otherwise."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


@pytest.mark.parametrize("class_path", ["bacp.BaCPTrainer", "trainer.Trainer"])
def test_self_method_calls_resolve(class_path):
    """Catch a *module-level function* being called as if it were a method.

    This is the test for `self._initialize_dyrelu_phasing()` in BaCPTrainer:
    _initialize_dyrelu_phasing is a module-level function in training_utils
    pulled in by a star-import, not a method. It is already called correctly a
    few lines earlier, so the second call is a duplicate that AttributeErrors
    during __init__ before training starts.

    Deliberately narrow. Both trainers copy every attribute off the args object
    with ``setattr(self, key, value)``, so ``self.model(...)``,
    ``self.criterion(...)`` and friends are legitimate instance attributes that
    ``hasattr(cls, ...)`` cannot see -- flagging those would make this test
    noise. So the rule is precise: flag ``self.X(...)`` only when X is absent
    from the class *and* exists as a module-level callable in something this
    file imports. That is exactly the confusion being guarded against, and it
    has no false positives on instance attributes.
    """
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)

    tree = ast.parse(Path(inspect.getsourcefile(cls)).read_text(encoding="utf-8"))
    class_node = next(n for n in ast.walk(tree)
                      if isinstance(n, ast.ClassDef) and n.name == class_name)

    called = {node.func.attr for node in ast.walk(class_node)
              if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute)
              and isinstance(node.func.value, ast.Name)
              and node.func.value.id == "self"}

    imported = []
    for name in _imported_module_names(tree):
        try:
            imported.append(importlib.import_module(name))
        except ImportError:
            continue

    shadowing = sorted(
        name for name in called
        if not hasattr(cls, name)
        and any(callable(getattr(mod, name, None)) for mod in imported)
    )
    assert not shadowing, (
        f"{class_name} calls self.{shadowing} but those are module-level "
        f"functions, not methods. Call them as bare functions: f(self)."
    )


def test_pruner_registry_contract():
    """The registry exposes exactly the documented pruners, and rejects others.

    Also pins the name: it is PRUNER_REGISTRY, and bacp.py's get_pruner reaches
    for PRUNING_REGISTRY.
    """
    import pruning_factory

    assert not hasattr(pruning_factory, "PRUNING_REGISTRY"), (
        "Two registry names would be worse than one. Keep PRUNER_REGISTRY."
    )
    # wanda joined the registry once it was actually implemented. Before that it
    # was 133 commented-out lines containing a syntax error, which means the
    # WANDA column in the submitted paper cannot have come from this code -- and
    # is why this set is pinned rather than merely spot-checked.
    assert set(pruning_factory.PRUNER_REGISTRY) == {
        "magnitude", "local_magnitude", "snip", "rigl", "east", "wanda",
    }

    with pytest.raises(ValueError) as excinfo:
        pruning_factory.get_pruner("does_not_exist", model=None, epochs=1, sparsity=0.5)
    # The error should name the valid options -- the notebooks are full of stale
    # names like 'magnitude_pruning' and 'wanda_pruning'.
    assert "magnitude" in str(excinfo.value)


def test_cli_model_choices_are_all_registered():
    """Every --model_name the CLI accepts must exist in model_factory.MODELS.

    These two lists had drifted: the parsers offered vgg11 and vgg19 while both
    entries were commented out of MODELS, so those runs died with
    "Unknown model vgg11" after argument parsing had already succeeded.
    """
    sys.path.insert(0, str(PROJECT / "scripts"))
    import scripting_utils

    from model_factory import MODELS

    missing = sorted(set(scripting_utils.models) - set(MODELS))
    assert not missing, (
        f"CLI offers models that are not registered: {missing}. "
        "Either register them in model_factory.MODELS or remove them from the choices."
    )


def test_cli_dataset_choices_are_all_registered():
    """Every --dataset_name the CLI accepts must be loadable.

    CV datasets live in dataset_factory.CV_DATASETS, text datasets in
    TEXT_DATASETS. The CLI previously offered only cifar10/cifar100 while seven CV
    datasets were registered, and offered no text dataset at all even though the
    paper reports SST-2 and WikiText-2.
    """
    sys.path.insert(0, str(PROJECT / "scripts"))
    import scripting_utils

    from dataset_factory import CV_DATASETS, TEXT_DATASETS

    known = set(CV_DATASETS) | set(TEXT_DATASETS)
    missing = sorted(set(scripting_utils.datasets) - known)
    assert not missing, f"CLI offers unloadable datasets: {missing}"


def test_registered_model_types_are_cv_or_llm():
    """spec.type must use the same vocabulary as --model_type.

    It previously held 'vision' while the CLI and Trainer._handle_metrics used
    'cv'/'llm' -- two vocabularies for one concept, which is how the language path
    came to be unreachable.
    """
    sys.path.insert(0, str(PROJECT / "scripts"))
    import scripting_utils

    from model_factory import MODELS

    allowed = set(scripting_utils.model_types)
    bad = {name: spec.type for name, spec in MODELS.items() if spec.type not in allowed}
    assert not bad, f"model specs with a type outside {allowed}: {bad}"


def test_every_dataset_has_a_class_count_except_mlm():
    """DATASET_NUM_CLASSES must cover every dataset with a fixed label set.

    wikitext2 is deliberately absent: masked language modelling predicts over the
    tokenizer vocabulary, so there is no fixed class count.
    """
    from dataset_factory import CV_DATASETS, DATASET_NUM_CLASSES, TEXT_DATASETS

    labelled = (set(CV_DATASETS) | set(TEXT_DATASETS)) - {"wikitext2"}
    missing = sorted(labelled - set(DATASET_NUM_CLASSES))
    assert not missing, f"datasets with no declared class count: {missing}"
    assert "wikitext2" not in DATASET_NUM_CLASSES


def test_load_weights_has_no_bare_except():
    """``utils.load_weights`` must not swallow every exception.

    A bare `except:` there hides real load failures and falls through to a
    strict=False partial load that tolerates any key mismatch -- which is how a
    silently-random-initialised "pretrained" model gets into an experiment.
    """
    import utils

    tree = ast.parse(Path(inspect.getsourcefile(utils)).read_text(encoding="utf-8"))
    func = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "load_weights")

    bare = [h.lineno for h in ast.walk(func)
            if isinstance(h, ast.ExceptHandler) and h.type is None]
    assert not bare, f"bare 'except:' at line(s) {bare}; catch a named exception"
