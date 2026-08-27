#!/usr/bin/env python3
"""Regenerate ``Paper/neurips2026/results_macros.tex`` from the run records.

    python Paper/neurips2026/tools/make_results_macros.py results/runs

The paper never contains a typed accuracy number.  Every figure in prose, in a
table or in the abstract is written ``\\result{<key>}``.  A key that this script
has not registered renders as a loud red ``??`` instead of erroring, so the
document compiles at every stage of the sweep and silently becomes correct as
cells land.

This file -- not the generated ``.tex`` -- is the source of truth.  The three
hand-entered blocks (macro definitions, MEASURED, LITERATURE) are string
constants below, so the output is fully reproducible: delete
``results_macros.tex``, run this, and you get it back.  Never hand-edit the
generated file; edit the constants here.

Key namespaces
--------------
The authoritative spelling is whatever ``sections/*.tex`` uses, because that is
where the 187 call sites live.  Where this script's own preferred spelling
differs, BOTH are emitted with the same value, so either resolves.

``<model>.<arm>.<pruner>.<sparsity>``
    The headline matched sweep, emitted by this script from the records.
    ``arm`` is ``bacp`` or ``ip`` (the iterative-pruning baseline).
    e.g. ``resnet50.bacp.magnitude.0.95``, ``resnet50.ip.wanda.0.999``.
``<model>.dense.none.0.0``  (alias: ``<model>.dense``)
    A dense baseline.
``<model>.delta.<pruner>.<sparsity>``
    bacp minus ip, computed here, carrying an explicit sign.
``<model>.abl{ip,control,legacy,delta}.<pruner>.<sparsity>``
    The objective ablation.  Run under the PREVIOUS protocol (60 epochs,
    delta_T 88, no validation split, single seed), so the ``abl`` prefix on
    the ARM keeps it out of the headline namespace -- a headline table can
    never be filled in by an old-protocol number.
    Alias: ``abl.<model>.<arm>.<pruner>.<sparsity>``.
``<model>.probe*``, ``cifar10.noise.floor.{lo,hi}``
    Measured, reported, but not a headline result: the VGG learning-rate
    probe, the run-to-run noise floor, the defective-stem dense ResNet-34
    seed spread.
``lit.*``
    Published comparators quoted from the literature.  Constants, not our
    measurements, but routed through the same macro so no bare accuracy is
    ever typed into the paper.
``summary.*``
    Reserved for cross-cell summary statistics.  Nothing is registered, so
    every such key renders ``??`` until a human decides what to claim.

Metric
------
``test_acc_exact_pct`` is preferred over ``test_acc_pct`` and the script says
which it used per cell.  The two differ whenever the eval loader dropped a
partial final batch -- the defect that voided the earlier ResNet-34 numbers --
so preferring the sample-weighted figure is deliberate.  ``--metric`` overrides.

Records that are ``status != 'ok'``, that are smoke runs, that dropped eval
samples, or that carry no usable accuracy are skipped and counted in the
report printed to stderr.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# Phases that can carry a final reported accuracy, best first.  A pruning arm's
# number is the post-fine-tune one; `prune` is only a fallback for a cell whose
# fine-tune record is missing.
PHASE_PRIORITY = ('finetune', 'prune', 'dense', 'eval_only', 'contrastive')

# Arm names as they appear in `results.canonical_method`, mapped to the token
# used in a macro key.  A bare pruner name (no '+') is the iterative-pruning
# baseline, which is the arm BaCP is compared against.
ARM_ALIASES = {
    'bacp': 'bacp',
    'ce_only': 'ceonly',
    'kd_kl': 'kdkl',
    'kd_feature': 'kdfeat',
}


# --------------------------------------------------------------------------
# hand-maintained blocks
# --------------------------------------------------------------------------

PREAMBLE = r"""% !TeX root = ../main.tex
%
% GENERATED FILE -- do not edit by hand.
% Regenerate with:
%     python Paper/neurips2026/tools/make_results_macros.py results/runs
% The macro definitions and the MEASURED / LITERATURE blocks are string
% constants inside that script; edit them there.
%
% Contract
% --------
%   \defresult{key}{value}    register a value
%   \defresultsd{key}{sd}     register a standard deviation for that key
%   \defresultn{key}{n}       register the number of seeds behind it
%   \result{key}              -> the value, or a loud red ?? if unregistered
%   \resultpm{key}            -> "value +/- sd", or just "value" if no sd,
%                                or a loud red ?? if unregistered
%   \resultn{key}             -> the seed count, or ??
%   \resultif{key}{yes}{no}   -> branch on whether a key is registered
%
% Nothing here can raise a LaTeX error on a missing key.  A missing key is a
% visible red ?? in the PDF and a "MISSING RESULT KEY" line in the .log, so an
% unfinished cell is impossible to ship by accident but never blocks a build:
%     grep "MISSING RESULT KEY" main.log | sort -u
%
% \input this file in the PREAMBLE, after xcolor.  The \@ifpackageloaded guard
% below is a preamble-only command.

\makeatletter
\@ifpackageloaded{xcolor}{}{\RequirePackage{xcolor}}

\newcommand{\bacp@missing}{\textcolor{red}{\textbf{??}}}
\newcommand{\bacp@warn}[1]{\GenericWarning{}{MISSING RESULT KEY: #1}}

\newcommand{\defresult}[2]{\expandafter\gdef\csname bacp@val@#1\endcsname{#2}}
\newcommand{\defresultsd}[2]{\expandafter\gdef\csname bacp@sd@#1\endcsname{#2}}
\newcommand{\defresultn}[2]{\expandafter\gdef\csname bacp@n@#1\endcsname{#2}}

\DeclareRobustCommand{\result}[1]{%
  \ifcsname bacp@val@#1\endcsname
    \csname bacp@val@#1\endcsname
  \else
    \bacp@warn{#1}\bacp@missing
  \fi
}

\DeclareRobustCommand{\resultpm}[1]{%
  \ifcsname bacp@val@#1\endcsname
    \csname bacp@val@#1\endcsname
    \ifcsname bacp@sd@#1\endcsname
      \,\ensuremath{\pm}\,\csname bacp@sd@#1\endcsname
    \fi
  \else
    \bacp@warn{#1}\bacp@missing
  \fi
}

\DeclareRobustCommand{\resultn}[1]{%
  \ifcsname bacp@n@#1\endcsname
    \csname bacp@n@#1\endcsname
  \else
    \bacp@warn{#1}\bacp@missing
  \fi
}

% \resultif{key}{if registered}{if not} -- for prose that must change shape,
% not just its number, when a cell is missing.
\DeclareRobustCommand{\resultif}[3]{%
  \ifcsname bacp@val@#1\endcsname #2\else #3\fi
}
\makeatother
"""

MEASURED = r"""
% ==========================================================================
% MEASURED -- hand-entered from Paper/neurips2026/BRIEF.md.
% These are the only numbers of ours that may be stated as results today.
% ==========================================================================

% --- MobileNetV2 spot-check ------------------------------------------------
% CIFAR-10 / magnitude / 3 seeds, mean and sample sd.  A fourth architecture
% family, added to test whether the effect is specific to the ResNet/VGG
% shapes: MobileNetV2 is inverted-residual and entirely depthwise-separable,
% with 2.2M prunable weights against ResNet-34's 21M.
%
% PROTOCOL DEVIATION, stated because it is a real degree of freedom.  Both
% arms run SGD 0.1 here, not the grid's I.P. 0.01 / BaCP 0.1.  MobileNetV2
% inherited the ResNet recipe untuned, and a learning-rate probe showed 0.01
% is simply wrong for this model's I.P. arm: at 0.99 it collapses to chance
% (10.02 +/- 0.03) while 0.1 reaches 78.63.  Reporting the 0.01 arm would
% have manufactured a +70 point delta out of an untuned baseline.  BaCP was
% probed at 0.5 as well and is 7.2 points worse there, so 0.1 is each arm's
% measured optimum at the sparsities that matter.
%
% 0.999 is dead for BOTH arms on this architecture in every configuration
% measured -- I.P. at 0.001/0.01/0.1 and BaCP at 0.01/0.1/0.5 all land on
% 10.0, chance on CIFAR-10.  Only ~2,200 of 2.2M prunable weights survive
% there, about 42 per tensor.  This is a MobileNetV2 result, NOT a general
% one: ResNet-50 and VGG-19 both retain ~80-85 at 0.999.
\defresult{mobilenet.dense}{92.36}
\defresultsd{mobilenet.dense}{0.12}
\defresultn{mobilenet.dense}{3}
\defresult{mobilenet.dense.none.0.0}{92.36}

\defresult{mobilenet.ip.magnitude.0.5}{84.61}
\defresultsd{mobilenet.ip.magnitude.0.5}{0.14}
\defresultn{mobilenet.ip.magnitude.0.5}{3}
\defresult{mobilenet.bacp.magnitude.0.5}{89.92}
\defresultsd{mobilenet.bacp.magnitude.0.5}{0.21}
\defresultn{mobilenet.bacp.magnitude.0.5}{3}
\defresult{mobilenet.delta.magnitude.0.5}{+5.31}

\defresult{mobilenet.ip.magnitude.0.95}{83.46}
\defresultsd{mobilenet.ip.magnitude.0.95}{1.32}
\defresultn{mobilenet.ip.magnitude.0.95}{3}
\defresult{mobilenet.bacp.magnitude.0.95}{87.38}
\defresultsd{mobilenet.bacp.magnitude.0.95}{0.32}
\defresultn{mobilenet.bacp.magnitude.0.95}{3}
\defresult{mobilenet.delta.magnitude.0.95}{+3.92}

\defresult{mobilenet.ip.magnitude.0.97}{82.64}
\defresultsd{mobilenet.ip.magnitude.0.97}{1.36}
\defresultn{mobilenet.ip.magnitude.0.97}{3}
\defresult{mobilenet.bacp.magnitude.0.97}{85.67}
\defresultsd{mobilenet.bacp.magnitude.0.97}{0.56}
\defresultn{mobilenet.bacp.magnitude.0.97}{3}
\defresult{mobilenet.delta.magnitude.0.97}{+3.04}

\defresult{mobilenet.ip.magnitude.0.99}{78.63}
\defresultsd{mobilenet.ip.magnitude.0.99}{3.51}
\defresultn{mobilenet.ip.magnitude.0.99}{3}
\defresult{mobilenet.bacp.magnitude.0.99}{80.34}
\defresultsd{mobilenet.bacp.magnitude.0.99}{0.23}
\defresultn{mobilenet.bacp.magnitude.0.99}{3}
\defresult{mobilenet.delta.magnitude.0.99}{+1.71}

\defresult{mobilenet.ip.magnitude.0.999}{10.00}
\defresultsd{mobilenet.ip.magnitude.0.999}{0.00}
\defresultn{mobilenet.ip.magnitude.0.999}{3}
\defresult{mobilenet.bacp.magnitude.0.999}{10.00}
\defresultsd{mobilenet.bacp.magnitude.0.999}{0.00}
\defresultn{mobilenet.bacp.magnitude.0.999}{3}
\defresult{mobilenet.delta.magnitude.0.999}{+0.00}

% --- MobileNetV2 spot-check: SNIP-it and WANDA -----------------------------
% Same protocol as the magnitude block above: CIFAR-10, three seeds, BOTH arms
% at SGD 0.1.  Sparsity 0.995 replaces 0.999 for these two criteria because
% 0.999 is dead for both arms on this architecture in every configuration
% measured, so three more criteria x three seeds there would only reproduce
% chance.
%
% WANDA CAVEAT.  pruning_factory's WANDA path falls back to plain magnitude
% ranking for any layer whose activation statistic width does not match the
% flattened weight, and grouped/depthwise Conv2d hit this deterministically
% because F.unfold is group-unaware.  Measured on the trained model (param
% counts cross-checked against the run records, 2,236,682 total / 2,202,560
% prunable): 17 of 52 convolutional layers are depthwise, holding 64,224
% weights = 2.92% of the prunable pool.  So 97.08% of the pool is genuinely
% WANDA-scored, but the row is not purely WANDA and must not be reported as
% though it were.
%
% WANDA AT 0.99 IS A COLLAPSE, NOT A DELTA.  Per-seed I.P. is 39.76 / 10.00 /
% 10.00 and BaCP is 10.00 / 10.00 / 10.00.  The I.P. mean of 19.92 is one
% surviving seed averaged with two dead ones; its sd of 17.18 is the tell.
% Both arms fail here.  No delta is registered for that cell on purpose.
\defresult{mobilenet.ip.snip.0.95}{83.52}
\defresultsd{mobilenet.ip.snip.0.95}{0.19}
\defresultn{mobilenet.ip.snip.0.95}{3}
\defresult{mobilenet.bacp.snip.0.95}{87.26}
\defresultsd{mobilenet.bacp.snip.0.95}{0.17}
\defresultn{mobilenet.bacp.snip.0.95}{3}
\defresult{mobilenet.delta.snip.0.95}{+3.74}

\defresult{mobilenet.ip.snip.0.97}{82.56}
\defresultsd{mobilenet.ip.snip.0.97}{1.36}
\defresultn{mobilenet.ip.snip.0.97}{3}
\defresult{mobilenet.bacp.snip.0.97}{85.54}
\defresultsd{mobilenet.bacp.snip.0.97}{0.11}
\defresultn{mobilenet.bacp.snip.0.97}{3}
\defresult{mobilenet.delta.snip.0.97}{+2.98}

\defresult{mobilenet.ip.snip.0.99}{74.40}
\defresultsd{mobilenet.ip.snip.0.99}{1.37}
\defresultn{mobilenet.ip.snip.0.99}{3}
\defresult{mobilenet.bacp.snip.0.99}{80.33}
\defresultsd{mobilenet.bacp.snip.0.99}{0.15}
\defresultn{mobilenet.bacp.snip.0.99}{3}
\defresult{mobilenet.delta.snip.0.99}{+5.93}

\defresult{mobilenet.ip.snip.0.995}{61.13}
\defresultsd{mobilenet.ip.snip.0.995}{0.38}
\defresultn{mobilenet.ip.snip.0.995}{3}
\defresult{mobilenet.bacp.snip.0.995}{74.43}
\defresultsd{mobilenet.bacp.snip.0.995}{0.28}
\defresultn{mobilenet.bacp.snip.0.995}{3}
\defresult{mobilenet.delta.snip.0.995}{+13.30}

\defresult{mobilenet.ip.wanda.0.95}{82.47}
\defresultsd{mobilenet.ip.wanda.0.95}{0.48}
\defresultn{mobilenet.ip.wanda.0.95}{3}
\defresult{mobilenet.bacp.wanda.0.95}{85.85}
\defresultsd{mobilenet.bacp.wanda.0.95}{0.30}
\defresultn{mobilenet.bacp.wanda.0.95}{3}
\defresult{mobilenet.delta.wanda.0.95}{+3.38}

\defresult{mobilenet.ip.wanda.0.97}{79.10}
\defresultsd{mobilenet.ip.wanda.0.97}{0.82}
\defresultn{mobilenet.ip.wanda.0.97}{3}
\defresult{mobilenet.bacp.wanda.0.97}{84.08}
\defresultsd{mobilenet.bacp.wanda.0.97}{0.57}
\defresultn{mobilenet.bacp.wanda.0.97}{3}
\defresult{mobilenet.delta.wanda.0.97}{+4.98}

\defresult{mobilenet.ip.wanda.0.99}{19.92}
\defresultsd{mobilenet.ip.wanda.0.99}{17.18}
\defresultn{mobilenet.ip.wanda.0.99}{3}
\defresult{mobilenet.bacp.wanda.0.99}{10.00}
\defresultsd{mobilenet.bacp.wanda.0.99}{0.00}
\defresultn{mobilenet.bacp.wanda.0.99}{3}

\defresult{mobilenet.ip.wanda.0.995}{10.00}
\defresultsd{mobilenet.ip.wanda.0.995}{0.00}
\defresultn{mobilenet.ip.wanda.0.995}{3}
\defresult{mobilenet.bacp.wanda.0.995}{10.00}
\defresultsd{mobilenet.bacp.wanda.0.995}{0.00}
\defresultn{mobilenet.bacp.wanda.0.995}{3}

\defresult{mobilenet.wanda.fallback.pct}{2.92}
\defresult{mobilenet.wanda.scored.pct}{97.08}
\defresult{mobilenet.wanda.dwlayers}{17}
\defresult{mobilenet.wanda.convlayers}{52}


% --- MobileNetV2: magnitude at 0.995 ---------------------------------------
% Run after SNIP/WANDA to close the one hole in the grid. It overturned the
% reading that was in 04_results.tex: magnitude's margin does NOT keep
% narrowing. It is +1.71 at 0.99 and +42.49 at 0.995, because the baseline
% starts to fail there (24.98 +/- 13.47, per-seed 36.10 / 28.84 / 10.00 --
% one seed already at chance) while BaCP holds 67.47 +/- 1.60.
% Same protocol: three seeds, both arms SGD 0.1.
\defresult{mobilenet.ip.magnitude.0.995}{24.98}
\defresultsd{mobilenet.ip.magnitude.0.995}{13.47}
\defresultn{mobilenet.ip.magnitude.0.995}{3}
\defresult{mobilenet.bacp.magnitude.0.995}{67.47}
\defresultsd{mobilenet.bacp.magnitude.0.995}{1.60}
\defresultn{mobilenet.bacp.magnitude.0.995}{3}
\defresult{mobilenet.delta.magnitude.0.995}{+42.49}


% --- ResNet-34 / CIFAR-100: the second dataset -----------------------------
% Three criteria x four sparsities x three seeds x two arms, plus three dense
% baselines. Same matched protocol as the CIFAR-10 grid, and the transfer was
% verified rather than assumed: a 100-way cross-entropy has a different loss
% scale than a 10-way one, which changes how the CE term competes with the
% contrastive terms. A learning-rate probe (dense 72.67, I.P. at 0.95 scoring
% 70.27 under 0.01 against 63.45 under 0.1) confirmed the CIFAR-10 split
% transfers, so nothing was retuned.
%
% ResNet-34 carries no published comparator row -- the original work used
% ResNet-50/101 -- so these numbers are not under reproduction pressure.
%
% BaCP is ahead in all twelve cells, and by more than on CIFAR-10 in eleven of
% them. The exception is WANDA at 0.999, where both arms are near-destroyed
% (I.P. 2.87 against a 1.00 chance floor) and there is little left to recover.
\defresult{c100.dense}{70.24}
\defresultsd{c100.dense}{2.10}
\defresultn{c100.dense}{3}
\defresult{c100.dense.none.0.0}{70.24}

\defresult{c100.ip.magnitude.0.95}{68.77}
\defresultsd{c100.ip.magnitude.0.95}{1.30}
\defresultn{c100.ip.magnitude.0.95}{3}
\defresult{c100.bacp.magnitude.0.95}{70.48}
\defresultsd{c100.bacp.magnitude.0.95}{0.14}
\defresultn{c100.bacp.magnitude.0.95}{3}
\defresult{c100.delta.magnitude.0.95}{+1.71}

\defresult{c100.ip.magnitude.0.97}{68.47}
\defresultsd{c100.ip.magnitude.0.97}{1.30}
\defresultn{c100.ip.magnitude.0.97}{3}
\defresult{c100.bacp.magnitude.0.97}{69.97}
\defresultsd{c100.bacp.magnitude.0.97}{0.15}
\defresultn{c100.bacp.magnitude.0.97}{3}
\defresult{c100.delta.magnitude.0.97}{+1.50}

\defresult{c100.ip.magnitude.0.99}{66.42}
\defresultsd{c100.ip.magnitude.0.99}{1.34}
\defresultn{c100.ip.magnitude.0.99}{3}
\defresult{c100.bacp.magnitude.0.99}{66.92}
\defresultsd{c100.bacp.magnitude.0.99}{0.35}
\defresultn{c100.bacp.magnitude.0.99}{3}
\defresult{c100.delta.magnitude.0.99}{+0.50}

\defresult{c100.ip.magnitude.0.999}{19.84}
\defresultsd{c100.ip.magnitude.0.999}{2.17}
\defresultn{c100.ip.magnitude.0.999}{3}
\defresult{c100.bacp.magnitude.0.999}{42.80}
\defresultsd{c100.bacp.magnitude.0.999}{0.44}
\defresultn{c100.bacp.magnitude.0.999}{3}
\defresult{c100.delta.magnitude.0.999}{+22.96}

\defresult{c100.ip.snip.0.95}{66.10}
\defresultsd{c100.ip.snip.0.95}{1.48}
\defresultn{c100.ip.snip.0.95}{3}
\defresult{c100.bacp.snip.0.95}{69.84}
\defresultsd{c100.bacp.snip.0.95}{0.16}
\defresultn{c100.bacp.snip.0.95}{3}
\defresult{c100.delta.snip.0.95}{+3.74}

\defresult{c100.ip.snip.0.97}{65.71}
\defresultsd{c100.ip.snip.0.97}{1.45}
\defresultn{c100.ip.snip.0.97}{3}
\defresult{c100.bacp.snip.0.97}{68.76}
\defresultsd{c100.bacp.snip.0.97}{0.38}
\defresultn{c100.bacp.snip.0.97}{3}
\defresult{c100.delta.snip.0.97}{+3.05}

\defresult{c100.ip.snip.0.99}{61.91}
\defresultsd{c100.ip.snip.0.99}{0.54}
\defresultn{c100.ip.snip.0.99}{3}
\defresult{c100.bacp.snip.0.99}{64.33}
\defresultsd{c100.bacp.snip.0.99}{0.04}
\defresultn{c100.bacp.snip.0.99}{3}
\defresult{c100.delta.snip.0.99}{+2.42}

\defresult{c100.ip.snip.0.999}{30.66}
\defresultsd{c100.ip.snip.0.999}{0.58}
\defresultn{c100.ip.snip.0.999}{3}
\defresult{c100.bacp.snip.0.999}{36.18}
\defresultsd{c100.bacp.snip.0.999}{0.62}
\defresultn{c100.bacp.snip.0.999}{3}
\defresult{c100.delta.snip.0.999}{+5.52}

\defresult{c100.ip.wanda.0.95}{63.93}
\defresultsd{c100.ip.wanda.0.95}{0.76}
\defresultn{c100.ip.wanda.0.95}{3}
\defresult{c100.bacp.wanda.0.95}{68.20}
\defresultsd{c100.bacp.wanda.0.95}{0.39}
\defresultn{c100.bacp.wanda.0.95}{3}
\defresult{c100.delta.wanda.0.95}{+4.27}

\defresult{c100.ip.wanda.0.97}{57.42}
\defresultsd{c100.ip.wanda.0.97}{1.16}
\defresultn{c100.ip.wanda.0.97}{3}
\defresult{c100.bacp.wanda.0.97}{66.28}
\defresultsd{c100.bacp.wanda.0.97}{0.34}
\defresultn{c100.bacp.wanda.0.97}{3}
\defresult{c100.delta.wanda.0.97}{+8.86}

\defresult{c100.ip.wanda.0.99}{30.40}
\defresultsd{c100.ip.wanda.0.99}{1.14}
\defresultn{c100.ip.wanda.0.99}{3}
\defresult{c100.bacp.wanda.0.99}{58.11}
\defresultsd{c100.bacp.wanda.0.99}{0.15}
\defresultn{c100.bacp.wanda.0.99}{3}
\defresult{c100.delta.wanda.0.99}{+27.71}

\defresult{c100.ip.wanda.0.999}{2.87}
\defresultsd{c100.ip.wanda.0.999}{1.35}
\defresultn{c100.ip.wanda.0.999}{3}
\defresult{c100.bacp.wanda.0.999}{12.51}
\defresultsd{c100.bacp.wanda.0.999}{0.84}
\defresultn{c100.bacp.wanda.0.999}{3}
\defresult{c100.delta.wanda.0.999}{+9.64}


% --- Paired significance test ----------------------------------------------
% Computed by tools/significance.py from the same records the accuracy macros
% come from; re-run it after any change to results/runs_grid and update these
% together. The unit is the CELL (model x criterion x sparsity) and the arms
% are paired BY SEED within a cell, since both arms of a seed share an
% initialisation and a dense checkpoint.
%
% Cells are NOT independent -- four backbones x three criteria x four
% sparsities share models and data -- so this is a statement about the grid as
% run, not a population claim, and the paper says so.
\defresult{summary.test.ncells}{48}
\defresult{summary.test.npositive}{35}
\defresult{summary.test.clears}{34}
\defresult{summary.test.signp}{0.0021}
\defresult{summary.test.wilcoxonp}{9.3\times10^{-6}}
\defresult{summary.test.wilcoxonw}{179}


% --- Objective ablation ----------------------------------------------------
% ResNet-50 / CIFAR-10 / magnitude, seed 1, under the PREVIOUS protocol
% (60 epochs, delta_T 88, no validation split).  Single seed.  The `abl`
% prefix on the ARM keeps this out of the headline namespace, so it can never
% be mistaken for, or silently substituted into, a matched-sweep cell.
%   ablip      = iterative-pruning baseline
%   ablcontrol = CAP-form objective with a frozen tied head
%   abllegacy  = the implemented BaCP objective (SupCon + NT-Xent, 2Bx2B,
%                trainable per-model projection heads)
%   abldelta   = abllegacy minus ablip
% Each is registered twice, under the spelling the sections use and under the
% `abl.<model>....` spelling, so either resolves to the same number.
\defresult{resnet50.ablip.magnitude.0.95}{91.71}
\defresult{abl.resnet50.ip.magnitude.0.95}{91.71}
\defresult{resnet50.ablcontrol.magnitude.0.95}{91.78}
\defresult{abl.resnet50.control.magnitude.0.95}{91.78}
\defresult{resnet50.abllegacy.magnitude.0.95}{92.43}
\defresult{abl.resnet50.legacy.magnitude.0.95}{92.43}
\defresult{resnet50.abldelta.magnitude.0.95}{+0.72}
\defresult{abl.resnet50.delta.magnitude.0.95}{+0.72}

\defresult{resnet50.ablip.magnitude.0.97}{91.35}
\defresult{abl.resnet50.ip.magnitude.0.97}{91.35}
\defresult{resnet50.ablcontrol.magnitude.0.97}{91.40}
\defresult{abl.resnet50.control.magnitude.0.97}{91.40}
\defresult{resnet50.abllegacy.magnitude.0.97}{92.04}
\defresult{abl.resnet50.legacy.magnitude.0.97}{92.04}
\defresult{resnet50.abldelta.magnitude.0.97}{+0.69}
\defresult{abl.resnet50.delta.magnitude.0.97}{+0.69}

\defresult{resnet50.ablip.magnitude.0.99}{89.80}
\defresult{abl.resnet50.ip.magnitude.0.99}{89.80}
% resnet50.ablcontrol.magnitude.0.99 is deliberately UNREGISTERED: that run was
% killed and never re-run.  The table wraps it in \resultif so the cell prints
% --- rather than a red ??, which would read as a broken document rather than
% as a cell that was never measured.  Registering it later needs no table edit.
\defresult{resnet50.abllegacy.magnitude.0.99}{91.24}
\defresult{abl.resnet50.legacy.magnitude.0.99}{91.24}
\defresult{resnet50.abldelta.magnitude.0.99}{+1.44}
\defresult{abl.resnet50.delta.magnitude.0.99}{+1.44}

% --- Dense ResNet-34 seed spread -------------------------------------------
% CIFAR-10 from scratch, 5 seeds.  Obtained with a DEFECTIVE STEM, so this is
% a measurement of run-to-run spread, NOT a dense baseline.  It is deliberately
% NOT registered as `resnet34.dense.none.0.0` for that reason.  The .run1..run5
% keys follow the order listed in the BRIEF; that order is not a seed mapping.
\defresult{probe.resnet34.densescratch}{91.64}
\defresultsd{probe.resnet34.densescratch}{0.30}
\defresultn{probe.resnet34.densescratch}{5}
\defresult{probe.resnet34.densescratch.run1}{91.6535}
\defresult{probe.resnet34.densescratch.run2}{91.5447}
\defresult{probe.resnet34.densescratch.run3}{91.5843}
\defresult{probe.resnet34.densescratch.run4}{91.2876}
\defresult{probe.resnet34.densescratch.run5}{92.1064}

% --- VGG learning-rate probe -----------------------------------------------
% vgg19 / CIFAR-10 / magnitude / 0.95, everything else held identical.
% Reported as a protocol detail: both VGGs carry zero BatchNorm layers and
% hold ~85% of their parameters in the classifier MLP, so they cannot absorb
% the contrastive phase's 0.1 step size.
\defresult{vgg19.probelr010.magnitude.0.95}{89.99}
\defresult{probe.vgg19.bacplr010.magnitude.0.95}{89.99}
\defresult{vgg19.probelr005.magnitude.0.95}{91.88}
\defresult{probe.vgg19.bacplr005.magnitude.0.95}{91.88}
\defresult{vgg19.probeip.magnitude.0.95}{91.02}
\defresult{probe.vgg19.ip.magnitude.0.95}{91.02}

% --- VGG-11 learning-rate confirmation -------------------------------------
% vgg11 / CIFAR-10 / magnitude, all four sparsities, everything else identical
% to the matched sweep (95_vgg11_lr_probe, run 441994225603083, sequential --
% run_parallel was tried first and measured 0.87x aggregate throughput on
% this workload, so it was re-run one cell at a time). The uniform-0.05
% choice was fixed from a SINGLE vgg19 cell before the sweep and applied to
% vgg11 without a vgg11-specific measurement; this closes that gap. 0.05 wins
% at 0.95, 0.97 and 0.99; 0.1 is marginally ahead only at 0.999, where both
% arms sit closest to the I.P. baseline. Mean over the four cells: 0.05 gives
% -0.57, 0.1 gives -0.87 -- 0.05 remains the better choice on the evidence,
% not merely the untested default.
\defresult{vgg11.probelr010.magnitude.0.95}{89.17}
\defresult{probe.vgg11.bacplr010.magnitude.0.95}{89.17}
\defresult{vgg11.probelr010.magnitude.0.97}{88.93}
\defresult{probe.vgg11.bacplr010.magnitude.0.97}{88.93}
\defresult{vgg11.probelr010.magnitude.0.99}{88.68}
\defresult{probe.vgg11.bacplr010.magnitude.0.99}{88.68}
\defresult{vgg11.probelr010.magnitude.0.999}{83.82}
\defresult{probe.vgg11.bacplr010.magnitude.0.999}{83.82}

% --- Noise floor -----------------------------------------------------------
\defresult{cifar10.noise.floor.lo}{0.1}
\defresult{probe.noisefloor.lo}{0.1}
\defresult{cifar10.noise.floor.hi}{0.3}
\defresult{probe.noisefloor.hi}{0.3}
"""

LITERATURE = r"""
% ==========================================================================
% LITERATURE -- published comparators, quoted exactly.  Not our measurements.
% Routed through the same macro so that no bare accuracy is ever typed into
% the paper, and so a transcription error has exactly one place to live.
%
% Protocol gap, which must be stated wherever these appear: GraNet's recipe is
% 160 epochs at batch 128, SGD(0.9) LR 0.1, x10 at [80,120], wd 5e-4.  Ours is
% 50+25 epochs at batch 512.  The controlled claim is our I.P.-vs-BaCP delta
% under a matched budget, never a state-of-the-art absolute.
% ==========================================================================

% --- GraNet Table 2, ResNet-50 / CIFAR-10 ----------------------------------
% Liu et al., NeurIPS 2021, arXiv:2106.10404.
\defresult{lit.granet.resnet50.dense}{94.75}
\defresultsd{lit.granet.resnet50.dense}{0.01}
% GMP at 95% is our exact cell.
\defresult{lit.granet.resnet50.gmp.0.95}{94.52}
\defresultsd{lit.granet.resnet50.gmp.0.95}{0.08}
% The BRIEF labels only the GMP row with a sparsity, so these carry no
% sparsity field rather than an assumed one.
\defresult{lit.granet.resnet50.snip}{90.86}
\defresult{lit.granet.resnet50.grasp}{91.32}
\defresult{lit.granet.resnet50.synflow}{91.22}
\defresult{lit.granet.resnet50.rigl}{93.86}
\defresultsd{lit.granet.resnet50.rigl}{0.25}
\defresult{lit.granet.resnet50.granet}{94.44}
\defresultsd{lit.granet.resnet50.granet}{0.01}

% --- VGG-19 / CIFAR-10 -----------------------------------------------------
% The two papers disagree on the dense figure; both are registered and the
% disagreement must be noted in the text rather than resolved silently.
\defresult{lit.granet.vgg19.dense}{93.85}
\defresultsd{lit.granet.vgg19.dense}{0.05}
\defresult{lit.grasp.vgg19.dense}{94.23}

\defresult{lit.granet.vgg19.gmp.0.90}{93.59}
\defresultsd{lit.granet.vgg19.gmp.0.90}{0.10}
\defresult{lit.granet.vgg19.snip.0.95}{93.43}
\defresultsd{lit.granet.vgg19.snip.0.95}{0.20}
\defresult{lit.granet.vgg19.snip.0.98}{92.05}
\defresultsd{lit.granet.vgg19.snip.0.98}{0.28}
\defresult{lit.granet.vgg19.grasp.0.95}{93.04}
\defresultsd{lit.granet.vgg19.grasp.0.95}{0.18}
\defresult{lit.granet.vgg19.grasp.0.98}{92.19}
\defresultsd{lit.granet.vgg19.grasp.0.98}{0.12}
\defresult{lit.granet.vgg19.rigl.0.90}{93.38}
\defresultsd{lit.granet.vgg19.rigl.0.90}{0.11}
"""


# --------------------------------------------------------------------------
# record -> key
# --------------------------------------------------------------------------

def fmt_sparsity(value) -> str:
    """0.95 -> '0.95', 0.999 -> '0.999'. Trailing zeros dropped, no exponent."""
    return f'{float(value):g}'


def fmt_acc(value) -> str:
    return f'{float(value):.2f}'


def fmt_sd(value) -> str:
    return f'{float(value):.2f}'


def fmt_delta(value) -> str:
    return f'{float(value):+.2f}'


def cell_key(rec) -> str | None:
    """The macro key a record belongs to, or None if it is not a paper cell."""
    model = rec.get('model_name')
    method = rec.get('method')
    if not model or not method:
        return None

    sparsity = rec.get('sparsity_requested')
    if method == 'dense' or sparsity in (None, '', 0, 0.0):
        # The sections spell a dense cell `<model>.dense.none.0.0`; render()
        # also emits the shorter `<model>.dense` alias.
        return f'{model}.dense.none.0.0'

    if '+' in method:
        arm, pruner = method.split('+', 1)
        arm = ARM_ALIASES.get(arm, arm.replace('_', '').replace('+', ''))
    else:
        # A bare pruner name is the iterative-pruning baseline: same schedule,
        # same criterion, same budget, cross-entropy objective.
        arm, pruner = 'ip', method

    return f'{model}.{arm}.{pruner}.{fmt_sparsity(sparsity)}'


def pick_metric(rec, preferred):
    """(value, which_field) for a record, preferring the sample-weighted figure."""
    order = [preferred] if preferred else []
    order += [f for f in ('test_acc_exact_pct', 'test_acc_pct') if f not in order]
    for field in order:
        value = rec.get(field)
        if value is None:
            continue
        try:
            return float(value), field
        except (TypeError, ValueError):
            continue
    return None, None


def usable(rec, skips) -> bool:
    if rec.get('status') != 'ok':
        skips['status not ok'] += 1
        return False
    if rec.get('is_smoke'):
        skips['smoke run'] += 1
        return False
    dropped = rec.get('eval_samples_dropped')
    if dropped:
        # The defect that voided every earlier ResNet-34 figure: a drop_last
        # eval loader scoring 9,728 of 10,000 test images.  Such a record must
        # never reach a table.
        skips['eval samples dropped'] += 1
        return False
    return True


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

def collect(records_dir: Path, preferred_metric, group_filter, skips):
    """{key: {seed: (value, metric_field, phase, ended_at)}}"""
    by_key = defaultdict(dict)
    phase_rank = {p: i for i, p in enumerate(PHASE_PRIORITY)}

    for path in sorted(records_dir.glob('*.json')):
        try:
            rec = json.loads(path.read_text(encoding='utf-8'))
        except Exception:                                          # noqa: BLE001
            skips['unreadable json'] += 1
            continue

        if group_filter and rec.get('experiment_group') != group_filter:
            skips['other experiment_group'] += 1
            continue
        if not usable(rec, skips):
            continue

        key = cell_key(rec)
        if key is None:
            skips['no model/method'] += 1
            continue

        value, field = pick_metric(rec, preferred_metric)
        if value is None:
            skips['no accuracy field'] += 1
            continue

        seed = rec.get('seed')
        seed = 'NA' if seed is None else int(seed)
        rank = phase_rank.get(rec.get('phase'), len(phase_rank))
        candidate = (rank, rec.get('ended_at') or '', value, field, rec.get('phase'))

        current = by_key[key].get(seed)
        # Lower phase rank wins; within a phase, the later record wins.
        better = (current is None
                  or candidate[0] < current[0]
                  or (candidate[0] == current[0] and candidate[1] > current[1]))
        if better:
            by_key[key][seed] = candidate

    return by_key


def aggregate(by_key):
    """{key: (mean, sd_or_None, n, metric_field)}"""
    out = {}
    for key, per_seed in by_key.items():
        values = [c[2] for c in per_seed.values()]
        fields = {c[3] for c in per_seed.values()}
        mean = statistics.fmean(values)
        sd = statistics.stdev(values) if len(values) > 1 else None
        out[key] = (mean, sd, len(values), '+'.join(sorted(fields)))
    return out


def deltas(agg):
    """bacp minus ip for every (model, pruner, sparsity) where both landed."""
    out = {}
    for key, (mean, _sd, _n, _f) in agg.items():
        parts = key.split('.')
        # model.bacp.pruner.<sparsity...> -- sparsity itself contains a dot.
        if len(parts) < 4 or parts[1] != 'bacp':
            continue
        model, pruner, sparsity = parts[0], parts[2], '.'.join(parts[3:])
        baseline = agg.get(f'{model}.ip.{pruner}.{sparsity}')
        if baseline is None:
            continue
        out[f'{model}.delta.{pruner}.{sparsity}'] = mean - baseline[0]
    return out


# --------------------------------------------------------------------------
# emit
# --------------------------------------------------------------------------

def summaries(delta_map):
    """Sweep-wide aggregates over the per-cell deltas.

    Keys are grouped so the prose can say "the margin widened with sparsity"
    and cite the four numbers that show it, rather than asserting a shape and
    leaving the reader to verify it against a 13-column table.
    """
    import statistics as _st
    if not delta_map:
        return {}

    vals = list(delta_map.values())
    by_sp, by_cr, by_md = {}, {}, {}
    for key, v in delta_map.items():
        # <model>.delta.<pruner>.<sparsity>
        model, _, pruner, sparsity = key.split('.', 3)
        by_sp.setdefault(sparsity, []).append(v)
        by_cr.setdefault(pruner, []).append(v)
        by_md.setdefault(model, []).append(v)

    out = {
        'summary.delta.mean': _st.fmean(vals),
        'summary.delta.max': max(vals),
        'summary.delta.min': min(vals),
    }
    for sp, a in by_sp.items():
        out[f'summary.delta.sparsity.{sp}'] = _st.fmean(a)
    for cr, a in by_cr.items():
        out[f'summary.delta.criterion.{cr}'] = _st.fmean(a)
    for md, a in by_md.items():
        out[f'summary.delta.model.{md}'] = _st.fmean(a)

    # counts are not deltas, so they are rendered verbatim rather than signed
    out['#summary.delta.npositive'] = str(sum(1 for v in vals if v > 0))
    out['#summary.delta.ncells'] = str(len(vals))
    return out


def render_summaries(summary):
    if not summary:
        return []
    lines = ['% --- sweep-wide aggregates, derived from the deltas above -----',
             '% Regenerated with the rest of this file; never hand-edit.']
    for key in sorted(summary):
        if key.startswith('#'):
            lines.append(f'\\defresult{{{key[1:]}}}{{{summary[key]}}}')
        else:
            lines.append(f'\\defresult{{{key}}}{{{fmt_delta(summary[key])}}}')
    lines.append('')
    return lines


def render(agg, delta_map, source: Path) -> str:
    lines = [
        '',
        '% ' + '=' * 72,
        '% GENERATED from the run records -- everything below this line is',
        f'% rebuilt on every invocation.  Source: {source.as_posix()}',
        '% ' + '=' * 72,
        '',
    ]

    if not agg:
        lines += [
            '% No usable records found.  Every headline key is therefore',
            '% unregistered and renders as a red ?? -- which is the correct',
            '% state for a sweep that has not run.',
            '',
        ]
        return '\n'.join(lines)

    for key in sorted(agg):
        mean, sd, n, field = agg[key]
        # A dense cell is spelled `<model>.dense.none.0.0` in the sections and
        # `<model>.dense` in this script's own docs.  Emit both.
        names = [key]
        if key.endswith('.dense.none.0.0'):
            names.append(key[:-len('.none.0.0')])
        lines.append(f'% {key}: n={n}, metric={field}')
        for name in names:
            lines.append(f'\\defresult{{{name}}}{{{fmt_acc(mean)}}}')
            if sd is not None:
                lines.append(f'\\defresultsd{{{name}}}{{{fmt_sd(sd)}}}')
            lines.append(f'\\defresultn{{{name}}}{{{n}}}')
        lines.append('')

    lines.extend(render_summaries(summaries(delta_map)))

    if delta_map:
        lines += ['% --- BaCP minus the matched iterative-pruning baseline ---', '']
        for key in sorted(delta_map):
            lines.append(f'\\defresult{{{key}}}{{{fmt_delta(delta_map[key])}}}')
        lines.append('')

    return '\n'.join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('records', nargs='?', default='results/runs',
                        help='directory of per-run JSON records (default: results/runs)')
    parser.add_argument('-o', '--out', default=None,
                        help='output .tex (default: ../results_macros.tex next to this script)')
    parser.add_argument('--metric', default=None,
                        help="accuracy field to prefer (default: test_acc_exact_pct, "
                             "falling back to test_acc_pct)")
    parser.add_argument('--group', default=None,
                        help='only use records with this experiment_group')
    parser.add_argument('--dry-run', action='store_true',
                        help='print to stdout instead of writing the file')
    args = parser.parse_args(argv)

    records_dir = Path(args.records)
    out_path = Path(args.out) if args.out \
        else Path(__file__).resolve().parent.parent / 'results_macros.tex'

    skips = defaultdict(int)
    if records_dir.is_dir():
        by_key = collect(records_dir, args.metric, args.group, skips)
    else:
        print(f'[results-macros] no such directory: {records_dir} -- '
              'emitting the hand-entered blocks only', file=sys.stderr)
        by_key = {}

    agg = aggregate(by_key)
    delta_map = deltas(agg)

    body = PREAMBLE + MEASURED + LITERATURE + render(agg, delta_map, records_dir)
    if not body.endswith('\n'):
        body += '\n'

    if args.dry_run:
        sys.stdout.write(body)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(body, encoding='utf-8', newline='\n')
        print(f'[results-macros] wrote {out_path}', file=sys.stderr)

    print(f'[results-macros] {len(agg)} cell(s), {len(delta_map)} delta(s) '
          f'from {records_dir}', file=sys.stderr)
    for reason, count in sorted(skips.items()):
        print(f'[results-macros]   skipped {count:4d}  {reason}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
