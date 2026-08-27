#!/usr/bin/env python3
"""Double-blind anonymity scanner for the NeurIPS 2026 workshop submission.

Scans every ``*.tex`` and ``*.md`` file under the paper directory for strings
that would identify the author, and exits non-zero if it finds any.

Why this exists
---------------
The submission is double-blind.  The single largest de-anonymisation risk in
this project is not the author block -- the style file suppresses that -- but a
code-availability sentence citing the git remote, whose URL contains the
author's account name.  Any reproducibility statement that pastes that URL
de-anonymises the paper.  The rule this tool enforces is: promise release on
acceptance, or use an anonymized mirror, never the real URL.

Usage
-----
    python Paper/neurips2026/tools/anonymization_check.py
    python Paper/neurips2026/tools/anonymization_check.py --root Paper/neurips2026
    python Paper/neurips2026/tools/anonymization_check.py --strict
    python Paper/neurips2026/tools/anonymization_check.py --ignore-review

Exit codes
----------
    0   clean
    1   at least one hit remains
    2   bad invocation (root does not exist)

Severities
----------
    BLOCK    an identifying string.  Never acceptable in a submitted file.
    REVIEW   context-dependent: usually fine in repo notes, never fine in a
             compiled ``.tex``.  Counts as a hit by default; pass
             ``--ignore-review`` to exclude these from the exit code.

Suppressing a deliberate mention
--------------------------------
Put ``anon-check: ignore`` anywhere on the line (inside a ``%`` LaTeX comment or
an HTML comment in Markdown).  Use it only for prose that *discusses* the trap,
never to silence a real leak.

Two more sources of patterns are derived at run time, so a leak is caught even
if the hard-coded list goes stale: the ``origin`` git remote URL, and the git
``user.name`` / ``user.email`` of the checkout.

WARNING: this file necessarily contains the identifying strings it searches for.
It is a repo-local development tool.  Do NOT include ``tools/`` in any
supplementary-material upload.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

SUPPRESS = "anon-check: ignore"

# --------------------------------------------------------------------------
# Rules.  (name, severity, compiled regex, scope, note)
#   scope: 'any'  -> every scanned file
#          'tex'  -> only *.tex (these reach the compiled PDF)
# --------------------------------------------------------------------------
_RULES: list[tuple[str, str, str, str, str]] = [
    # ---- identity: the author -------------------------------------------
    ("author-name-given", "BLOCK", r"haroon", "any",
     "author's given name"),
    ("author-name-family", "BLOCK", r"khawaja", "any",
     "author's family name"),
    ("author-email", "BLOCK", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "any",
     "an email address"),

    # ---- identity: the code host ----------------------------------------
    ("github-account-url", "BLOCK", r"github\.com/[A-Za-z0-9._-]+", "any",
     "a GitHub URL naming an account -- THE anonymity trap for this paper"),
    ("gitlab-account-url", "BLOCK", r"gitlab\.com/[A-Za-z0-9._-]+", "any",
     "a GitLab URL naming an account"),

    # ---- identity: the compute environment -------------------------------
    ("databricks-workspace-id", "BLOCK", r"adb-\d{10,}", "any",
     "the Databricks workspace host id"),
    ("workspace-user-path", "BLOCK", r"/(?:Workspace/)?Users/[A-Za-z0-9._%+-]+", "any",
     "a workspace path containing a user account"),
    ("dbfs-path", "REVIEW", r"dbfs:/", "any",
     "a DBFS path -- points at a specific workspace"),
    ("local-windows-path", "BLOCK", r"[A-Za-z]:[\\/]Users[\\/]", "any",
     "an absolute local path containing a user directory"),
    ("cloud-account-hint", "REVIEW", r"\bdatabricks\b", "any",
     "names the specific platform used; harmless in repo notes, avoid in the PDF"),

    # ---- LaTeX constructs that break blinding ----------------------------
    ("latex-thanks", "BLOCK", r"\\thanks\b", "tex",
     "a \\thanks footnote"),
    ("latex-acknowledgements", "BLOCK", r"\\(?:sub)?section\*?\{\s*Acknowledg", "tex",
     "an acknowledgements section -- omit at submission"),
    ("latex-acknowledgements-env", "BLOCK", r"\\begin\{acks?\}|\\acknowledg", "tex",
     "an acknowledgements block"),
    ("funding", "REVIEW", r"\b(?:funded|supported)\s+by\b|\bgrant\s*(?:no\.?|number|#)", "tex",
     "a funding acknowledgement"),
    ("self-reference", "REVIEW", r"\b(?:our|my)\s+(?:previous|prior|earlier)\s+(?:work|paper|submission)\b", "tex",
     "an undisguised self-citation"),
    ("prior-submission", "REVIEW", r"\bAAAI\s+submission\b", "tex",
     "names an earlier submission of this work"),
]

# Placeholders: (name, pattern, fatal_under_strict).
# The anonymous author block is CORRECT for a double-blind submission, so it is
# reported for visibility but is never fatal -- it only matters when preparing a
# non-anonymous preprint or camera-ready build.
_PLACEHOLDERS = [
    ("todo-marker", r"\b(?:FIXME|TODO|XXX|PLACEHOLDER)\b", True),
    ("workshop-title", r"\\workshoptitle\{[^}]*(?:FIXME|WORKSHOP TITLE|TBD|\?\?)", True),
    ("author-block-anonymous", r"Anonymous Author\(s\)", False),
]


def _compile(rules):
    out = []
    for name, sev, pat, scope, note in rules:
        out.append((name, sev, re.compile(pat, re.IGNORECASE), scope, note))
    return out


def _derived_rules(repo_root: Path):
    """Patterns read from the checkout itself, so the list cannot go stale."""
    derived = []

    def _git(*args):
        try:
            r = subprocess.run(["git", *args], cwd=str(repo_root),
                               capture_output=True, text=True, timeout=10)
            return r.stdout.strip() if r.returncode == 0 else ""
        except Exception:
            return ""

    remote = _git("remote", "get-url", "origin")
    if remote:
        # The account segment of the remote is the thing that must never ship.
        m = re.search(r"[:/]([A-Za-z0-9._-]+)/[A-Za-z0-9._-]+?(?:\.git)?$", remote)
        if m and len(m.group(1)) > 2:
            derived.append(("git-remote-account", "BLOCK",
                            re.escape(m.group(1)), "any",
                            "the account name in this checkout's git remote"))

    for key, label in (("user.name", "git-user-name"),
                       ("user.email", "git-user-email")):
        value = _git("config", "--get", key)
        if value and len(value) > 2:
            for token in re.split(r"[\s@]+", value):
                if len(token) > 3 and token.lower() not in {"none", "user", "gmail.com"}:
                    derived.append((label, "BLOCK", re.escape(token), "any",
                                    f"the checkout's git {key}"))
    return derived


def scan(root: Path, rules, placeholders):
    hits, placeholder_hits = [], []
    files = sorted(
        p for ext in ("*.tex", "*.md")
        for p in root.rglob(ext)
        if ".git" not in p.parts
    )
    for path in files:
        scope_tex = path.suffix.lower() == ".tex"
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:                       # pragma: no cover
            print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if SUPPRESS in line:
                continue
            for name, sev, rx, scope, note in rules:
                if scope == "tex" and not scope_tex:
                    continue
                m = rx.search(line)
                if m:
                    hits.append((path, lineno, name, sev, m.group(0),
                                 line.strip()[:160], note))
            for name, pat, fatal in placeholders:
                m = re.search(pat, line)
                if m:
                    placeholder_hits.append((path, lineno, name, m.group(0),
                                             line.strip()[:160], fatal))
    return files, hits, placeholder_hits


def main() -> int:
    here = Path(__file__).resolve()
    default_root = here.parent.parent            # Paper/neurips2026
    repo_root = default_root.parent.parent       # repo root

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(default_root),
                    help="directory to scan (default: the paper directory)")
    ap.add_argument("--strict", action="store_true",
                    help="also fail on unresolved FIXME/TODO/placeholder markers")
    ap.add_argument("--ignore-review", action="store_true",
                    help="print REVIEW hits but do not let them set the exit code")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 2

    rules = _compile(_RULES) + _compile(_derived_rules(repo_root))
    files, hits, placeholders = scan(root, rules, _PLACEHOLDERS)

    print(f"anonymization_check: scanned {len(files)} file(s) under {root}")

    blocking = [h for h in hits if h[3] == "BLOCK"]
    review = [h for h in hits if h[3] == "REVIEW"]

    for label, group in (("BLOCK", blocking), ("REVIEW", review)):
        if not group:
            continue
        print(f"\n--- {label} ({len(group)}) ---")
        for path, lineno, name, _sev, matched, line, note in group:
            try:
                shown = path.relative_to(root)
            except ValueError:                       # pragma: no cover
                shown = path
            print(f"{shown}:{lineno}: [{label}] {name}: {note}")
            print(f"    matched: {matched!r}")
            print(f"    line:    {line}")

    fatal_placeholders = [p for p in placeholders if p[5]]
    if placeholders:
        print(f"\n--- PLACEHOLDERS ({len(placeholders)}, of which "
              f"{len(fatal_placeholders)} fatal under --strict) ---")
        for path, lineno, name, matched, line, fatal in placeholders:
            try:
                shown = path.relative_to(root)
            except ValueError:                       # pragma: no cover
                shown = path
            mark = "" if fatal else "  (informational)"
            print(f"{shown}:{lineno}: [{name}] {matched}{mark}")
            print(f"    line:    {line}")

    failed = bool(blocking)
    if review and not args.ignore_review:
        failed = True
    if fatal_placeholders and args.strict:
        failed = True

    print()
    if failed:
        print("anonymization_check: FAIL -- resolve the hits above before submitting.")
        return 1
    if review:
        print("anonymization_check: PASS (REVIEW hits present but not fatal).")
    else:
        print("anonymization_check: PASS -- no identifying strings found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
