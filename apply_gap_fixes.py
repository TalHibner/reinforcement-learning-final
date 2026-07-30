"""Apply the GAP_ANALYSIS.md fixes to 2026B_FinalProjectBugsFix.ipynb.

Run once:  python3 apply_gap_fixes.py

What it does
------------
* archives the current notebook (with its outputs) to
  drafts/2026B_FinalProjectBugsFix.preGapFix.ipynb
* copies the prepared cell sources into ./gap_fix_cells/ so they no longer
  depend on /tmp
* replaces 34 cells and inserts 12 new ones (see REPLACE / INSERT below)
* clears the outputs of every cell it rewrites, because a stored output that
  predates its source is exactly the "edited notebook" signal the audit flagged

Idempotent-ish: safe to re-run only against the *archived* notebook, since a
second run would replace already-replaced cells and insert the new cells again.
If you need to start over, restore from drafts/…preGapFix.ipynb first.
"""
import json
import os
import shutil
import sys

PROJ = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(PROJ, "2026B_FinalProjectBugsFix.ipynb")
ARCHIVE = os.path.join(PROJ, "drafts", "2026B_FinalProjectBugsFix.preGapFix.ipynb")

# Prepared cell sources: the session scratchpad first, then the copy kept in the
# project (written by preserve_sources on the first successful run).
SOURCE_CANDIDATES = [
    "/tmp/claude-1000/-home-hibta/e9037e8a-7b4e-4ae8-b3f2-b882cde784d0/scratchpad/new_cells",
    os.path.join(PROJ, "gap_fix_cells"),
]

# original cell index -> (source filename, expected cell_type)
REPLACE = {
    5:  ("cell_05.py", "code"),      # E5  single storage root, working fallback
    8:  ("cell_08.py", "code"),      # C7  drop unused imports, add json/Markdown
    10: ("cell_10.py", "code"),      # E5  stop clobbering STORAGE_ROOT
    31: ("cell_31.py", "code"),      # C2/C8/M7 logger, plots, merge, checkpointer
    33: ("cell_33.py", "code"),      # M8/LOW shaping: no positions getter, comment
    35: ("cell_35.py", "code"),      # M8  audited shaping constants
    40: ("cell_40.py", "code"),      # E1/E4/M3 buffer memory, orthogonal init
    41: ("cell_41.md", "markdown"),
    42: ("cell_42.py", "code"),      # M4  target sync on the env-step clock
    43: ("cell_43.md", "markdown"),
    44: ("cell_44.py", "code"),      # M2/E2 REINFORCE loss scales + baselines
    45: ("cell_45.md", "markdown"),
    46: ("cell_46.py", "code"),      # M1/M3 PPO truncation, minibatches, KL
    47: ("cell_47.md", "markdown"),  # C5  frame-stacking decision + memory notes
    49: ("cell_49.py", "code"),      # M1/M5/M7/E6/C7 training + eval loops
    50: ("cell_50.py", "code"),      # regression tests for M1, M2, M4, C2, C8
    51: ("cell_51.md", "markdown"),
    53: ("cell_53.py", "code"),
    55: ("cell_55.py", "code"),
    57: ("cell_57.py", "code"),
    61: ("cell_61.py", "code"),
    62: ("cell_62.py", "code"),      # LOW Categorical(logits=...)
    65: ("cell_65.md", "markdown"),
    67: ("cell_67.py", "code"),      # C2/C7 single call, no phantom notes
    69: ("cell_69.py", "code"),      # C7/M7 advisory comments out, real budget
    71: ("cell_71.py", "code"),
    73: ("cell_73.py", "code"),
    74: ("cell_74.py", "code"),
    76: ("cell_76.py", "code"),
    77: ("cell_77.py", "code"),
    78: ("cell_78.py", "code"),
    79: ("cell_79.md", "markdown"),  # C3  stale video text
    80: ("cell_80.py", "code"),      # C3  all six mid-training clips
    81: ("cell_81.md", "markdown"),  # C5/C6/C7/M6/M7/M8 discussion
}

# after original index -> [(source filename, cell_type), ...] in display order
INSERT = {
    35: [("insert_after_35_a.md", "markdown"),   # M8 shaping-budget audit
         ("insert_after_35_b.py", "code")],
    50: [("insert_after_50_a.md", "markdown"),   # C4 dedicated best-settings cell
         ("insert_after_50_b.py", "code"),
         ("insert_after_50_c.md", "markdown"),   # C6 reload from checkpoints
         ("insert_after_50_d.py", "code")],
    64: [("insert_after_64_a.md", "markdown"),   # C6 value-baseline ablation
         ("insert_after_64_b.py", "code")],
    78: [("insert_after_78_a.md", "markdown"),   # M7 generated results tables
         ("insert_after_78_b.py", "code")],
}


def find_sources():
    for path in SOURCE_CANDIDATES:
        if os.path.isdir(path):
            return path
    sys.exit("cell sources not found; looked in:\n  " + "\n  ".join(SOURCE_CANDIDATES))


def preserve_sources(src):
    dest = os.path.join(PROJ, "gap_fix_cells")
    if os.path.abspath(src) == os.path.abspath(dest):
        return
    os.makedirs(dest, exist_ok=True)
    for fname in sorted(os.listdir(src)):
        shutil.copy2(os.path.join(src, fname), os.path.join(dest, fname))
    print(f"cell sources preserved in {dest}")


def read_source(src, fname):
    with open(os.path.join(src, fname), encoding="utf-8") as f:
        text = f.read()
    if text.endswith("\n"):
        text = text[:-1]            # notebook sources carry no trailing newline
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def main():
    src = find_sources()
    preserve_sources(src)

    with open(NB, encoding="utf-8") as f:
        raw = f.read()
    nb = json.loads(raw)
    literal_unicode = any(ord(c) > 127 for c in raw)
    cells = nb["cells"]
    n_before = len(cells)
    used_ids = {c.get("id") for c in cells if c.get("id")}
    has_ids = bool(used_ids)

    if not os.path.exists(ARCHIVE):
        os.makedirs(os.path.dirname(ARCHIVE), exist_ok=True)
        shutil.copy2(NB, ARCHIVE)
        print(f"archived pre-fix notebook (with its outputs) -> {ARCHIVE}")

    for idx, (fname, ctype) in sorted(REPLACE.items()):
        cell = cells[idx]
        if cell["cell_type"] != ctype:
            sys.exit(f"cell {idx}: expected {ctype}, found {cell['cell_type']} — "
                     "the notebook is not the revision these fixes were built "
                     "against; restore from drafts/ and re-check.")
        cell["source"] = read_source(src, fname)
        if ctype == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        print(f"replaced cell {idx:>3} <- {fname}")

    counter = 0
    for idx in sorted(INSERT, reverse=True):
        for offset, (fname, ctype) in enumerate(INSERT[idx]):
            new_cell = {"cell_type": ctype}
            if has_ids:
                while True:
                    counter += 1
                    cid = f"cell-fix-{counter:03d}"
                    if cid not in used_ids:
                        used_ids.add(cid)
                        break
                new_cell["id"] = cid
            new_cell["metadata"] = {}
            new_cell["source"] = read_source(src, fname)
            if ctype == "code":
                new_cell["execution_count"] = None
                new_cell["outputs"] = []
            cells.insert(idx + 1 + offset, new_cell)
            print(f"inserted after {idx:>3} (+{offset}) <- {fname}")

    with open(NB, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=not literal_unicode)
        f.write("\n")

    print(f"\ncells: {n_before} -> {len(cells)}")
    print("Next: open the notebook and Run All (or set LOAD_FROM_ARTIFACTS=True to "
          "render from saved artifacts).")


if __name__ == "__main__":
    main()
