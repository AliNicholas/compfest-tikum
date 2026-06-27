"""Run dedel-main.ipynb as V8 with standard source_id target encoding.

This runner keeps the V6 notebook pipeline intact but changes the source_id
target encoding choice that the sanity check suggested was worth testing:
source_id uses the standard fold-safe TE path instead of anchored TE.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "dedel-main.ipynb"


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells: list[str] = []

    for idx, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))

        if idx == 6:
            source += """

# V8 override: use standard fold-safe TE for source_id as well.
# The original V6 add_te_full uses anchored source_id TE.
def add_te_full(X_tr0, y_tr, X_va0, X_te0):
    return add_te_fold(X_tr0, y_tr, X_va0, X_te0, te_source_cols)

print('[V8] add_te_full override active: source_id uses standard fold-safe TE')
"""

        if idx == 8:
            source = """
import lightgbm as lgb
print('[V8] skipped anchored-vs-standard sanity probe; running full standard source_id TE experiment')
"""

        code_cells.append(source)

    code = "\n\n".join(code_cells)
    replacements = {
        "checkpoint_v6.npz": "checkpoint_v8_standard_source_te.npz",
        "submission_v6.csv": "submission_v8_standard_source_te.csv",
        "model_summary_v6.csv": "model_summary_v8_standard_source_te.csv",
        "ensemble_v6_CHOSEN_": "ensemble_v8_standard_source_te_CHOSEN_",
        "ensemble_v6_": "ensemble_v8_standard_source_te_",
        "strategi {best_strategy}": "strategi V8 standard_source_te {best_strategy}",
    }
    for old, new in replacements.items():
        code = code.replace(old, new)

    exec_globals: dict[str, object] = {"__name__": "__main__"}
    exec(compile(code, str(NOTEBOOK), "exec"), exec_globals)


if __name__ == "__main__":
    main()
