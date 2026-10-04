"""The notebook source compiles, the built .ipynb is in sync with it, and the live-narrative statistics behave."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import pytest

from prepeval import readings as RD
from prepeval import stats as ST
from prepeval.grading import Grade

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "notebook", "prep_eval_colab.py")
IPYNB = os.path.join(ROOT, "notebook", "prep_eval_colab.ipynb")


def test_notebook_source_compiles():
  src = open(PY, encoding="utf-8").read()
  compile(src, PY, "exec")
  assert "matplotlib.use(\"Agg\")" in src and "if not IN_KERNEL" in src, "the backend must only be forced outside a kernel"
  assert "plt.show()" not in src, "figures are displayed through show_fig, never plt.show()"


def test_ipynb_matches_source():
  jupytext = pytest.importorskip("jupytext")
  nbformat = pytest.importorskip("nbformat")
  from_py = jupytext.read(PY, fmt="py:percent")
  built = nbformat.read(IPYNB, as_version=4)
  a = [(c.cell_type, c.source.rstrip()) for c in from_py.cells]
  b = [(c.cell_type, c.source.rstrip()) for c in built.cells]
  assert a == b, "rebuild with: .venv/bin/jupytext --to ipynb notebook/prep_eval_colab.py"
  assert all(not c.get("outputs") for c in built.cells if c.cell_type == "code"), "the committed notebook carries no outputs"


# ----------------------------------------------------------------------------------------------------
# derive_readings on a tiny synthetic run
# ----------------------------------------------------------------------------------------------------
@dataclass
class _Item:
  id: str
  family: str
  feasible: bool
  schema: dict = field(default_factory=lambda: {"x_uL": "n", "feasible": "b"})


def _grade(it, accepted, pred_feasible, error_class=None, pred=None):
  return Grade(it.id, accepted, error_class=None if accepted else error_class, pred=pred, pred_feasible=pred_feasible)


def _synthetic():
  fams = ["dilute_stock", "serial_dilution", "normalize_samples"]
  items = [_Item(f"{f}-{i}", f, feasible=(i % 2 == 0)) for f in fams for i in range(4)]  # 12 items, 6 traps
  base, arm, rec_b, rec_a = {}, {}, [], []
  for k, it in enumerate(items):
    # baseline: accepts every other feasible item, refuses nothing, format-fails one
    b_acc = it.feasible and k % 4 == 0 and it.family != "normalize_samples"  # never solves a held-out item
    base[it.id] = _grade(it, b_acc, True, "wrong_value" if it.feasible else "feasibility", pred={"x_uL": 1, "feasible": True})
    rec_b.append({"item_id": it.id, "family": it.family, "raw": '```json\n{"x_uL": 1, "feasible": true}\n```', "graded_text": None,
                  "sandbox": None, "sandbox_err": None, "accepted": b_acc, "error_class": base[it.id].error_class, "hit_token_cap": False})
    # second arm: a tool arm that refuses every trap and solves feasible items outside the held-out family
    a_acc = (not it.feasible) or it.family != "normalize_samples"
    arm[it.id] = _grade(it, a_acc, it.feasible, "wrong_value", pred={"x_uL": 1, "feasible": it.feasible})  # refuses every trap
    rec_a.append({"item_id": it.id, "family": it.family, "raw": "```python\nresult = {}\n```", "graded_text": "{}",
                  "sandbox": "rejected" if k == 0 else "ok", "sandbox_err": "__SANDBOX_REJECTED__ SyntaxError" if k == 0 else "",
                  "accepted": a_acc, "error_class": arm[it.id].error_class, "hit_token_cap": k == 0})
  grades = {"baseline": base, "pot": arm}
  records = {"baseline": rec_b, "pot": rec_a}
  summary = ST.summarize(items, grades, baseline="baseline")
  return items, grades, records, summary


def test_derive_readings_shapes_and_edge_cases():
  items, grades, records, summary = _synthetic()
  R = RD.derive_readings(items, grades, records, summary, fewshot=None, caps={"baseline": 512, "pot": 320})
  assert R["n_items"] == 12 and R["n_traps"] == 6
  b, a = R["arms"]["baseline"], R["arms"]["pot"]
  assert b["n_with_reasoning"] == 0  # bare JSON
  assert a["n_with_reasoning"] is None  # tool arm: not measured
  assert b["refusals"]["fisher_p"] is None  # baseline never refuses: a zero margin
  assert a["refusals"]["fisher_p"] is not None and a["refusals"]["trap"] == 6
  for arm in R["arms"].values():
    sp = arm["accepted_split"]
    assert sp["solved_feasible"] + sp["trap_refusals"] == arm["accepted"]
    ho, tr = arm["split"]["heldout"], arm["split"]["in_train"]
    assert ho["n"] + tr["n"] == arm["n"] and ho["accepted"] + tr["accepted"] == arm["accepted"]
  assert R["pot_sandbox"]["rejected"] == 1 and R["pot_sandbox"]["rejected_all_cap_hits"] is True
  assert R["pot_sandbox"]["cap"] == 320 and R["pot_sandbox"]["other_cap"] == 512
  assert R["comparisons"]["pot"]["b_only"] + R["comparisons"]["pot"]["a_only"] == summary["comparisons"]["pot"]["discordant"]
  assert R["heldout_any_solved_feasible"] is False  # the tool arm only refuses traps in the held-out family
  import json

  json.dumps(R)  # plain JSON


def test_formatters_never_emit_nan():
  assert RD.fmt_num(float("nan")) == "n/a" and RD.fmt_ci([float("nan"), 1.0]) == "" and RD.fmt_p(None) == "p n/a"
  assert RD.fmt_p(1e-5) == "p < 0.001" and RD.fmt_p(0.0167) == "p = 0.017" and RD.pct(0.25) == "25 %"
  assert RD.frac(0, 0) == "0/0 (n/a)" and RD.fmt_signed(0.3708) == "+0.371"


def test_renderers_run_on_synthetic_data():
  items, grades, records, summary = _synthetic()
  controls = {"effective_floor": 0.5}
  R = RD.derive_readings(items, grades, records, summary, caps={"baseline": 512, "pot": 320})
  text = RD.render_prompting(summary, R, controls, {"baseline": 1.0, "pot": 2.0}, {"baseline": 512, "pot": 320}, 2, "CPU", smoke=True)
  assert "Smoke run" in text and "nan" not in text.lower().replace("fewshot", "") and "floor" in text
  assert RD.floor_lines(summary, controls).count("- `") == 2
  assert "did not run" in RD.render_sft(summary, R, controls, None)
