"""Figure smoke tests: every figure renders from synthetic data, and returns None on empty / missing inputs."""

from __future__ import annotations

import io
import math
import warnings
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from prepeval import figures as FG  # noqa: E402
from prepeval.families import FAMILIES, HELDOUT_FAMILIES  # noqa: E402
from prepeval.grading import Grade  # noqa: E402
from prepeval.stats import summarize  # noqa: E402

ARMS = ["baseline", "fewshot", "sft"]  # "pot" is deliberately missing


@dataclass
class FakeItem:
  id: str
  family: str
  feasible: bool


def make_items() -> list[FakeItem]:
  return [FakeItem(f"{fam}-{k}", fam, k != 3) for fam in FAMILIES for k in range(4)]  # one trap per family


def _outcome(ai: int, i: int, feasible: bool) -> tuple[bool, str | None, bool | None]:
  """Deterministic (accepted, error_class, pred_feasible); later arms accept more."""
  r = (i * 7 + ai * 3) % 10
  if not feasible:
    return (True, None, False) if r < 4 + 2 * ai else (False, "feasibility", True)
  if r < 3 + 2 * ai:
    return True, None, True
  if r in (7, 8):
    return False, "wrong_value", True
  if r == 9:
    return False, "feasibility", False
  return False, "format", None


def make_arms(items):
  grades_by_arm, records_by_arm = {}, {}
  for ai, arm in enumerate(ARMS):
    grades, records = {}, []
    for i, it in enumerate(items):
      acc, cls, pf = _outcome(ai, i, it.feasible)
      grades[it.id] = Grade(it.id, acc, error_class=cls, pred_feasible=pf)
      records.append({"item_id": it.id, "family": it.family, "raw": "{}", "graded_text": None, "sandbox": None,
                      "accepted": acc, "error_class": cls, "hit_token_cap": False, "sandbox_err": None})
    grades_by_arm[arm] = grades
    records_by_arm[arm] = records
  return grades_by_arm, records_by_arm


def make_readings(items, grades_by_arm) -> dict:
  out = {"arms": {}}
  for arm, grades in grades_by_arm.items():
    traps = [it for it in items if not it.feasible]
    feas = [it for it in items if it.feasible]
    refusals = {"trap": sum(grades[it.id].pred_feasible is False for it in traps), "trap_n": len(traps),
                "feasible": sum(grades[it.id].pred_feasible is False for it in feas), "feasible_n": len(feas),
                "fisher_p": 0.0004 if arm == "sft" else (None if arm == "fewshot" else 0.31)}
    split = {}
    for key, fams in (("in_train", [f for f in FAMILIES if f not in HELDOUT_FAMILIES]), ("heldout", HELDOUT_FAMILIES)):
      sub = [it for it in items if it.family in fams]
      acc = [it for it in sub if grades[it.id].accepted]
      split[key] = {"n": len(sub), "accepted": len(acc), "solved_feasible": sum(it.feasible for it in acc),
                    "trap_refusals": sum(not it.feasible for it in acc), "acc": len(acc) / len(sub)}
    out["arms"][arm] = {"refusals": refusals, "split": split}
  return out


def make_fixture() -> dict:
  items = make_items()
  grades_by_arm, records_by_arm = make_arms(items)
  summary = summarize(items, grades_by_arm)
  return {
    "test": items, "grades_by_arm": grades_by_arm, "records_by_arm": records_by_arm, "summary": summary,
    "readings": make_readings(items, grades_by_arm), "controls": {"effective_floor": 0.25},
    "lens": [412, 455, 470, 498, 503, 517, 522, 540, 561, 575, 590, 611, 640, 672, 705, 760],
    "log_history": [{"loss": 1.42, "step": 1, "learning_rate": 2e-4, "epoch": 0.1},
                    {"loss": 0.77, "step": 2, "learning_rate": 1e-4, "epoch": 0.5},
                    {"loss": 0.41, "step": 3, "learning_rate": 0.0, "epoch": 1.0},
                    {"train_loss": 0.86, "step": 3, "epoch": 1.0}],
  }


@pytest.fixture(scope="module")
def fx():
  return make_fixture()


@pytest.fixture(autouse=True)
def _close_figures():
  yield
  plt.close("all")


def _assert_renders(fig):
  assert isinstance(fig, Figure)
  buf = io.BytesIO()
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    fig.savefig(buf, format="png", dpi=72)
  assert buf.getvalue()[:4] == b"\x89PNG"


def test_summary_shape(fx):
  s = fx["summary"]
  assert set(s["arms"]) == set(ARMS)
  assert math.isnan(s["arms"]["sft"]["per_family"]["labware_fit"]["ci95"][0])  # n < 10 per family
  assert "sft" in s["comparisons"] and "baseline" not in s["comparisons"]
  classes = {c for a in ARMS for c in s["arms"][a]["error_classes"]}
  assert classes == {"format", "feasibility", "wrong_value"}


def test_fmt_p():
  assert FG.fmt_p(None) == "p n/a"
  assert FG.fmt_p(float("nan")) == "p n/a"
  assert FG.fmt_p(0.0004) == "p < 0.001"
  assert FG.fmt_p(0.034567) == "p = 0.035"
  assert FG.fmt_p(1.0) == "p = 1"


def test_fig_acceptance(fx):
  _assert_renders(FG.fig_acceptance(fx["summary"], fx["controls"]))
  _assert_renders(FG.fig_acceptance(fx["summary"], fx["controls"], families=FAMILIES))
  assert FG.fig_acceptance({"n_items": 0, "arms": {}, "comparisons": {}}, fx["controls"]) is None


def test_fig_outcomes(fx):
  _assert_renders(FG.fig_outcomes(fx["records_by_arm"], fx["test"], ["baseline", "fewshot", "pot", "sft"]))
  _assert_renders(FG.fig_outcomes(fx["records_by_arm"], fx["test"], ["sft"]))
  assert FG.fig_outcomes(fx["records_by_arm"], fx["test"], ["pot"]) is None
  assert FG.fig_outcomes({}, fx["test"], ARMS) is None
  assert FG.fig_outcomes(fx["records_by_arm"], [], ARMS) is None


def test_fig_refusals(fx):
  _assert_renders(FG.fig_refusals(fx["summary"], fx["readings"]))
  assert FG.fig_refusals(fx["summary"], {"arms": {}}) is None
  assert FG.fig_refusals({"arms": {}}, fx["readings"]) is None


def test_fig_train_lengths(fx):
  _assert_renders(FG.fig_train_lengths(fx["lens"], 1024))
  _assert_renders(FG.fig_train_lengths(list(range(300, 1100, 7)), 1024))  # >= 50 values -> 32-token bins
  assert FG.fig_train_lengths([], 1024) is None


def test_fig_loss_curve(fx):
  _assert_renders(FG.fig_loss_curve(fx["log_history"]))
  _assert_renders(FG.fig_loss_curve([{"loss": 50.0, "step": 1}, {"loss": 1.0, "step": 2}]))  # log scale
  assert FG.fig_loss_curve([]) is None
  assert FG.fig_loss_curve([{"loss": 1.0, "step": 1}]) is None
  assert FG.fig_loss_curve([{"train_loss": 1.0, "step": 3}]) is None


def test_fig_transitions(fx):
  cmp = fx["summary"]["comparisons"]["sft"]
  _assert_renders(FG.fig_transitions(fx["grades_by_arm"], fx["test"]))
  _assert_renders(FG.fig_transitions(fx["grades_by_arm"], fx["test"], "baseline", "fewshot", comparison=cmp))
  assert FG.fig_transitions(fx["grades_by_arm"], fx["test"], "baseline", "pot") is None
  assert FG.fig_transitions(fx["grades_by_arm"], [], "baseline", "sft") is None


def test_fig_heldout_split(fx):
  _assert_renders(FG.fig_heldout_split(fx["summary"], fx["readings"]))
  assert FG.fig_heldout_split(fx["summary"], {"arms": {}}) is None
