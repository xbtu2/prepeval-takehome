"""Core tests: hand-worked golds, grader controls, JSON extraction, dataset leak checks."""

from __future__ import annotations

import json

import pytest

from prepeval import dataset as D
from prepeval import families as F
from prepeval import grading as G
from prepeval import labware as L

ALL_TEST = [F.make_item(f, s, "test") for f in F.FAMILIES for s in range(40)]


# ----------------------------------------------------------------------------------------------------
# hand-worked golds (one per family), computed independently of the generator code
# ----------------------------------------------------------------------------------------------------
def _find(family, pred):
  for it in ALL_TEST:
    if it.family == family and pred(it):
      return it
  raise AssertionError("no item matches")


def test_dilute_stock_c1v1():
  it = _find("dilute_stock", lambda i: i.feasible)
  w = it.world
  # independent arithmetic from the world block
  s_val, s_unit = w["stock_conc"].split()
  t_val, t_unit = w["target_conc"].split()
  fam = next(f for _, f, *_ in F.REAGENTS if s_unit in f and t_unit in f)
  ratio = (float(s_val) * fam[s_unit]) / (float(t_val) * fam[t_unit])
  stock = w["final_volume_uL"] / ratio
  assert it.gold["stock_uL"] == pytest.approx(stock, rel=1e-4)
  assert it.gold["diluent_uL"] == pytest.approx(w["final_volume_uL"] - stock, rel=1e-4)


def test_serial_dilution_two_fold_example():
  it = _find("serial_dilution", lambda i: i.feasible and i.world["factor"] == 2)
  v = it.world["final_uL"]
  assert it.gold["carry_uL"] == pytest.approx(v)  # 2-fold: carry equals the final volume
  assert it.gold["diluent_per_well_uL"] == pytest.approx(v)
  ckey = next(k for k in it.gold if k.startswith("concentrations"))
  assert it.gold[ckey][1] == pytest.approx(it.world["c0"] / 2)
  assert len(it.gold[ckey]) == it.world["points"]


def test_master_mix_totals():
  it = _find("master_mix", lambda i: i.feasible)
  w = it.world
  n_prep = w["N"] * (1 + w["excess_pct"] / 100)
  assert it.gold["reactions_prepared"] == pytest.approx(n_prep)
  comp_sum = sum(v for _, v in w["components"])
  assert it.gold["water_uL"] == pytest.approx((w["R"] - comp_sum - w["template_uL"]) * n_prep, rel=1e-4)
  for name, v in w["components"]:
    assert it.gold["component_totals_uL"][name] == pytest.approx(v * n_prep, rel=1e-4)


def test_normalize_outcome():
  it = _find("normalize_samples", lambda i: i.feasible)
  p = it.grader["params"]
  for i in range(3):
    s, d = it.gold[f"S{i + 1}_sample_uL"], it.gold[f"S{i + 1}_diluent_uL"]
    assert p["concs"][i] * s / (s + d) == pytest.approx(p["Ct"], rel=1e-3)
    assert s + d >= p["Vmin"] - G.ABS_UL and s <= p["avail"][i] + G.ABS_UL and s >= p["pmin"] - G.ABS_UL


def test_well_addressing_known_cases():
  assert L.expand_range("A1:C3", 8, 12) == ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
  assert L.column_wells(5, 8, 12) == ["A5", "B5", "C5", "D5", "E5", "F5", "G5", "H5"]
  assert L.well_at_index(0, 8, 12) == "A1" and L.well_at_index(8, 8, 12) == "A2" and L.well_at_index(10, 8, 12) == "C2"
  assert L.quadrant_well_384("A1", "tl") == "A1" and L.quadrant_well_384("B1", "tl") == "C1"
  assert L.quadrant_well_384("A2", "tr") == "A4" and L.quadrant_well_384("H12", "br") == "P24"
  with pytest.raises(IndexError):
    L.expand_range("A10:C13", 8, 12)
  assert L.normalize_well("a01") == "A1"


def test_labware_fit_trough_load():
  it = _find("labware_fit", lambda i: i.feasible and i.world["sub"] == 0)
  assert it.gold["source_container"] == "C4"
  # load = v * 96 * N + dead; recover v and N from the prompt
  import re

  v = float(re.search(r"Dispense ([\d.]+) µL", it.prompt).group(1))
  n = int(re.search(r"every well of (\d+) 96-well", it.prompt).group(1))
  assert it.gold["load_uL"] == pytest.approx(v * 96 * n + it.world["dead_uL"])


# ----------------------------------------------------------------------------------------------------
# grader controls
# ----------------------------------------------------------------------------------------------------
def test_gold_accepted_everywhere():
  bad = [it.id for it in ALL_TEST if not G.grade(it, G.gold_completion(it)).accepted]
  assert bad == []


def test_every_mutation_rejected():
  leaks = [(it.id, n) for it in ALL_TEST for n, txt in G.mutations(it).items() if G.grade(it, txt).accepted]
  assert leaks == []


def test_constant_policies_sit_at_the_floor():
  n_trap = sum(not it.feasible for it in ALL_TEST)
  always_feasible = sum(G.grade(it, G.policy_always_feasible_zeros(it)).accepted for it in ALL_TEST)
  always_infeasible = sum(G.grade(it, G.policy_always_infeasible(it)).accepted for it in ALL_TEST)
  assert always_feasible == 0
  assert always_infeasible == n_trap
  assert n_trap == len(ALL_TEST) // 5  # stratified: every 5th seed


def test_refusing_a_feasible_item_is_a_feasibility_error():
  it = _find("dilute_stock", lambda i: i.feasible)
  g = G.grade(it, G.policy_always_infeasible(it))
  assert not g.accepted and g.error_class == "feasibility"


def test_acting_on_an_infeasible_item_scores_zero():
  it = _find("dilute_stock", lambda i: not i.feasible)
  g = G.grade(it, json.dumps({"stock_uL": 1.0, "diluent_uL": 99.0, "feasible": True}))
  assert not g.accepted and g.error_class == "feasibility"


# ----------------------------------------------------------------------------------------------------
# extraction robustness
# ----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
  "text,expect",
  [
    ('Reasoning...\n```json\n{"stock_uL": 40, "diluent_uL": 60, "feasible": true}\n```', True),
    ('so {"stock_uL": 40.0, "diluent_uL": 60.0, "feasible": true}. That is the answer.', True),
    ("{'stock_uL': 40, 'diluent_uL': 60, 'feasible': True}", True),
    ('{"stock_uL": "40 µL", "diluent_uL": "60", "feasible": "true"}', True),
    ('{"stock_ul": 40, "diluent_ul": 60, "feasible": true}', True),
    ('first {"x": 1} then final {"stock_uL": 40, "diluent_uL": 60, "feasible": true}', True),
    ('{"stock_uL": "<number>", "diluent_uL": "<number>", "feasible": "<true|false>"}', False),
    ("I think 40 and 60.", False),
    ('{"stock_uL": 40, "diluent_uL": 61, "feasible": true}', False),
  ],
)
def test_extraction_and_tolerance(text, expect):
  it = F.Item(id="x", family="dilute_stock", template=0, seed=0, split="test", prompt="", world={}, schema={},
              gold={"stock_uL": 40.0, "diluent_uL": 60.0, "feasible": True}, feasible=True,
              grader={"kinds": {"stock_uL": "volume", "diluent_uL": "volume", "feasible": "bool"}, "params": {}}, trace="")
  assert G.grade(it, text).accepted is expect


def test_units_equivalence_in_well_lists():
  it = _find("well_addressing", lambda i: i.feasible and i.world["sub"] == 2)
  lower = [w.lower().replace("1", "01") if w.endswith("1") else w.lower() for w in it.gold["wells"]]
  assert G.grade(it, json.dumps({"wells": lower, "count": 8, "feasible": True})).accepted


# ----------------------------------------------------------------------------------------------------
# dataset / leak checks
# ----------------------------------------------------------------------------------------------------
def test_dataset_leak_checks_pass():
  test, train, fewshot = D.build_items()
  rep = D.leak_check(test, train, fewshot)
  assert rep["PASS"], rep
  assert len(test) == 240 and 1400 <= len(train) <= 1500 and len(fewshot) == 2
  assert not any(it.family in F.HELDOUT_FAMILIES for it in train)


def test_items_are_deterministic():
  a = F.make_item("master_mix", 7, "test")
  b = F.make_item("master_mix", 7, "test")
  assert a.to_json() == b.to_json()
  assert F.make_item("master_mix", 7, "train").prompt != a.prompt
