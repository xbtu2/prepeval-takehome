"""PyLabRobot is the ground truth: every addressing function and every cached labware fact must agree with
the real PLR objects. Skipped when pylabrobot is not importable (the eval itself never needs it)."""

from __future__ import annotations

import random
import warnings

import pytest

from prepeval import labware as L

plr = pytest.importorskip("pylabrobot")


def _name(w):
  return w.name.split("_well_")[1] if "_well_" in w.name else w.name


@pytest.fixture(scope="module")
def plates():
  warnings.simplefilter("ignore")
  import pylabrobot.resources as R

  return {k: getattr(R, v["plr_id"])(k) for k, v in L.load_catalogue().items()}


def test_cached_facts_match_live_plr(plates):
  cat = L.load_catalogue()
  for key, facts in cat.items():
    obj = plates[key]
    if facts["kind"] == "trough":
      assert round(float(obj.max_volume), 1) == facts["well_max_uL"]
    else:
      assert (obj.num_items_y, obj.num_items_x) == (facts["rows"], facts["cols"])
      assert round(float(obj.get_well("A1").max_volume), 1) == facts["well_max_uL"]


@pytest.mark.parametrize("key", ["plate_96_flat", "plate_384"])
def test_addressing_matches_plr(plates, key):
  from pylabrobot.utils import expand_string_range

  p = plates[key]
  rows, cols = p.num_items_y, p.num_items_x
  assert [_name(w) for w in p.get_all_items()] == L.all_wells(rows, cols)
  rng = random.Random(1)
  for _ in range(200):
    k = rng.randrange(rows * cols)
    assert _name(p.get_item(k)) == L.well_at_index(k, rows, cols)
    w = L.well_at_index(k, rows, cols)
    assert p.index_of_item(p.get_well(w)) == L.index_of_well(w, rows, cols)
    c = rng.randrange(cols)
    assert [_name(x) for x in p.column(c)] == L.column_wells(c + 1, rows, cols)
    a, b = L.well_at_index(rng.randrange(rows * cols), rows, cols), L.well_at_index(rng.randrange(rows * cols), rows, cols)
    assert [_name(x) for x in p.get_wells(f"{a}:{b}")] == L.expand_range(f"{a}:{b}", rows, cols)
    assert expand_string_range(f"{a}:{b}") == L.expand_range(f"{a}:{b}", rows, cols)
  with pytest.raises(IndexError):
    p.get_wells(f"A1:{L.row_label(rows)}1")
  with pytest.raises(IndexError):
    L.expand_range(f"A1:{L.row_label(rows)}1", rows, cols)


def test_quadrants_match_plr(plates):
  p384 = plates["plate_384"]
  for q in ("tl", "tr", "bl", "br"):
    plr_wells = [_name(w) for w in p384.get_quadrant(q, "checkerboard", "column-major")]
    ours = [L.quadrant_well_384(s, q) for s in L.all_wells(8, 12)]
    assert plr_wells == ours
