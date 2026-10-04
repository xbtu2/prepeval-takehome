"""Labware facts and well addressing.

Ground truth is PyLabRobot (PLR, pinned commit in pyproject). PLR is only needed at data-generation /
test time: `labware.json` caches the facts PLR reports, and `tests/test_labware_plr.py` asserts that every
function here agrees with PLR on the real objects whenever PLR is importable.

Conventions (these are PLR's and are stated to the model in every prompt that needs them):
  * well names are row letter + 1-based column number ("A1", "H12", "P24");
  * the linear index of a well is column-major: A1=0, B1=1, ..., H1=7, A2=8 (PLR `get_item(int)`);
  * a range "A1:C3" is the rectangle spanned by its corners, listed row by row (PLR `expand_string_range`);
  * 96 -> 384 stamping into an "interleaved" quadrant: 96-well (r, c) -> 384-well (2r + qr, 2c + qc) with
    (qr, qc) = tl (0,0), tr (0,1), bl (1,0), br (1,1) (PLR `Plate.get_quadrant(..., "checkerboard")`).
"""

from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Iterable

PLR_COMMIT = "160bfe5dd4d8897834a11c088567192ffb379fbb"
_CACHE_PATH = Path(__file__).with_name("labware.json")

# key -> (PLR constructor name, kind). Facts (rows, cols, well max volume) are read from PLR and cached.
CATALOGUE_SPEC: dict[str, tuple[str, str]] = {
  "plate_96_flat": ("cor_96_wellplate_360uL_Fb", "plate"),
  "plate_96_deep": ("cor_96_wellplate_2mL_Vb", "deep-well plate"),
  "plate_384": ("biorad_384_wellplate_50uL_Vb", "plate"),
  "trough_60mL": ("hamilton_1_trough_60mL_Vb", "trough"),
}

QUADRANT_OFFSETS = {"tl": (0, 0), "tr": (0, 1), "bl": (1, 0), "br": (1, 1)}
QUADRANT_WORDS = {"tl": "top-left", "tr": "top-right", "bl": "bottom-left", "br": "bottom-right"}

_WELL_RE = re.compile(r"^([A-Za-z]{1,2})0*(\d{1,2})$")


# ----------------------------------------------------------------------------------------------------
# catalogue
# ----------------------------------------------------------------------------------------------------
def load_catalogue() -> dict[str, dict]:
  """Return {key: {plr_id, kind, rows, cols, well_max_uL, display}} from the cached JSON."""
  with _CACHE_PATH.open() as fh:
    return json.load(fh)["labware"]


def regenerate_catalogue_from_plr() -> dict:
  """Rebuild labware.json from live PLR objects. Requires pylabrobot (pinned commit)."""
  import warnings

  import pylabrobot  # noqa: F401
  import pylabrobot.resources as R

  warnings.simplefilter("ignore")
  labware: dict[str, dict] = {}
  for key, (ctor_name, kind) in CATALOGUE_SPEC.items():
    obj = getattr(R, ctor_name)(key)
    if kind == "trough":
      rows, cols, max_ul = 1, 1, float(obj.max_volume)
    else:
      rows, cols = obj.num_items_y, obj.num_items_x
      max_ul = float(obj.get_well("A1").max_volume)
    max_ul = round(max_ul, 1)
    labware[key] = {
      "plr_id": ctor_name,
      "kind": kind,
      "rows": rows,
      "cols": cols,
      "well_max_uL": max_ul,
      "display": _display(kind, rows, cols, max_ul),
    }
  payload = {"plr_commit": PLR_COMMIT, "plr_version": pylabrobot.__version__, "labware": labware}
  with _CACHE_PATH.open("w") as fh:
    json.dump(payload, fh, indent=2)
    fh.write("\n")
  return payload


def _display(kind: str, rows: int, cols: int, max_ul: float) -> str:
  vol = f"{max_ul:g} uL"
  if kind == "trough":
    return f"single-compartment trough, one shared liquid for all channels, max {vol}"
  return f"{rows * cols}-well {kind}, {rows} rows x {cols} columns, each well max {vol}"


# ----------------------------------------------------------------------------------------------------
# well addressing (pure python; cross-checked against PLR in tests)
# ----------------------------------------------------------------------------------------------------
def row_label(r: int) -> str:
  """0 -> 'A', 25 -> 'Z', 26 -> 'AA' (PLR's row_index_to_label)."""
  letters = string.ascii_uppercase
  if r < 26:
    return letters[r]
  return letters[r // 26 - 1] + letters[r % 26]


def row_index(label: str) -> int:
  label = label.upper()
  n = 0
  for ch in label:
    n = n * 26 + (ord(ch) - ord("A") + 1)
  return n - 1


def normalize_well(name: str) -> str:
  """'a01' -> 'A1'. Raises ValueError on anything that is not a well name."""
  m = _WELL_RE.match(str(name).strip())
  if not m:
    raise ValueError(f"not a well name: {name!r}")
  return f"{m.group(1).upper()}{int(m.group(2))}"


def parse_well(name: str) -> tuple[int, int]:
  """'C2' -> (row 2, col 1), both 0-based."""
  w = normalize_well(name)
  m = _WELL_RE.match(w)
  assert m
  return row_index(m.group(1)), int(m.group(2)) - 1


def well_name(r: int, c: int) -> str:
  return f"{row_label(r)}{c + 1}"


def on_plate(name: str, rows: int, cols: int) -> bool:
  try:
    r, c = parse_well(name)
  except ValueError:
    return False
  return 0 <= r < rows and 0 <= c < cols


def all_wells(rows: int, cols: int) -> list[str]:
  """Column-major order, matching PLR's linear item order."""
  return [well_name(r, c) for c in range(cols) for r in range(rows)]


def well_at_index(k: int, rows: int, cols: int) -> str:
  if not 0 <= k < rows * cols:
    raise IndexError(f"index {k} off a {rows}x{cols} plate")
  return well_name(k % rows, k // rows)


def index_of_well(name: str, rows: int, cols: int) -> int:
  r, c = parse_well(name)
  if not (0 <= r < rows and 0 <= c < cols):
    raise IndexError(f"{name} off a {rows}x{cols} plate")
  return c * rows + r


def expand_range(range_str: str, rows: int, cols: int) -> list[str]:
  """Rectangle spanned by the two corners, listed row by row (PLR `expand_string_range` order).

  Raises IndexError if any corner is off the plate (PLR `get_items` raises IndexError).
  """
  if ":" not in range_str:
    raise ValueError(f"invalid range: {range_str}")
  a, b = (normalize_well(x) for x in range_str.split(":"))
  (r0, c0), (r1, c1) = parse_well(a), parse_well(b)
  for r, c in ((r0, c0), (r1, c1)):
    if not (0 <= r < rows and 0 <= c < cols):
      raise IndexError(f"{well_name(r, c)} off a {rows}x{cols} plate")
  rr = range(r0, r1 + 1) if r0 <= r1 else range(r0, r1 - 1, -1)
  cc = range(c0, c1 + 1) if c0 <= c1 else range(c0, c1 - 1, -1)
  return [well_name(r, c) for r in rr for c in cc]


def column_wells(col_1based: int, rows: int, cols: int) -> list[str]:
  """Wells an N-channel head touches when placed on 1-based column `col_1based` (top to bottom)."""
  if not 1 <= col_1based <= cols:
    raise IndexError(f"column {col_1based} off a {rows}x{cols} plate")
  return [well_name(r, col_1based - 1) for r in range(rows)]


def row_wells(label: str, rows: int, cols: int) -> list[str]:
  r = row_index(label)
  if not 0 <= r < rows:
    raise IndexError(f"row {label} off a {rows}x{cols} plate")
  return [well_name(r, c) for c in range(cols)]


def quadrant_well_384(src_96: str, quadrant: str) -> str:
  """Destination well on a 384 plate when 96-well `src_96` is stamped into an interleaved quadrant."""
  if quadrant not in QUADRANT_OFFSETS:
    raise ValueError(f"quadrant must be one of {sorted(QUADRANT_OFFSETS)}")
  r, c = parse_well(src_96)
  if not (0 <= r < 8 and 0 <= c < 12):
    raise IndexError(f"{src_96} off a 96-well plate")
  qr, qc = QUADRANT_OFFSETS[quadrant]
  return well_name(2 * r + qr, 2 * c + qc)


def wells_equal(a: Iterable[str], b: Iterable[str]) -> bool:
  """Set equality after normalisation ('A01' == 'A1'); False if either side has a non-well entry."""
  try:
    return {normalize_well(x) for x in a} == {normalize_well(x) for x in b}
  except ValueError:
    return False


if __name__ == "__main__":  # pragma: no cover
  print(json.dumps(regenerate_catalogue_from_plr(), indent=2))
