"""Programmatic grading. No LLM judge.

A completion is graded by (1) extracting the last JSON object it contains, (2) checking every required key
against the gold with a kind-specific rule, (3) accepting the item only if every key passes. When the gold is
infeasible, only the `feasible` flag is graded ("a specific report scores 1.0; acting anyway scores 0").
Each failed item gets one error class: `format` (no usable JSON), `feasibility` (flag wrong), `wrong_value`.

Kinds (item.grader["kinds"]):
  volume        |pred-gold| <= max(ABS_UL, REL*|gold|)
  number        |pred-gold| <= REL*|gold| (or ABS_NUM when gold == 0)
  conc_list     same length, elementwise `number`
  volume_dict   same component names (normalised), each `volume`
  count         exact integer
  wells         exact set equality after well-name normalisation ("A01" == "A1")
  well          exact well-name equality after normalisation
  choice        exact string equality (case/space-insensitive)
  bool          exact
  normalize_pair / paired   outcome-based: (sample, diluent) must hit the target concentration within REL,
                total >= Vmin, sample <= available, sample >= pipette minimum (any construction passes)
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field

from . import labware as L
from .families import Item

ABS_UL = 0.05
REL = 0.01
ABS_NUM = 1e-6

_FENCE_RE = re.compile(r"```(?:json|python|JSON)?\s*|```", re.IGNORECASE)
_NUM_RE = re.compile(r"[-+]?(?:\d+(?:[.,]\d+)?|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass
class Grade:
  item_id: str
  accepted: bool
  measured: bool = True
  error_class: str | None = None  # format | feasibility | wrong_value | None
  axes: dict = field(default_factory=dict)  # key -> {"pass": bool, "pred": ..., "gold": ..., "note": str}
  pred: dict | None = None
  pred_feasible: bool | None = None

  def to_json(self) -> dict:
    return {"item_id": self.item_id, "accepted": self.accepted, "measured": self.measured,
            "error_class": self.error_class, "axes": self.axes, "pred": self.pred, "pred_feasible": self.pred_feasible}


# ----------------------------------------------------------------------------------------------------
# extraction
# ----------------------------------------------------------------------------------------------------
def norm_key(k: str) -> str:
  k = str(k).strip().lower().replace("µ", "u").replace("μ", "u")
  k = re.sub(r"\s+", "_", k)
  return k


def _try_parse(s: str):
  try:
    return json.loads(s)
  except Exception:
    pass
  try:
    py = re.sub(r"\btrue\b", "True", s)
    py = re.sub(r"\bfalse\b", "False", py)
    py = re.sub(r"\bnull\b", "None", py)
    v = ast.literal_eval(py)
    return v if isinstance(v, dict) else None
  except Exception:
    return None


def extract_json(text: str) -> dict | None:
  """Last balanced {...} in `text` that parses as a dict (JSON, or a Python dict literal).

  Prefers the last object that carries a `feasible` key, so a stray trailing object (a note, a restated
  schema) does not shadow the answer."""
  if not text:
    return None
  s = _FENCE_RE.sub(" ", text)
  end = len(s)
  fallback = None
  for _ in range(6):  # try the last few candidate objects
    close = s.rfind("}", 0, end)
    if close < 0:
      return None
    depth, i, in_str, esc = 0, close, False, False
    start = -1
    while i >= 0:
      ch = s[i]
      if in_str:
        # walking backwards: a quote ends the string unless escaped (approximate but adequate)
        if ch == '"' and not (i > 0 and s[i - 1] == "\\"):
          in_str = False
      else:
        if ch == '"':
          in_str = True
        elif ch == "}":
          depth += 1
        elif ch == "{":
          depth -= 1
          if depth == 0:
            start = i
            break
      i -= 1
    if start >= 0:
      obj = _try_parse(s[start : close + 1])
      if isinstance(obj, dict):
        normed = {norm_key(k): v for k, v in obj.items()}
        if "feasible" in normed:
          return normed
        fallback = fallback or normed
    end = close  # retry with an earlier object
  return fallback


def coerce_number(v):
  """Numbers, numeric strings ('20 µL', '1,000', '2.5e3'); bool/None/other -> None."""
  if isinstance(v, bool) or v is None:
    return None
  if isinstance(v, (int, float)):
    return float(v)
  if isinstance(v, str):
    m = _NUM_RE.search(v.replace(",", ""))
    if m:
      try:
        return float(m.group(0))
      except ValueError:
        return None
  return None


def coerce_bool(v):
  if isinstance(v, bool):
    return v
  if isinstance(v, str):
    s = v.strip().lower()
    if "false" in s or "infeasible" in s or "not feasible" in s or s.startswith("no"):
      return False
    if "true" in s or s.startswith("yes") or s in ("feasible", "1", "y", "t"):
      return True
  if isinstance(v, (int, float)) and v in (0, 1):
    return bool(v)
  return None


# ----------------------------------------------------------------------------------------------------
# per-kind checks
# ----------------------------------------------------------------------------------------------------
def _close(pred: float, gold: float, abs_tol: float) -> bool:
  return abs(pred - gold) <= max(abs_tol, REL * abs(gold))


def _check_volume(pred, gold):
  p = coerce_number(pred)
  return p is not None and _close(p, float(gold), ABS_UL), p


def _check_number(pred, gold):
  p = coerce_number(pred)
  return p is not None and _close(p, float(gold), ABS_NUM), p


def _check_conc_list(pred, gold):
  if isinstance(pred, dict):
    pred = list(pred.values())
  if not isinstance(pred, (list, tuple)) or len(pred) != len(gold):
    return False, pred
  ps = [coerce_number(x) for x in pred]
  return all(p is not None and _close(p, float(g), ABS_NUM) for p, g in zip(ps, gold)), ps


def _check_volume_dict(pred, gold):
  if not isinstance(pred, dict):
    return False, pred
  pn = {norm_key(k): v for k, v in pred.items()}
  ok = True
  out = {}
  used: set = set()
  for name, g in gold.items():
    key = norm_key(name)
    match = key if key in pn else None
    if match is None:  # tolerate minor renames, but each predicted key may serve only one component
      cands = [k for k in pn if (key in k or k in key) and k not in used]
      match = cands[0] if len(cands) == 1 else None
    if match is None or match in used:
      out[name] = None
      ok = False
      continue
    used.add(match)
    p = coerce_number(pn[match])
    out[name] = p
    ok = ok and p is not None and _close(p, float(g), ABS_UL)
  return ok, out


def _check_count(pred, gold):
  p = coerce_number(pred)
  return p is not None and float(p).is_integer() and int(p) == int(gold), p


def _check_wells(pred, gold, params=None):
  params = params or {}
  if isinstance(pred, str):
    pred = [x for x in re.split(r"[,;\s]+(?![^:]*\b$)|[,;]\s*|\s+", pred) if x]
  if not isinstance(pred, (list, tuple)):
    return False, pred
  names = []
  for x in pred:
    x = re.sub(r"^(?:well|wells)\s+", "", str(x).strip(), flags=re.IGNORECASE)
    if ":" in x and "rows" in params:  # a range such as "A5:H5" names the same wells as the list
      try:
        names.extend(L.expand_range(x, params["rows"], params["cols"]))
        continue
      except (ValueError, IndexError):
        return False, pred
    names.append(x)
  return L.wells_equal(names, gold), names


def _check_well(pred, gold):
  try:
    return L.normalize_well(str(pred)) == L.normalize_well(gold), pred
  except ValueError:
    return False, pred


def _check_choice(pred, gold):
  return isinstance(pred, str) and pred.strip().upper() == str(gold).strip().upper(), pred


def _check_normalize_pair(pred_s, pred_d, i: int, params: dict):
  s, d = coerce_number(pred_s), coerce_number(pred_d)
  if s is None or d is None or s <= 0 or d < 0:
    return False, (s, d)
  c = params["concs"][i]
  total = s + d
  conc = c * s / total
  # concentration within REL; volume constraints within the same absolute band as every other volume
  ok = (_close(conc, params["Ct"], ABS_NUM) and total >= params["Vmin"] - ABS_UL
        and s <= params["avail"][i] + ABS_UL and s >= params["pmin"] - 1e-6)
  return ok, (s, d)


# ----------------------------------------------------------------------------------------------------
# item grading
# ----------------------------------------------------------------------------------------------------
def grade(item: Item, completion: str) -> Grade:
  pred = extract_json(completion or "")
  if pred is None:
    return Grade(item.id, False, error_class="format", pred=None)
  kinds = item.grader["kinds"]
  params = item.grader.get("params", {})
  axes: dict = {}

  pf = coerce_bool(pred.get("feasible"))
  if pf is None:
    # a schema echo or missing flag: treat as unparsable output
    return Grade(item.id, False, error_class="format", pred=pred)

  # infeasible gold: only the report channel is graded
  if not item.feasible:
    ok = pf is False
    axes["feasible"] = {"pass": ok, "pred": pf, "gold": False}
    return Grade(item.id, ok, error_class=None if ok else "feasibility", axes=axes, pred=pred, pred_feasible=pf)

  axes["feasible"] = {"pass": pf is True, "pred": pf, "gold": True}
  if pf is not True:
    for k in kinds:
      if k != "feasible":
        axes[k] = {"pass": False, "pred": pred.get(norm_key(k)), "gold": item.gold.get(k), "note": "refused a feasible request"}
    return Grade(item.id, False, error_class="feasibility", axes=axes, pred=pred, pred_feasible=pf)

  all_ok = True
  for key, kind in kinds.items():
    if key == "feasible" or kind == "paired":
      continue
    gold = item.gold[key]
    pv = pred.get(norm_key(key))
    if pv is None and kind != "normalize_pair":
      # tolerate a unit-suffix rename like "stock_ul" -> "stock_volume_ul"
      stem = norm_key(key).replace("_ul", "")
      cands = [k for k in pred if stem and (k.startswith(stem) or stem in k)]
      pv = pred[cands[0]] if len(cands) == 1 else None
    if kind == "volume":
      ok, p = _check_volume(pv, gold)
    elif kind == "number":
      ok, p = _check_number(pv, gold)
    elif kind == "conc_list":
      ok, p = _check_conc_list(pv, gold)
    elif kind == "volume_dict":
      ok, p = _check_volume_dict(pv, gold)
    elif kind == "count":
      ok, p = _check_count(pv, gold)
    elif kind == "wells":
      ok, p = _check_wells(pv, gold, params)
    elif kind == "well":
      ok, p = _check_well(pv, gold)
    elif kind == "choice":
      ok, p = _check_choice(pv, gold)
    elif kind == "normalize_pair":
      i = int(key[1]) - 1  # "S2_sample_uL" -> 1
      ok, p = _check_normalize_pair(pred.get(norm_key(key)), pred.get(norm_key(f"S{i + 1}_diluent_uL")), i, params)
      axes[f"S{i + 1}_diluent_uL"] = {"pass": ok, "pred": p[1], "gold": item.gold.get(f"S{i + 1}_diluent_uL")}
      p = p[0]
    else:
      raise ValueError(f"unknown kind {kind}")
    axes[key] = {"pass": bool(ok), "pred": p, "gold": gold}
    all_ok = all_ok and bool(ok)
  return Grade(item.id, all_ok, error_class=None if all_ok else "wrong_value", axes=axes, pred=pred, pred_feasible=pf)


# ----------------------------------------------------------------------------------------------------
# controls
# ----------------------------------------------------------------------------------------------------
def gold_completion(item: Item) -> str:
  return "Final answer:\n```json\n" + json.dumps(item.gold) + "\n```"


def mutations(item: Item) -> dict[str, str]:
  """Corrupted versions of the gold, one field at a time. Every one must be rejected by `grade`."""
  out: dict[str, str] = {}
  g = dict(item.gold)
  if not item.feasible:
    out["flag_flipped"] = json.dumps({**g, "feasible": True})
    return out
  out["flag_flipped"] = json.dumps({**g, "feasible": False})
  kinds = item.grader["kinds"]
  for key, kind in kinds.items():
    v = g[key]
    if kind in ("volume", "number"):
      # +3 % but never less than 2x the absolute volume tolerance, so the corruption is outside the band
      out[f"{key}+3%"] = json.dumps({**g, key: v * 1.03 + (2 * ABS_UL if kind == "volume" else 0)})
      out[f"{key}x1000"] = json.dumps({**g, key: v * 1000})
    elif kind == "count":
      out[f"{key}+1"] = json.dumps({**g, key: v + 1})
    elif kind == "conc_list":
      out[f"{key}[0]+3%"] = json.dumps({**g, key: [v[0] * 1.03] + list(v[1:])})
      out[f"{key}_short"] = json.dumps({**g, key: list(v[:-1])})
    elif kind == "volume_dict":
      k0 = next(iter(v))
      out[f"{key}.{k0}+3%"] = json.dumps({**g, key: {**v, k0: v[k0] * 1.03}})
    elif kind == "wells":
      out[f"{key}_drop_one"] = json.dumps({**g, key: list(v[1:])})
      out[f"{key}_shift"] = json.dumps({**g, key: [_shift_well(w) for w in v]})
    elif kind == "well":
      out[f"{key}_shift"] = json.dumps({**g, key: _shift_well(v)})
    elif kind == "choice":
      alt = [c for c in item.grader["params"]["choices"] if c != v][0]
      out[f"{key}_other"] = json.dumps({**g, key: alt})
    elif kind == "normalize_pair":
      out[f"{key}+5%"] = json.dumps({**g, key: v * 1.05})
      dkey = key.replace("_sample_uL", "_diluent_uL")
      out[f"{dkey}+5%"] = json.dumps({**g, dkey: g[dkey] * 1.05 + 0.5})
  return out


def _shift_well(w: str) -> str:
  r, c = L.parse_well(w)
  return L.well_name(r, c + 1) if c < 23 else L.well_name(r, c - 1)


def policy_always_feasible_zeros(item: Item) -> str:
  d = {}
  for key, kind in item.grader["kinds"].items():
    if kind == "bool":
      d[key] = True
    elif kind in ("conc_list", "wells"):
      d[key] = []
    elif kind == "volume_dict":
      d[key] = {}
    elif kind in ("well", "choice"):
      d[key] = "A1" if kind == "well" else "C2"
    else:
      d[key] = 0
  return json.dumps(d)


def policy_always_infeasible(item: Item) -> str:
  return json.dumps({**{k: None for k in item.grader["kinds"] if k != "feasible"}, "feasible": False})
