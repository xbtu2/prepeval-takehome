"""Six seeded item families: four quantitative (Q1-Q4) and two labware/addressing (L1-L2).

Design rules (the user's eval doctrine):
  * the prompt states the intent, the WORLD block carries every number, grading owns the answer key;
  * the lever is never named (no "account for dead volume", no "check feasibility");
  * the JSON schema printed in the prompt is the only affordance; `feasible` is a report channel;
  * exactly 20 % of items (every 5th seed) are infeasible traps so neither "always do it" nor "always refuse" scores well;
  * each family has 2-3 phrasings so in-family fine-tune gains are not string memorisation;
  * every item carries a programmatic reasoning trace (used for few-shot examples and SFT data).

Ground truth for labware facts and well addressing is PyLabRobot via `prepeval.labware`.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field

from . import labware as L

TRAP_EVERY = 5  # every 5th seed is an infeasible trap -> exactly 20 % of every split, stratified
FAMILIES = ["dilute_stock", "serial_dilution", "master_mix", "normalize_samples", "well_addressing", "labware_fit"]
TRAIN_FAMILIES = ["dilute_stock", "serial_dilution", "master_mix", "well_addressing"]
HELDOUT_FAMILIES = ["normalize_samples", "labware_fit"]

CAT = L.load_catalogue()


@dataclass
class Item:
  id: str
  family: str
  template: int
  seed: int
  split: str
  prompt: str  # intent + world + question + schema line (the arm prompts wrap this)
  world: dict
  schema: dict  # key -> short type hint shown to the model
  gold: dict
  feasible: bool
  grader: dict  # {"kinds": {key: kind}, "params": {...}}
  trace: str  # programmatic step-by-step solution ending before the JSON
  meta: dict = field(default_factory=dict)

  def to_json(self) -> dict:
    return asdict(self)

  @staticmethod
  def from_json(d: dict) -> "Item":
    return Item(**d)


# ----------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------
def q(x: float, nd: int = 6) -> float:
  """Round to `nd` significant figures for display/gold; strips float noise (0.30000000000000004 -> 0.3)."""
  if isinstance(x, bool) or x is None:
    return x
  x = float(x)
  if x == 0:
    return 0.0
  return float(f"{x:.{nd}g}")


def fmt(x: float) -> str:
  v = q(x)
  return str(int(v)) if float(v).is_integer() else str(v)


MOLAR = {"M": 1e6, "mM": 1e3, "µM": 1.0, "nM": 1e-3}  # value in µM
MASS = {"mg/mL": 1e3, "µg/mL": 1.0, "ng/µL": 1.0}  # value in µg/mL (1 ng/µL == 1 µg/mL)
FOLD = {"X": 1.0}
PERCENT = {"%": 1.0}

REAGENTS = [
  # (name, unit family, stock values, diluent)
  ("MgCl2", MOLAR, [("M", 1), ("mM", 25), ("mM", 50), ("mM", 100), ("mM", 500)], "nuclease-free water"),
  ("ATP", MOLAR, [("mM", 10), ("mM", 100), ("mM", 20)], "reaction buffer"),
  ("compound A (in DMSO)", MOLAR, [("mM", 10), ("mM", 1), ("mM", 50), ("µM", 500)], "assay medium"),
  ("NaCl", MOLAR, [("M", 5), ("M", 1), ("mM", 500)], "water"),
  ("BSA", MASS, [("mg/mL", 10), ("mg/mL", 50), ("mg/mL", 100), ("mg/mL", 2)], "PBS"),
  ("ampicillin", MASS, [("mg/mL", 100), ("mg/mL", 50), ("mg/mL", 10)], "LB medium"),
  ("DNA ladder", MASS, [("µg/mL", 500), ("ng/µL", 100), ("ng/µL", 50)], "TE buffer"),
  ("Tris buffer", FOLD, [("X", 10), ("X", 20), ("X", 50), ("X", 5)], "water"),
  ("PBS", FOLD, [("X", 10), ("X", 20)], "water"),
  ("Tween-20", PERCENT, [("%", 10), ("%", 20), ("%", 5)], "PBS"),
]

VOLUMES_UL = [50, 100, 150, 200, 250, 300, 400, 500, 600, 750, 800, 1000, 1200, 1500, 2000]
VOLUMES_ML = [1, 1.5, 2, 2.5, 4, 5, 10, 20, 25, 50]
RATIOS = [2, 2.5, 4, 5, 8, 10, 20, 25, 40, 50, 100, 200, 250, 400, 500, 1000]
PIPETTE_MIN = [0.5, 1.0, 2.0]


def _vol_text(v_ul: float) -> str:
  return f"{fmt(v_ul / 1000)} mL" if v_ul >= 1000 else f"{fmt(v_ul)} µL"


def _choose_target_unit(unit_fam: dict, c_base: float, rng: random.Random) -> tuple[str, float]:
  """Express a base-unit concentration in a unit that reads naturally (value >= 0.1), preferring smaller units."""
  options = []
  for u, scale in unit_fam.items():
    val = c_base / scale
    if 0.1 <= val <= 50000:
      options.append((u, val))
  if not options:
    u = min(unit_fam, key=lambda k: unit_fam[k])
    return u, c_base / unit_fam[u]
  return rng.choice(options)


def _schema_line(schema: dict) -> str:
  body = ", ".join(f'"{k}": {v}' for k, v in schema.items())
  return (
    "Answer with one JSON object with exactly these keys: {" + body + "}. "
    'Use "feasible": false if the request cannot be carried out as stated with the materials listed '
    "(the other values may then be null)."
  )


def _world_lines(lines: list[str]) -> str:
  return "World:\n" + "\n".join(f"- {ln}" for ln in lines)


def _assemble(intent: str, world_lines: list[str], question: str, schema: dict) -> str:
  return f"Task: {intent}\n{_world_lines(world_lines)}\nQuestion: {question}\n{_schema_line(schema)}"


# ----------------------------------------------------------------------------------------------------
# Q1 dilute_stock
# ----------------------------------------------------------------------------------------------------
Q1_INTENTS = [
  "Prepare {V} of {reagent} at {C2} from the stock listed below, using {diluent} as the diluent.",
  "A plate run needs {V} of working {reagent} solution at {C2}; make it from the stock using {diluent}.",
  "Make a {C2} working solution of {reagent} (total {V}) by diluting the stock with {diluent}.",
]


def gen_dilute_stock(rng: random.Random, template: int, split: str, seed: int, trap: bool) -> Item:
  name, fam, stocks, diluent = rng.choice(REAGENTS)
  s_unit, s_val = rng.choice(stocks)
  c1 = s_val * fam[s_unit]  # base units
  v_ul = float(rng.choice(VOLUMES_ML) * 1000 if rng.random() < 0.35 else rng.choice(VOLUMES_UL))
  pmin = rng.choice(PIPETTE_MIN)
  trap_kind = None
  if trap and rng.random() < 0.5:
    trap_kind = "target_above_stock"
    ratio = rng.choice([0.5, 0.25, 0.2, 0.1])  # target more concentrated than the stock
  elif trap:
    trap_kind = "below_pipette_min"
    ratio = rng.choice([r for r in RATIOS if v_ul / r < pmin] or [1000, 2000, 5000])
    if v_ul / ratio >= pmin:
      ratio = v_ul / (pmin / 2)  # force the stock volume to half the pipette minimum
  else:
    ratio = rng.choice([r for r in RATIOS if pmin <= v_ul / r <= v_ul * 0.5])
  c2 = c1 / ratio
  t_unit, t_val = _choose_target_unit(fam, c2, rng)
  stock_ul = v_ul / ratio
  dil_ul = v_ul - stock_ul
  feasible = (c2 <= c1) and (stock_ul >= pmin) and (stock_ul <= v_ul)

  intent = Q1_INTENTS[template % len(Q1_INTENTS)].format(
    V=_vol_text(v_ul), reagent=name, C2=f"{fmt(t_val)} {t_unit}", diluent=diluent
  )
  world = {
    "reagent": name, "stock_conc": f"{fmt(s_val)} {s_unit}", "target_conc": f"{fmt(t_val)} {t_unit}",
    "final_volume_uL": v_ul, "diluent": diluent, "pipette_min_uL": pmin,
  }
  lines = [
    f"stock: {name} at {fmt(s_val)} {s_unit}",
    f"diluent: {diluent}",
    f"final volume required: {_vol_text(v_ul)}",
    f"smallest volume the pipette can transfer: {fmt(pmin)} µL",
  ]
  question = "How much stock and how much diluent (in µL) go into the final solution?"
  schema = {"stock_uL": "<number>", "diluent_uL": "<number>", "feasible": "<true|false>"}
  gold = {"stock_uL": q(stock_ul), "diluent_uL": q(dil_ul), "feasible": feasible} if feasible else {
    "stock_uL": None, "diluent_uL": None, "feasible": False}

  # trace
  tr = []
  if s_unit != t_unit:
    tr.append(f"Put both concentrations in the same unit: stock {fmt(s_val)} {s_unit} = {fmt(c1 / fam[t_unit])} {t_unit}; target {fmt(t_val)} {t_unit}.")
  else:
    tr.append(f"Stock {fmt(s_val)} {s_unit}, target {fmt(t_val)} {t_unit} (same unit).")
  if v_ul >= 1000:
    tr.append(f"Final volume {_vol_text(v_ul)} = {fmt(v_ul)} µL.")
  if c2 > c1:
    tr.append("The target is more concentrated than the stock, so no dilution can reach it: infeasible.")
  else:
    tr.append(f"C1·V1 = C2·V2 → V1 = C2·V2/C1 = {fmt(c1 / fam[t_unit])} vs {fmt(t_val)}: dilution factor {fmt(ratio)}, so stock volume = {fmt(v_ul)}/{fmt(ratio)} = {fmt(stock_ul)} µL.")
    if stock_ul < pmin:
      tr.append(f"{fmt(stock_ul)} µL is below the pipette minimum of {fmt(pmin)} µL, so this cannot be pipetted as stated: infeasible.")
    else:
      tr.append(f"Diluent = {fmt(v_ul)} − {fmt(stock_ul)} = {fmt(dil_ul)} µL.")
  return Item(
    id=f"dilute_stock-{split}-{seed:05d}", family="dilute_stock", template=template, seed=seed, split=split,
    prompt=_assemble(intent, lines, question, schema), world=world, schema=schema, gold=gold, feasible=feasible,
    grader={"kinds": {"stock_uL": "volume", "diluent_uL": "volume", "feasible": "bool"}, "params": {}},
    trace=" ".join(tr), meta={"trap": trap_kind, "ratio": ratio},
  )


# ----------------------------------------------------------------------------------------------------
# Q2 serial_dilution
# ----------------------------------------------------------------------------------------------------
Q2_INTENTS = [
  "Set up a {n}-point {f}-fold serial dilution of {reagent} across columns 1 to {n} of the plate (one well per point, row A).",
  "Build a {f}-fold dilution series of {reagent} with {n} points along row A, wells A1 to A{n}.",
  "Lay out {n} concentrations of {reagent}, each {f}-fold lower than the previous, in row A columns 1-{n}.",
]
Q2_REAGENTS = [("compound B", "µM", [100, 200, 50, 1000, 400]), ("antibody", "µg/mL", [10, 20, 40, 5]),
               ("standard protein", "ng/mL", [1000, 2000, 500, 800]), ("inhibitor", "nM", [1000, 5000, 2000, 10000])]


def gen_serial_dilution(rng: random.Random, template: int, split: str, seed: int, trap: bool) -> Item:
  plate_key = rng.choice(["plate_96_flat", "plate_96_flat", "plate_96_deep", "plate_384"])
  plate = CAT[plate_key]
  name, unit, starts = rng.choice(Q2_REAGENTS)
  c0 = float(rng.choice(starts))
  f = rng.choice([2, 2, 3, 4, 5, 10])
  n_max = min(12, int(math.log(1e4) / math.log(f)) + 1)  # keep the last point >= 1e-4 x the top
  n = rng.randrange(4, n_max + 1)
  wmax = plate["well_max_uL"]
  # peak volume in wells 2..n is V·f/(f-1); V is the equal final volume per well
  if trap:
    candidates = [v for v in [40, 50, 60, 80, 100, 150, 200, 250, 300, 500, 1000, 1500, 2000] if v * f / (f - 1) > wmax]
    v = float(rng.choice(candidates or [wmax]))  # if nothing exceeds, fall through to feasible below
  else:
    candidates = [v for v in [20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 180, 200, 250, 300, 500, 800, 1000] if v * f / (f - 1) <= wmax]
    v = float(rng.choice(candidates))
  peak = v * f / (f - 1)
  feasible = peak <= wmax + 1e-9
  transfer = v / (f - 1)
  concs = [c0 / f**k for k in range(n)]

  intent = Q2_INTENTS[template % len(Q2_INTENTS)].format(n=n, f=f, reagent=name)
  lines = [
    f"plate: {plate['display']}",
    f"{name} top concentration (well A1): {fmt(c0)} {unit}",
    f"diluent: assay buffer",
    f"series scheme: well A1 holds the top concentration; wells A2-A{n} are pre-filled with diluent, each receives the same carry volume from the previous well and is mixed; the carry out of the last well is discarded",
    f"every well must end at exactly {fmt(v)} µL once the series is complete",
  ]
  world = {"plate": plate_key, "well_max_uL": wmax, "c0": c0, "unit": unit, "factor": f, "points": n, "final_uL": v}
  question = "What carry volume and per-well diluent volume (µL) build this series, what is discarded from the last well, and what concentration does each well end at (list from A1 to the last well)?"
  schema = {"carry_uL": "<number>", "diluent_per_well_uL": "<number>", "discard_uL": "<number>",
            f"concentrations_{unit.replace('/', '_per_')}": "<list of numbers>", "feasible": "<true|false>"}
  ckey = f"concentrations_{unit.replace('/', '_per_')}"
  gold = ({"carry_uL": q(transfer), "diluent_per_well_uL": q(v), "discard_uL": q(transfer), ckey: [q(c) for c in concs], "feasible": True}
          if feasible else {"carry_uL": None, "diluent_per_well_uL": None, "discard_uL": None, ckey: None, "feasible": False})
  tr = [
    f"Each well ends at {fmt(v)} µL. A {f}-fold step needs carry : diluent = 1 : {f - 1}, so diluent per well = {fmt(v)} µL and carry = {fmt(v)}/{f - 1} = {fmt(transfer)} µL.",
    f"Before the carry-out, wells A2-A{n} hold {fmt(v)} + {fmt(transfer)} = {fmt(peak)} µL.",
  ]
  if not feasible:
    tr.append(f"{fmt(peak)} µL exceeds the well maximum of {fmt(wmax)} µL, so the series cannot be built in this plate: infeasible.")
  else:
    tr.append(f"{fmt(peak)} µL fits under the well maximum of {fmt(wmax)} µL. The last well's carry-out ({fmt(transfer)} µL) is discarded.")
    tr.append("Concentrations: " + ", ".join(f"A{k + 1} = {fmt(c)}" for k, c in enumerate(concs)) + f" {unit}.")
  return Item(
    id=f"serial_dilution-{split}-{seed:05d}", family="serial_dilution", template=template, seed=seed, split=split,
    prompt=_assemble(intent, lines, question, schema), world=world, schema=schema, gold=gold, feasible=feasible,
    grader={"kinds": {"carry_uL": "volume", "diluent_per_well_uL": "volume", "discard_uL": "volume", ckey: "conc_list", "feasible": "bool"}, "params": {}},
    trace=" ".join(tr), meta={"trap": "well_overflow" if trap and not feasible else None, "plr_id": plate["plr_id"]},
  )


# ----------------------------------------------------------------------------------------------------
# Q3 master_mix
# ----------------------------------------------------------------------------------------------------
Q3_INTENTS = [
  "Assemble a master mix for {N} {kind} reactions of {R} µL each; template is added to each well separately afterwards.",
  "Prepare enough {kind} master mix for {N} reactions ({R} µL per reaction, template added per well later).",
  "We need {N} × {R} µL {kind} reactions; make one master mix (without template) that covers them.",
]
Q3_KITS = {
  "PCR": [("10X buffer", 0.10), ("dNTP mix", 0.02), ("forward primer", 0.04), ("reverse primer", 0.04), ("polymerase", 0.01)],
  "qPCR": [("2X qPCR mix", 0.50), ("primer pair", 0.05), ("ROX dye", 0.02)],
  "restriction digest": [("10X digest buffer", 0.10), ("enzyme", 0.02)],
  "ligation": [("2X ligase buffer", 0.50), ("T4 ligase", 0.05)],
}


def gen_master_mix(rng: random.Random, template: int, split: str, seed: int, trap: bool) -> Item:
  kind = rng.choice(list(Q3_KITS))
  R = rng.choice([10, 20, 25, 50])
  N = rng.randrange(8, 97)
  excess = rng.choice([5, 10, 10, 15, 20])
  comps = [(nm, q(frac * R, 4)) for nm, frac in Q3_KITS[kind]]
  tmpl = q(R * rng.choice([0.04, 0.08, 0.10]), 4)
  if trap:  # inflate one component so the recipe over-fills the reaction (guaranteed: others + template + extra > R)
    i = rng.randrange(len(comps))
    others = sum(v for j, (_, v) in enumerate(comps) if j != i)
    comps[i] = (comps[i][0], q(R - others - tmpl + R * rng.choice([0.05, 0.1, 0.2, 0.4]), 4))
  total_comp = sum(v for _, v in comps)
  water_per = R - total_comp - tmpl
  feasible = water_per >= -1e-9
  n_prep = N * (1 + excess / 100)
  totals = {nm: q(v * n_prep) for nm, v in comps}
  water_total = q(water_per * n_prep)

  intent = Q3_INTENTS[template % len(Q3_INTENTS)].format(N=N, R=R, kind=kind)
  lines = [f"per-reaction recipe ({R} µL total): " + ", ".join(f"{nm} {fmt(v)} µL" for nm, v in comps) + f", template {fmt(tmpl)} µL, water to volume",
           f"lab policy: prepare {excess}% more mix than the reactions strictly need",
           "available: all listed reagents and nuclease-free water"]
  world = {"kind": kind, "R": R, "N": N, "excess_pct": excess, "components": comps, "template_uL": tmpl}
  question = "How many reactions' worth of mix is prepared, how much of each component (µL) goes into the mix, and how much water (µL)?"
  schema = {"reactions_prepared": "<number>", "component_totals_uL": "{<component name>: <number>, ...}", "water_uL": "<number>", "feasible": "<true|false>"}
  gold = ({"reactions_prepared": q(n_prep), "component_totals_uL": totals, "water_uL": water_total, "feasible": True}
          if feasible else {"reactions_prepared": None, "component_totals_uL": None, "water_uL": None, "feasible": False})
  tr = [f"Components per reaction sum to {fmt(total_comp)} µL; with {fmt(tmpl)} µL template that leaves {fmt(R)} − {fmt(total_comp)} − {fmt(tmpl)} = {fmt(water_per)} µL water per reaction."]
  if not feasible:
    tr.append("That is negative: the recipe does not fit in the reaction volume, so the mix cannot be made as stated: infeasible.")
  else:
    tr.append(f"Reactions prepared = {N} × (1 + {excess}/100) = {fmt(n_prep)}.")
    tr.append("Totals: " + ", ".join(f"{nm} {fmt(v)} × {fmt(n_prep)} = {fmt(totals[nm])} µL" for nm, v in comps) + f"; water {fmt(water_per)} × {fmt(n_prep)} = {fmt(water_total)} µL.")
  return Item(
    id=f"master_mix-{split}-{seed:05d}", family="master_mix", template=template, seed=seed, split=split,
    prompt=_assemble(intent, lines, question, schema), world=world, schema=schema, gold=gold, feasible=feasible,
    grader={"kinds": {"reactions_prepared": "number", "component_totals_uL": "volume_dict", "water_uL": "volume", "feasible": "bool"}, "params": {}},
    trace=" ".join(tr), meta={"trap": "recipe_overfills" if trap and not feasible else None},
  )


# ----------------------------------------------------------------------------------------------------
# Q4 normalize_samples (held-out)
# ----------------------------------------------------------------------------------------------------
Q4_INTENTS = [
  "Normalise the three DNA samples below so that every one ends at {Ct} ng/µL in at least {Vmin} µL, using {diluent}.",
  "Bring samples S1-S3 to a common concentration of {Ct} ng/µL (final volume at least {Vmin} µL each) with {diluent}.",
  "Dilute each of the three samples to {Ct} ng/µL; each normalised sample must be at least {Vmin} µL. Diluent: {diluent}.",
]


def gen_normalize_samples(rng: random.Random, template: int, split: str, seed: int, trap: bool) -> Item:
  Ct = rng.choice([10, 20, 25, 50, 100])
  Vmin = rng.choice([20, 25, 30, 40, 50, 100])
  pmin = rng.choice(PIPETTE_MIN)
  diluent = rng.choice(["elution buffer", "nuclease-free water", "TE buffer"])
  concs = []
  for _ in range(3):
    c = q(Ct * rng.choice([1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12.5, 16, 20]) * rng.uniform(0.95, 1.05), 4)
    concs.append(c)
  if trap:
    i = rng.randrange(3)
    concs[i] = q(Ct * rng.uniform(0.3, 0.9), 3)
  feasible = all(c > Ct for c in concs)
  # available volume per sample: a round number comfortably above what the reference construction needs
  avail = []
  for c in concs:
    need = Ct * Vmin / c if c > Ct else 0.0
    need = max(need, pmin)
    avail.append(float(next((a for a in [10, 15, 20, 25, 30, 40, 50, 60, 80, 100] if a >= need * 1.2 + 1), math.ceil(need) + 10)))
  plan = {}
  tr = []
  for i, (c, a) in enumerate(zip(concs, avail), start=1):
    if c <= Ct:
      tr.append(f"S{i} is {fmt(c)} ng/µL, already below the {fmt(Ct)} ng/µL target; it cannot be made more concentrated by dilution: infeasible.")
      continue
    s = Ct * Vmin / c
    total = Vmin
    note = ""
    if s < pmin:
      total = pmin * c / Ct
      s = pmin
      note = f" (scaled up so the sample volume reaches the {fmt(pmin)} µL pipette minimum: total {fmt(total)} µL)"
    d = total - s
    plan[f"S{i}_sample_uL"] = q(s)
    plan[f"S{i}_diluent_uL"] = q(d)
    tr.append(f"S{i}: sample = {fmt(Ct)}×{fmt(Vmin)}/{fmt(c)} = {fmt(Ct * Vmin / c)} µL{note}; diluent = {fmt(total)} − {fmt(s)} = {fmt(d)} µL.")
  intent = Q4_INTENTS[template % len(Q4_INTENTS)].format(Ct=fmt(Ct), Vmin=fmt(Vmin), diluent=diluent)
  lines = [f"S{i}: measured {fmt(c)} ng/µL, {fmt(a)} µL available" for i, (c, a) in enumerate(zip(concs, avail), start=1)]
  lines += [f"diluent: {diluent}", f"smallest volume the pipette can transfer: {fmt(pmin)} µL"]
  world = {"Ct": Ct, "Vmin": Vmin, "pmin": pmin, "concs": concs, "avail": avail}
  question = "For each sample, how many µL of sample and how many µL of diluent are combined?"
  schema = {}
  for i in range(1, 4):
    schema[f"S{i}_sample_uL"] = "<number>"
    schema[f"S{i}_diluent_uL"] = "<number>"
  schema["feasible"] = "<true|false>"
  gold = {**plan, "feasible": True} if feasible else {**{k: None for k in schema if k != "feasible"}, "feasible": False}
  kinds = {k: "normalize_pair" for k in schema if k.endswith("_sample_uL")}
  kinds.update({k: "paired" for k in schema if k.endswith("_diluent_uL")})
  kinds["feasible"] = "bool"
  return Item(
    id=f"normalize_samples-{split}-{seed:05d}", family="normalize_samples", template=template, seed=seed, split=split,
    prompt=_assemble(intent, lines, question, schema), world=world, schema=schema, gold=gold, feasible=feasible,
    grader={"kinds": kinds, "params": {"Ct": Ct, "Vmin": Vmin, "pmin": pmin, "concs": concs, "avail": avail}},
    trace=" ".join(tr), meta={"trap": "sample_below_target" if trap and not feasible else None},
  )


# ----------------------------------------------------------------------------------------------------
# L1 well_addressing
# ----------------------------------------------------------------------------------------------------
INDEX_CONVENTION = "wells are numbered column by column, top to bottom, starting at 0 (A1 = 0, B1 = 1, …, then A2 continues after the last row of column 1)"


def gen_well_addressing(rng: random.Random, template: int, split: str, seed: int, trap: bool) -> Item:
  sub = template % 4
  if sub == 2 or sub == 3:
    plate_key = "plate_96_flat" if sub == 2 else rng.choice(["plate_96_flat", "plate_96_deep"])
  else:
    plate_key = rng.choice(["plate_96_flat", "plate_96_deep", "plate_384"])
  plate = CAT[plate_key]
  rows, cols = plate["rows"], plate["cols"]
  lines = [f"plate: {plate['display']}", f"well names: row letter (A = top row) + column number (1 = leftmost column)"]
  meta = {"plr_id": plate["plr_id"], "sub": sub}
  feasible = True
  if sub == 0:  # range expansion
    r0, r1 = sorted(rng.sample(range(rows), 2)) if rng.random() < 0.8 else (rng.randrange(rows),) * 2
    c0, c1 = sorted(rng.sample(range(cols), 2)) if rng.random() < 0.8 else (rng.randrange(cols),) * 2
    if (r1 - r0 + 1) * (c1 - c0 + 1) > 24:  # keep lists short enough to grade and generate
      r1 = min(r0 + 2, rows - 1)
      c1 = min(c0 + 5, cols - 1)
    a, b = L.well_name(r0, c0), L.well_name(r1, c1)
    if trap:
      b = L.well_name(r1, cols) if rng.random() < 0.5 else L.well_name(rows, c1)  # one corner off the plate
      feasible = False
    intent = f"A method step says: fill the block of wells {a}:{b} with buffer. The block is the rectangle spanned by those two corner wells."
    question = "List every well in that block and how many wells it contains."
    schema = {"wells": "<list of well names>", "count": "<integer>", "feasible": "<true|false>"}
    if feasible:
      wells = L.expand_range(f"{a}:{b}", rows, cols)
      gold = {"wells": wells, "count": len(wells), "feasible": True}
      tr = f"Corners {a} and {b}: rows {L.row_label(r0)}-{L.row_label(r1)} ({r1 - r0 + 1} rows) and columns {c0 + 1}-{c1 + 1} ({c1 - c0 + 1} columns). Listing row by row gives {', '.join(wells)}: {len(wells)} wells."
    else:
      gold = {"wells": None, "count": None, "feasible": False}
      tr = f"Corner {b} names a row or column that does not exist on a {rows}-row x {cols}-column plate, so the block cannot be addressed: infeasible."
    kinds = {"wells": "wells", "count": "count", "feasible": "bool"}
    meta["trap"] = "off_plate_corner" if not feasible else None
  elif sub == 1:  # index -> well
    k = rng.randrange(rows * cols)
    if trap:
      k = rows * cols + rng.randrange(1, 30)
      feasible = False
    lines.append(f"well numbering convention for this step: {INDEX_CONVENTION}")
    intent = f"A worklist row addresses the destination as well number {k} on the plate."
    question = "Which well is that?"
    schema = {"well": "<well name>", "feasible": "<true|false>"}
    if feasible:
      w = L.well_at_index(k, rows, cols)
      gold = {"well": w, "feasible": True}
      tr = f"Each column holds {rows} wells. {k} = {k // rows} × {rows} + {k % rows}, so column {k // rows + 1} (1-based) and row index {k % rows} = row {L.row_label(k % rows)}: well {w}."
    else:
      gold = {"well": None, "feasible": False}
      tr = f"The plate has {rows * cols} wells numbered 0-{rows * cols - 1}; {k} is outside that range: infeasible."
    kinds = {"well": "well", "feasible": "bool"}
    meta["trap"] = "index_off_plate" if not feasible else None
  elif sub == 2:  # 8-channel column
    c = rng.randrange(1, cols + 1)
    if trap:
      c = cols + rng.randrange(1, 4)
      feasible = False
    intent = f"An 8-channel pipetting head is positioned on column {c} of the plate and dispenses into all eight wells of that column at once."
    question = "Which wells receive liquid, and how many?"
    schema = {"wells": "<list of well names>", "count": "<integer>", "feasible": "<true|false>"}
    if feasible:
      wells = L.column_wells(c, rows, cols)
      gold = {"wells": wells, "count": len(wells), "feasible": True}
      tr = f"Column {c} has one well per row, rows A-{L.row_label(rows - 1)}: {', '.join(wells)} — {len(wells)} wells."
    else:
      gold = {"wells": None, "count": None, "feasible": False}
      tr = f"The plate has only {cols} columns; column {c} does not exist: infeasible."
    kinds = {"wells": "wells", "count": "count", "feasible": "bool"}
    meta["trap"] = "column_off_plate" if not feasible else None
  else:  # sub == 3: 96 -> 384 interleaved quadrant
    quad = rng.choice(list(L.QUADRANT_OFFSETS))
    src = L.well_name(rng.randrange(8), rng.randrange(12))
    if trap:
      src = L.well_name(8 + rng.randrange(2), rng.randrange(12)) if rng.random() < 0.5 else L.well_name(rng.randrange(8), 12 + rng.randrange(3))
      feasible = False
    p384 = CAT["plate_384"]
    lines.append(f"destination plate: {p384['display']}")
    lines.append("stamping convention: the 96-well plate is copied into one interleaved quadrant of the 384-well plate — source well in row r, column c (both counted from 0) lands in destination row 2r + dr, column 2c + dc, where (dr, dc) is (0,0) for the top-left quadrant, (0,1) top-right, (1,0) bottom-left, (1,1) bottom-right")
    intent = f"The 96-well source plate is stamped into the {L.QUADRANT_WORDS[quad]} interleaved quadrant of the 384-well destination plate."
    question = f"Which destination well receives the contents of source well {src}?"
    schema = {"well": "<well name>", "feasible": "<true|false>"}
    if feasible:
      w = L.quadrant_well_384(src, quad)
      r, c = L.parse_well(src)
      dr, dc = L.QUADRANT_OFFSETS[quad]
      gold = {"well": w, "feasible": True}
      tr = f"Source {src} is row {r}, column {c} (0-based). Destination row = 2×{r} + {dr} = {2 * r + dr} → row {L.row_label(2 * r + dr)}; column = 2×{c} + {dc} = {2 * c + dc} → column {2 * c + dc + 1}: well {w}."
    else:
      gold = {"well": None, "feasible": False}
      tr = f"{src} is not a well of an 8-row x 12-column plate, so there is nothing to stamp: infeasible."
    kinds = {"well": "well", "feasible": "bool"}
    meta["trap"] = "source_off_plate" if not feasible else None
  return Item(
    id=f"well_addressing-{split}-{seed:05d}", family="well_addressing", template=template, seed=seed, split=split,
    prompt=_assemble(intent, lines, question, schema), world={"plate": plate_key, "rows": rows, "cols": cols, "sub": sub},
    schema=schema, gold=gold, feasible=feasible, grader={"kinds": kinds, "params": {"rows": rows, "cols": cols}},
    trace=tr, meta=meta,
  )


# ----------------------------------------------------------------------------------------------------
# L2 labware_fit (held-out)
# ----------------------------------------------------------------------------------------------------
CONTAINER_LABELS = {"plate_384": "C1", "plate_96_flat": "C2", "plate_96_deep": "C3", "trough_60mL": "C4"}
TROUGH_DEAD_UL = [2000, 3000, 5000]


def gen_labware_fit(rng: random.Random, template: int, split: str, seed: int, trap: bool) -> Item:
  sub = template % 3
  dead = rng.choice(TROUGH_DEAD_UL)
  lines = ["containers available:"] + [f"  {CONTAINER_LABELS[k]}: {CAT[k]['display']}" for k in CONTAINER_LABELS]
  lines.append(f"lab policy: a trough must be loaded with {fmt(dead)} µL more than will be aspirated from it (liquid the channels cannot reach)")
  lines.append("plates of each format are in unlimited supply; one container type is used per task")
  meta = {"sub": sub}
  feasible = True
  trough = CAT["trough_60mL"]
  if sub == 0:  # shared reagent to every well of N 96-well plates with an 8-channel head
    v = rng.choice([20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 200])
    n_max = int((trough["well_max_uL"] - dead) // (96 * v))  # largest plate count that still fits the trough
    N = rng.randrange(1, max(2, min(6, n_max) + 1))
    if trap:
      N = rng.randrange(6, 12)
      v = rng.choice([150, 200, 250, 300])
    need = v * 96 * N
    load = need + dead
    feasible = load <= trough["well_max_uL"]
    intent = f"Dispense {fmt(v)} µL of the same wash buffer into every well of {N} 96-well assay plate{'s' if N > 1 else ''} using an 8-channel head that aspirates from one source container."
    question = "Which container holds the wash buffer source, and how many µL must be loaded into it?"
    schema = {"source_container": "<C1|C2|C3|C4>", "load_uL": "<number>", "feasible": "<true|false>"}
    if feasible:
      gold = {"source_container": "C4", "load_uL": q(load), "feasible": True}
      tr = f"One liquid shared by all channels → the trough (C4). Aspirated volume = {fmt(v)} × 96 × {N} = {fmt(need)} µL; plus the {fmt(dead)} µL that cannot be reached → load {fmt(load)} µL, within the trough's {fmt(trough['well_max_uL'])} µL."
    else:
      gold = {"source_container": None, "load_uL": None, "feasible": False}
      tr = f"Aspirated volume = {fmt(v)} × 96 × {N} = {fmt(need)} µL, plus {fmt(dead)} µL unreachable = {fmt(load)} µL, which exceeds the trough's {fmt(trough['well_max_uL'])} µL: infeasible."
    kinds = {"source_container": "choice", "load_uL": "volume", "feasible": "bool"}
    meta["trap"] = "trough_overfill" if not feasible else None
  elif sub == 1:  # n distinct samples at v µL each -> smallest plate format that fits
    n = rng.randrange(10, 500)
    v = rng.choice([20, 40, 50, 60, 100, 150, 200, 300, 350, 500, 800, 1000, 1500, 2000])
    if trap:
      v = rng.choice([2500, 3000, 4000, 5000])
    lines.append("rule for this step: use the plate format with the fewest wells whose wells can hold the per-sample volume")
    fits = [k for k in ["plate_384", "plate_96_flat", "plate_96_deep"] if v <= CAT[k]["well_max_uL"]]
    feasible = bool(fits)
    intent = f"Store {n} distinct samples, {fmt(v)} µL each, so that no two samples share a well."
    question = "Which container type is used and how many of them are needed?"
    schema = {"container": "<C1|C2|C3|C4>", "plates_needed": "<integer>", "feasible": "<true|false>"}
    if feasible:
      k = fits[0]
      wells = CAT[k]["rows"] * CAT[k]["cols"]
      plates = math.ceil(n / wells)
      gold = {"container": CONTAINER_LABELS[k], "plates_needed": plates, "feasible": True}
      tr = f"Samples must stay separate → a plate, not the trough. Per-well capacity: C1 {fmt(CAT['plate_384']['well_max_uL'])}, C2 {fmt(CAT['plate_96_flat']['well_max_uL'])}, C3 {fmt(CAT['plate_96_deep']['well_max_uL'])} µL; the fewest-well format holding {fmt(v)} µL is {CONTAINER_LABELS[k]} ({wells} wells). Plates = ceil({n}/{wells}) = {plates}."
    else:
      gold = {"container": None, "plates_needed": None, "feasible": False}
      tr = f"{fmt(v)} µL exceeds every well capacity (largest is {fmt(CAT['plate_96_deep']['well_max_uL'])} µL), so no plate can hold the samples: infeasible."
    kinds = {"container": "choice", "plates_needed": "count", "feasible": "bool"}
    meta["trap"] = "volume_exceeds_all_wells" if not feasible else None
  else:  # sub == 2: shared reagent to n wells (partial plate), 8-channel -> trough load
    n = rng.randrange(8, 97)
    v = rng.choice([50, 100, 150, 200, 250, 300, 400, 500])
    if trap:
      n = rng.randrange(80, 97)
      v = rng.choice([800, 1000, 1200])  # >= 80 x 800 + dead > 60 mL
    need = v * n
    load = need + dead
    feasible = load <= trough["well_max_uL"]
    intent = f"Add {fmt(v)} µL of one enzyme solution to {n} wells of a deep-well plate, aspirating from a single source container with the multichannel head."
    question = "Which container is the enzyme source, and how many µL are loaded into it?"
    schema = {"source_container": "<C1|C2|C3|C4>", "load_uL": "<number>", "feasible": "<true|false>"}
    if feasible:
      gold = {"source_container": "C4", "load_uL": q(load), "feasible": True}
      tr = f"A single shared liquid for the head → trough (C4). Aspirated = {fmt(v)} × {n} = {fmt(need)} µL; plus {fmt(dead)} µL unreachable → load {fmt(load)} µL (≤ {fmt(trough['well_max_uL'])} µL)."
    else:
      gold = {"source_container": None, "load_uL": None, "feasible": False}
      tr = f"Aspirated = {fmt(v)} × {n} = {fmt(need)} µL; plus {fmt(dead)} µL = {fmt(load)} µL, more than the trough holds ({fmt(trough['well_max_uL'])} µL): infeasible."
    kinds = {"source_container": "choice", "load_uL": "volume", "feasible": "bool"}
    meta["trap"] = "trough_overfill" if not feasible else None
  return Item(
    id=f"labware_fit-{split}-{seed:05d}", family="labware_fit", template=template, seed=seed, split=split,
    prompt=_assemble(intent, lines, question, schema), world={"sub": sub, "dead_uL": dead, "catalogue": {CONTAINER_LABELS[k]: CAT[k]["plr_id"] for k in CONTAINER_LABELS}},
    schema=schema, gold=gold, feasible=feasible, grader={"kinds": kinds, "params": {"choices": list(CONTAINER_LABELS.values())}},
    trace=tr, meta=meta,
  )


GENERATORS = {
  "dilute_stock": gen_dilute_stock,
  "serial_dilution": gen_serial_dilution,
  "master_mix": gen_master_mix,
  "normalize_samples": gen_normalize_samples,
  "well_addressing": gen_well_addressing,
  "labware_fit": gen_labware_fit,
}
N_TEMPLATES = {"dilute_stock": 3, "serial_dilution": 3, "master_mix": 3, "normalize_samples": 3, "well_addressing": 4, "labware_fit": 3}


# gold values that legitimately coincide with a number in the prompt by construction (the model must
# recognise them, not derive them): equal final volume of a series, its top concentration, the count of an
# 8-channel head, and a 2-fold carry (= the final volume). Everything else is resampled if it collides.
STRUCTURAL_EXEMPT = {"diluent_per_well_uL", "count"}


def _leaky_values(item: "Item") -> list[tuple[str, float]]:
  """Non-trivial gold values (non-integers or integers > 12) that appear verbatim in the prompt."""
  import re as _re

  hits = []
  exempt = set(STRUCTURAL_EXEMPT) | set(item.meta.get("leak_exempt", []))
  if not item.feasible:
    return hits
  for k, v in item.gold.items():
    if k in exempt or k == "feasible" or v is None:
      continue
    vals: list = []
    if isinstance(v, (int, float)) and not isinstance(v, bool):
      vals = [v]
    elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
      vals = v[1:]  # the top concentration is stated in the world block
    elif isinstance(v, dict):
      vals = list(v.values())
    for x in vals:
      if float(x).is_integer() and abs(x) <= 12:
        continue  # small integers (plate columns, fold factors) are not answers anyone could copy
      if _re.search(rf"(?<![\d.]){_re.escape(fmt(x))}(?![\d.])", item.prompt):
        hits.append((k, x))
  return hits


def make_item(family: str, seed: int, split: str, max_attempts: int = 25) -> Item:
  """Deterministic: (family, seed, split) fully determines the item. Items whose gold coincides with a number
  printed in their own prompt are resampled (deterministically) so the leak check holds by construction."""
  template = seed % N_TEMPLATES[family]
  trap = seed % TRAP_EVERY == 0
  for attempt in range(max_attempts):
    rng = random.Random(f"{family}:{split}:{seed}:{attempt}")
    item = GENERATORS[family](rng, template, split, seed, trap)
    if family == "serial_dilution" and item.world["factor"] == 2:
      item.meta["leak_exempt"] = ["carry_uL", "discard_uL"]
    if not _leaky_values(item):
      item.meta["attempt"] = attempt
      return item
  raise RuntimeError(f"could not draw a leak-free item for {family} seed {seed}")


def completion_text(item: Item) -> str:
  """Reasoning trace followed by the gold JSON — the target text for few-shot examples and SFT."""
  return f"{item.trace}\n\nFinal answer:\n```json\n{json.dumps(item.gold)}\n```"
