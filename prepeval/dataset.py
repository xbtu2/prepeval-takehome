"""Deterministic dataset build: test / train / few-shot splits, leak checks, manifest.

  test     6 families x 40 seeds (0..39)                      -> data/test.jsonl   (240 items)
  train    4 train families x 375 seeds (1000..1374)          -> data/train.jsonl  (1500 items; held-out families absent)
  fewshot  2 feasible train items (one quantitative, one addressing) -> data/fewshot.jsonl

Run `python -m prepeval.dataset` to (re)build and print the census + leak-check verdicts.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from . import labware as L
from .families import FAMILIES, HELDOUT_FAMILIES, TRAIN_FAMILIES, Item, _leaky_values, fmt, make_item

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GENERATOR_VERSION = "0.1.0"
TEST_SEEDS = range(0, 40)
TRAIN_SEEDS = range(1000, 1375)

def _gold_numbers(it: Item) -> set[float]:
  out: set[float] = set()
  for v in it.gold.values():
    if isinstance(v, (int, float)) and not isinstance(v, bool):
      out.add(float(v))
    elif isinstance(v, list):
      out.update(float(x) for x in v if isinstance(x, (int, float)) and not isinstance(x, bool))
    elif isinstance(v, dict):
      out.update(float(x) for x in v.values() if isinstance(x, (int, float)) and not isinstance(x, bool))
  return {x for x in out if not (x.is_integer() and abs(x) <= 12)}


def build_items() -> tuple[list[Item], list[Item], list[Item]]:
  test = [make_item(f, s, "test") for f in FAMILIES for s in TEST_SEEDS]
  test_prompts = {it.prompt for it in test}
  # low-entropy families (an 8-channel head on column 5 of a 96-well plate) can reproduce a test prompt
  # exactly; such train items are dropped so no test prompt is ever seen in training.
  train = [it for f in TRAIN_FAMILIES for s in TRAIN_SEEDS if (it := make_item(f, s, "train")).prompt not in test_prompts]
  # few-shot examples spend facts: pick train items that share no non-trivial gold value with any test gold
  test_vals = {}
  for it in test:
    test_vals.setdefault(it.family, set()).update(_gold_numbers(it))

  def pick(family, pred):
    for it in train:
      if it.family == family and it.feasible and pred(it) and not (_gold_numbers(it) & test_vals[family]):
        return it
    raise RuntimeError(f"no collision-free few-shot example for {family}")

  fewshot = [pick("dilute_stock", lambda i: True), pick("well_addressing", lambda i: i.world["sub"] == 0)]
  return test, train, fewshot


# ----------------------------------------------------------------------------------------------------
# leak checks
# ----------------------------------------------------------------------------------------------------
def leak_check(test: list[Item], train: list[Item], fewshot: list[Item]) -> dict:
  report: dict = {}
  train_prompts = {it.prompt for it in train}
  report["test_prompt_in_train"] = sum(it.prompt in train_prompts for it in test)
  report["heldout_in_train"] = sum(it.family in HELDOUT_FAMILIES for it in train)
  # gold values verbatim in their own prompt (non-trivial values only; structural coincidences exempt)
  hits = [(it.id, k, x) for it in test for k, x in _leaky_values(it)]
  report["gold_value_in_prompt"] = hits
  # few-shot values must not coincide with any non-trivial test gold of the same family
  fs_hits = []
  for ex in fewshot:
    ex_vals = _gold_numbers(ex)
    for it in test:
      if it.family == ex.family and it.feasible and (_gold_numbers(it) & ex_vals):
        fs_hits.append((ex.id, it.id))
  report["fewshot_value_in_test"] = fs_hits
  report["PASS"] = (report["test_prompt_in_train"] == 0 and report["heldout_in_train"] == 0
                    and not hits and not fs_hits)
  return report


# ----------------------------------------------------------------------------------------------------
# io
# ----------------------------------------------------------------------------------------------------
def write_jsonl(path: Path, items: list[Item]) -> str:
  path.parent.mkdir(parents=True, exist_ok=True)
  h = hashlib.sha256()
  with path.open("w") as fh:
    for it in items:
      data = (json.dumps(it.to_json(), ensure_ascii=False) + "\n").encode("utf-8")
      fh.write(data.decode("utf-8"))
      h.update(data)
  return h.hexdigest()


def read_jsonl(path: Path) -> list[Item]:
  with path.open() as fh:
    return [Item.from_json(json.loads(line)) for line in fh if line.strip()]


def load_split(name: str) -> list[Item]:
  return read_jsonl(DATA_DIR / f"{name}.jsonl")


def census(items: list[Item]) -> dict:
  fam = Counter(it.family for it in items)
  traps = Counter(it.family for it in items if not it.feasible)
  return {f: {"n": fam[f], "infeasible": traps[f], "trap_share": round(traps[f] / fam[f], 3)} for f in fam}


def build(write: bool = True) -> dict:
  test, train, fewshot = build_items()
  leaks = leak_check(test, train, fewshot)
  manifest = {
    "generator_version": GENERATOR_VERSION,
    "plr_commit": L.PLR_COMMIT,
    "families": FAMILIES,
    "train_families": TRAIN_FAMILIES,
    "heldout_families": HELDOUT_FAMILIES,
    "test_seeds": [TEST_SEEDS.start, TEST_SEEDS.stop],
    "train_seeds": [TRAIN_SEEDS.start, TRAIN_SEEDS.stop],
    "fewshot_ids": [it.id for it in fewshot],
    "census": {"test": census(test), "train": census(train)},
    "leak_check": {k: (v if not isinstance(v, list) else len(v)) for k, v in leaks.items()},
  }
  if write:
    manifest["sha256"] = {
      "test.jsonl": write_jsonl(DATA_DIR / "test.jsonl", test),
      "train.jsonl": write_jsonl(DATA_DIR / "train.jsonl", train),
      "fewshot.jsonl": write_jsonl(DATA_DIR / "fewshot.jsonl", fewshot),
    }
    with (DATA_DIR / "manifest.json").open("w") as fh:
      json.dump(manifest, fh, indent=2)
      fh.write("\n")
  manifest["_leak_detail"] = leaks
  return manifest


if __name__ == "__main__":
  m = build(write="--no-write" not in sys.argv)
  print(json.dumps({k: v for k, v in m.items() if k != "_leak_detail"}, indent=2))
  d = m["_leak_detail"]
  print("leak check:", "PASS" if d["PASS"] else "FAIL", {k: v for k, v in d.items() if k != "PASS"})
