"""Accuracy with bootstrap CIs, exact McNemar between paired arms, per-family tables, results JSON."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass

import numpy as np
from scipy.stats import binomtest

from .families import FAMILIES, HELDOUT_FAMILIES, Item
from .grading import Grade


def bootstrap_ci(correct, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
  x = np.asarray(correct, dtype=float)
  if len(x) == 0:
    return (float("nan"), float("nan"))
  rng = np.random.default_rng(seed)
  idx = rng.integers(0, len(x), size=(n_boot, len(x)))
  means = x[idx].mean(axis=1)
  return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap_diff(a, b, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
  d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
  rng = np.random.default_rng(seed)
  idx = rng.integers(0, len(d), size=(n_boot, len(d)))
  means = d[idx].mean(axis=1)
  return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mcnemar_exact(a, b) -> dict:
  """Exact (binomial) McNemar test on discordant pairs; a and b are aligned 0/1 vectors."""
  a = np.asarray(a, dtype=bool)
  b = np.asarray(b, dtype=bool)
  b_only = int((~a & b).sum())
  a_only = int((a & ~b).sum())
  n = a_only + b_only
  p = float(binomtest(min(a_only, b_only), n, 0.5).pvalue) if n > 0 else 1.0
  return {"b_only": b_only, "a_only": a_only, "discordant": n, "p_value": p}


def _acc(flags) -> float:
  return float(np.mean(flags)) if len(flags) else float("nan")


def summarize(items: list[Item], grades_by_arm: dict[str, dict[str, Grade]], baseline: str = "baseline") -> dict:
  """grades_by_arm: arm -> {item_id -> Grade}. Items missing a grade count as not measured (reported, not 0)."""
  by_id = {it.id: it for it in items}
  out: dict = {"n_items": len(items), "arms": {}, "comparisons": {}}
  vectors: dict[str, list[int]] = {}
  for arm, grades in grades_by_arm.items():
    measured = [it for it in items if it.id in grades and grades[it.id].measured]
    acc_flags = [int(grades[it.id].accepted) for it in measured]
    vectors[arm] = [int(it.id in grades and grades[it.id].measured and grades[it.id].accepted) for it in items]
    fam_rows = {}
    present = {it.family for it in items}
    for fam in [f for f in FAMILIES if f in present] + sorted(present - set(FAMILIES)):  # canonical order, held-out last
      fi = [it for it in measured if it.family == fam]
      flags = [int(grades[it.id].accepted) for it in fi]
      lo, hi = bootstrap_ci(flags) if len(flags) >= 10 else (float("nan"), float("nan"))
      fam_rows[fam] = {"n": len(fi), "accepted": int(sum(flags)), "acc": _acc(flags), "ci95": [lo, hi],
                       "heldout": fam in HELDOUT_FAMILIES}
    traps = [it for it in measured if not it.feasible]
    feas = [it for it in measured if it.feasible]
    trap_recall = _acc([int(grades[it.id].pred_feasible is False) for it in traps])
    false_refusal = _acc([int(grades[it.id].pred_feasible is False) for it in feas])
    classes = Counter(grades[it.id].error_class for it in measured if not grades[it.id].accepted)
    lo, hi = bootstrap_ci(acc_flags)
    out["arms"][arm] = {
      "n_measured": len(measured), "n_unmeasured": len(items) - len(measured),
      "accepted": int(sum(acc_flags)), "acc": _acc(acc_flags), "ci95": [lo, hi],
      "per_family": fam_rows,
      "in_train_families_acc": _acc([int(grades[it.id].accepted) for it in measured if it.family not in HELDOUT_FAMILIES]),
      "heldout_families_acc": _acc([int(grades[it.id].accepted) for it in measured if it.family in HELDOUT_FAMILIES]),
      "trap_recall": trap_recall, "false_refusal_rate": false_refusal,
      "format_failure_rate": _acc([int(grades[it.id].error_class == "format") for it in measured]),
      "error_classes": dict(classes),
    }
  if baseline in vectors:
    for arm in vectors:
      if arm == baseline:
        continue
      lo, hi = paired_bootstrap_diff(vectors[baseline], vectors[arm])
      out["comparisons"][arm] = {"vs": baseline, "delta_acc": _acc(vectors[arm]) - _acc(vectors[baseline]),
                                 "delta_ci95": [lo, hi], **mcnemar_exact(vectors[baseline], vectors[arm])}
  return out


def results_table(summary: dict) -> str:
  rows = ["| arm | n | accepted | acc | 95% CI | in-train fams | held-out fams | trap recall | false refusal | format fail | Δ vs baseline [CI] | McNemar p |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
  for arm, a in summary["arms"].items():
    cmp = summary["comparisons"].get(arm)
    delta = f"{cmp['delta_acc']:+.3f} [{cmp['delta_ci95'][0]:+.3f}, {cmp['delta_ci95'][1]:+.3f}]" if cmp else "—"
    p = f"{cmp['p_value']:.2g}" if cmp else "—"
    rows.append(f"| {arm} | {a['n_measured']} | {a['accepted']} | {a['acc']:.3f} | [{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}] | "
                f"{a['in_train_families_acc']:.3f} | {a['heldout_families_acc']:.3f} | {a['trap_recall']:.2f} | {a['false_refusal_rate']:.2f} | "
                f"{a['format_failure_rate']:.2f} | {delta} | {p} |")
  return "\n".join(rows)


def family_table(summary: dict) -> str:
  arms = list(summary["arms"])
  fams = list(next(iter(summary["arms"].values()))["per_family"])
  rows = ["| family | n | " + " | ".join(arms) + " |", "|---|---|" + "---|" * len(arms)]
  for fam in fams:
    cells = []
    for arm in arms:
      r = summary["arms"][arm]["per_family"][fam]
      cells.append(f"{r['acc']:.2f} ({r['accepted']}/{r['n']})")
    tag = " (held-out)" if summary["arms"][arms[0]]["per_family"][fam]["heldout"] else ""
    rows.append(f"| {fam}{tag} | {summary['arms'][arms[0]]['per_family'][fam]['n']} | " + " | ".join(cells) + " |")
  return "\n".join(rows)


def dump_results(path, summary: dict, grades_by_arm: dict[str, dict[str, Grade]], provenance: dict) -> None:
  payload = {
    "provenance": provenance,
    "summary": summary,
    "grades": {arm: {iid: (g.to_json() if hasattr(g, "to_json") else (asdict(g) if is_dataclass(g) else g)) for iid, g in gs.items()}
               for arm, gs in grades_by_arm.items()},
  }
  with open(path, "w") as fh:
    json.dump(payload, fh, indent=1, default=str)
