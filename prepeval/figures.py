"""Notebook figures: acceptance, outcome stacks, refusal scatter, training diagnostics, paired transitions.

Every function takes plain data (the `summary` dict from `stats.summarize`, record lists, grade maps) and
returns a matplotlib Figure, or None when the inputs are empty or a required arm is missing. The module
never selects a backend and never calls `plt.show()`; the caller decides both.

Palette: the project's validated reference set. Categorical hues follow the arm name (never its rank);
magnitude uses the one-hue blue ramp; text wears text tokens, never a series colour.
"""

from __future__ import annotations

import math

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.textpath import TextPath
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter

from .families import FAMILIES, HELDOUT_FAMILIES

try:  # the readings module owns the p-value format when it exists
  from .readings import fmt_p  # type: ignore
except ImportError:

  def fmt_p(p) -> str:
    """None/NaN -> 'p n/a'; p < 1e-3 -> 'p < 0.001'; else 'p = 0.034' (two significant digits)."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
      return "p n/a"
    if p < 1e-3:
      return "p < 0.001"
    return f"p = {p:.2g}"


plt.rcParams["font.family"] = "DejaVu Sans"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots in fixed order
ARM_COLOR = {"baseline": SERIES[0], "fewshot": SERIES[1], "pot": SERIES[2], "sft": SERIES[3]}
BLUE = {600: "#184f95", 450: "#2a78d6", 350: "#5598e7", 250: "#86b6ef"}  # one-hue ordinal ramp
RED = "#e34948"
MUTED, GRID, TEXT, TEXT2, SURFACE = "#898781", "#e1e0d9", "#0b0b0b", "#52514e", "#fcfcfb"

OUTCOMES = [("solved feasible", BLUE[600]), ("refused trap", BLUE[450]),
            ("wrong value", SERIES[1]), ("feasibility", SERIES[2]), ("format", SERIES[3])]
_ERROR_SLOT = {"wrong_value": 2, "feasibility": 3, "format": 4}


# ----------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------
def _finite(v) -> bool:
  try:
    return v is not None and math.isfinite(float(v))
  except (TypeError, ValueError):
    return False


def _style(ax, grid="y") -> None:
  """Recessive chrome: surface face, no top/right spines, hairline grid on one axis only."""
  ax.set_facecolor(SURFACE)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    ax.spines[side].set_color(GRID)
  ax.tick_params(colors=MUTED, labelcolor=TEXT2, length=0)
  for axis, on in ((ax.xaxis, grid in ("x", "both")), (ax.yaxis, grid in ("y", "both"))):
    if on:
      axis.grid(True, color=GRID, linewidth=1)
    else:
      axis.grid(False)
  ax.set_axisbelow(True)


def _legend(target, **kw):
  """Frameless legend with secondary-ink text; `target` is an Axes or a Figure."""
  kw.setdefault("frameon", False)
  kw.setdefault("fontsize", 9)
  leg = target.legend(**kw)
  for t in leg.get_texts():
    t.set_color(TEXT2)
  return leg


def _title(ax, text: str, **kw) -> None:
  ax.set_title(text, color=TEXT, fontsize=11, loc="left", **kw)


def _arm_color(arm: str, j: int) -> str:
  return ARM_COLOR.get(arm, SERIES[j % len(SERIES)])


def _fam_label(fam: str, heldout: bool) -> str:
  return fam + (" (held-out)" if heldout else "")


def _families(test, families=None) -> list[str]:
  """Canonical family order (FAMILIES first, unknown families after, in order of appearance)."""
  if families:
    return list(families)
  seen = list(dict.fromkeys(it.family for it in test))
  return [f for f in FAMILIES if f in seen] + [f for f in seen if f not in FAMILIES]


def _ink_on(fill: str) -> str:
  """Ink for a label set inside a fill: surface-white on dark fills, text-black on light ones."""
  lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in mcolors.to_rgb(fill)]
  lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
  return SURFACE if lum < 0.2 else TEXT


def _place_labels(ax, pts, texts, colors, fontsize=9, min_dy=28.0, min_dx=96.0) -> None:
  """Direct labels beside points. A label that would sit on an earlier one is pushed down one row."""
  fig = ax.figure
  placed: list[tuple[float, float]] = []
  order = sorted(range(len(pts)), key=lambda i: (-pts[i][1], pts[i][0]))
  for i in order:
    x, y = pts[i]
    px, py = ax.transData.transform((x, y)) * 72.0 / fig.dpi  # points
    left = x > 0.72
    dx = -12.0 if left else 12.0
    dy = 0.0
    while any(abs(px + dx - qx) < min_dx and abs(py + dy - qy) < min_dy for qx, qy in placed):
      dy -= min_dy
    placed.append((px + dx, py + dy))
    leader = dict(arrowstyle="-", color=GRID, lw=0.8, shrinkA=6, shrinkB=2) if dy else None  # displaced -> thin leader
    ax.annotate(texts[i], (x, y), xytext=(dx, dy), textcoords="offset points", ha="right" if left else "left",
                va="center", color=colors[i], fontsize=fontsize, linespacing=1.25, arrowprops=leader)


def _label_floor(ax, floor: float, tops: list[float], k: int, text: str = "effective floor", fontsize: float = 8) -> None:
  """Label a horizontal reference line in the first x-interval (left edge, gaps, right edge) free of bars that
  reach it; when no interval is wide enough, widen the x-range so the label sits in a gutter after the last bar."""
  fig = ax.figure
  width_pts = ax.get_position().width * fig.get_figwidth() * 72.0
  need_pts = TextPath((0, 0), text, size=fontsize).get_extents().width + 8.0
  edges = [-0.5] + [e for i, t in enumerate(tops) if t > floor - 0.01 for e in (i - 0.3, i + 0.3)] + [k - 0.5]
  free = [(a, b) for a, b in zip(edges[::2], edges[1::2]) if (b - a) * width_pts / k >= need_pts]
  kw = dict(textcoords="offset points", va="bottom", color=TEXT2, fontsize=fontsize, zorder=4,
            bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=1.5))
  if not free:
    gutter = need_pts * k / max(width_pts - need_pts, 1.0)
    ax.set_xlim(-0.5, k - 0.5 + gutter)
    free = [(k - 0.5, k - 0.5 + gutter)]
  a, b = free[0]
  if a <= -0.5:
    ax.annotate(text, (-0.5, floor + 0.012), xytext=(4, 0), ha="left", **kw)
  elif b >= k - 0.5:
    ax.annotate(text, (b, floor + 0.012), xytext=(-4, 0), ha="right", **kw)
  else:
    ax.annotate(text, ((a + b) / 2, floor + 0.012), xytext=(0, 0), ha="center", **kw)


# ----------------------------------------------------------------------------------------------------
# 1. acceptance per arm and per family
# ----------------------------------------------------------------------------------------------------
def fig_acceptance(summary: dict, controls: dict, families=None):
  """Left: acceptance per arm with 95 % CI whiskers and the effective floor. Right: per-family dots per arm."""
  arms = list(summary.get("arms", {}))
  if not arms:
    return None
  first = summary["arms"][arms[0]]
  fams = list(families) if families else list(first.get("per_family", {}))
  if not fams:
    return None
  n_items = summary.get("n_items") or sum(first["per_family"][f]["n"] for f in fams)

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.5]})
  fig.patch.set_facecolor(SURFACE)
  _style(ax1, "y")
  _style(ax2, "x")

  # left: one bar per arm, single hue, CI whiskers (none when the CI is NaN), value label above the whisker
  x = np.arange(len(arms))
  acc = [summary["arms"][a]["acc"] for a in arms]
  ax1.bar(x, [v if _finite(v) else 0.0 for v in acc], width=0.55, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
  tops = [0.0] * len(arms)  # top of each bar's value label, in data units (for placing the floor label)
  for j, (xi, a, v) in enumerate(zip(x, arms, acc)):
    if not _finite(v):
      continue
    lo, hi = summary["arms"][a].get("ci95", (float("nan"), float("nan")))
    top = v
    if _finite(lo) and _finite(hi):
      ax1.errorbar([xi], [v], yerr=[[max(0.0, v - lo)], [max(0.0, hi - v)]], fmt="none", ecolor=TEXT2,
                   elinewidth=1.5, capsize=4, zorder=3)
      top = max(v, hi)
    ax1.text(xi, min(top + 0.025, 1.02), f"{v:.2f}", ha="center", va="bottom", color=TEXT, fontsize=10)
    tops[j] = min(top + 0.025, 1.02) + 0.05
  floor = (controls or {}).get("effective_floor")
  if _finite(floor):
    ax1.axhline(floor, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
  ax1.set_xticks(x, arms)
  ax1.set_xlim(-0.5, len(arms) - 0.5)
  ax1.set_ylim(0, 1.1)
  ax1.set_ylabel("accepted (all fields within tolerance)", color=TEXT2)
  _title(ax1, f"Acceptance by arm, n = {n_items} items, 95 % CI")

  # right: per-family dots, colour follows the arm name
  y = np.arange(len(fams))
  for j, a in enumerate(arms):
    pf = summary["arms"][a]["per_family"]
    vals = [pf[f]["acc"] if f in pf and _finite(pf[f]["acc"]) else float("nan") for f in fams]
    ax2.scatter(vals, y + (j - (len(arms) - 1) / 2) * 0.16, s=64, color=_arm_color(a, j), edgecolor=SURFACE,
                linewidth=1.5, label=a, zorder=3)
  ax2.set_yticks(y, [_fam_label(f, bool(first["per_family"].get(f, {}).get("heldout", f in HELDOUT_FAMILIES))) for f in fams])
  ax2.invert_yaxis()
  ax2.set_xlim(-0.02, 1.02)
  ax2.set_xlabel("accepted", color=TEXT2)
  n_fam = first["per_family"][fams[0]]["n"] if fams[0] in first["per_family"] else 0
  _title(ax2, f"Per family (n = {n_fam} each; directional)")
  _legend(ax2, loc="lower right")
  fig.tight_layout()
  if _finite(floor):  # after layout, so the label is measured against the final axes width
    _label_floor(ax1, float(floor), tops, len(arms))
  return fig


# ----------------------------------------------------------------------------------------------------
# 2. outcome per item, stacked by family, one panel per arm
# ----------------------------------------------------------------------------------------------------
def _outcome_slot(rec: dict, feasible: bool):
  if rec.get("accepted"):
    return 0 if feasible else 1
  return _ERROR_SLOT.get(rec.get("error_class"))


def fig_outcomes(records_by_arm: dict, test, arms, families=None):
  """Horizontal stacks of the five outcomes (solved, refused trap, wrong value, feasibility, format) per family."""
  present = [a for a in arms if a in records_by_arm and records_by_arm[a]]
  if not present or not test:
    return None
  fams = _families(test, families)
  by_id = {it.id: it for it in test}
  counts = {}
  for a in present:
    rows = {f: [0] * len(OUTCOMES) for f in fams}
    for rec in records_by_arm[a]:
      it = by_id.get(rec.get("item_id"))
      fam = rec.get("family", it.family if it else None)
      if it is None or fam not in rows:
        continue
      slot = _outcome_slot(rec, it.feasible)
      if slot is not None:
        rows[fam][slot] += 1
    counts[a] = rows
  n_max = max((sum(v) for rows in counts.values() for v in rows.values()), default=0)
  if n_max == 0:
    return None

  k = len(present)
  width = 11 if k >= 3 else 4 + 3.4 * k
  fig, axes = plt.subplots(1, k, figsize=(width, 3.9), sharey=True, squeeze=False)
  axes = axes[0]
  fig.patch.set_facecolor(SURFACE)
  y = np.arange(len(fams))
  for ax, a in zip(axes, present):
    _style(ax, "x")
    left = np.zeros(len(fams))
    for s, (label, color) in enumerate(OUTCOMES):
      vals = np.array([counts[a][f][s] for f in fams], dtype=float)
      ax.barh(y, vals, left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=2, label=label)
      for yi, v, l in zip(y, vals, left):
        if v > 0 and v / n_max >= 0.12:  # label only when the text fits inside the segment
          ax.text(l + v / 2, yi, f"{int(v)}", ha="center", va="center", color=_ink_on(color), fontsize=8.5)
      left += vals
    ax.set_xlim(0, n_max)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=min(n_max, 8)))
    ax.set_xlabel("items", color=TEXT2)
    ax.set_title(a, color=TEXT2, fontsize=10, loc="left")
  axes[0].set_yticks(y, [_fam_label(f, f in HELDOUT_FAMILIES) for f in fams])
  axes[0].invert_yaxis()
  handles = [Patch(facecolor=c, edgecolor="none", label=l) for l, c in OUTCOMES]
  fig.suptitle("Outcome per item, by family and arm", x=0.01, y=0.99, ha="left", va="top", color=TEXT, fontsize=11)
  _legend(fig, handles=handles, loc="lower center", ncol=len(OUTCOMES), bbox_to_anchor=(0.5, 0.0), handlelength=1.2)
  fig.tight_layout(rect=(0, 0.1, 1, 0.94))
  return fig


# ----------------------------------------------------------------------------------------------------
# 3. refusal on traps vs refusal on feasible items
# ----------------------------------------------------------------------------------------------------
def fig_refusals(summary: dict, readings: dict):
  """One point per arm: x = false-refusal rate, y = trap recall; constant policies sit on y = x."""
  r_arms = (readings or {}).get("arms", {})
  arms = [a for a in summary.get("arms", {}) if a in r_arms]
  pts, texts, colors = [], [], []
  for j, a in enumerate(arms):
    x, y = summary["arms"][a].get("false_refusal_rate"), summary["arms"][a].get("trap_recall")
    if not (_finite(x) and _finite(y)):
      continue
    p = (r_arms[a].get("refusals") or {}).get("fisher_p")
    pts.append((float(x), float(y)))
    texts.append(f"{a}\n{fmt_p(p)}")
    colors.append(_arm_color(a, j))
  if not pts:
    return None

  fig, ax = plt.subplots(figsize=(10, 6))
  fig.patch.set_facecolor(SURFACE)
  _style(ax, "both")
  lo, hi = -0.03, 1.03
  ax.plot([lo, hi], [lo, hi], color=GRID, linewidth=1, zorder=1)
  ax.text(0.62, 0.595, "constant policies lie here (refusals carry no information)", ha="left", va="top",
          color=MUTED, fontsize=8, rotation=0)
  ax.scatter([0, 1], [0, 1], s=90, facecolors="none", edgecolors=MUTED, linewidth=1.5, zorder=2)
  for (x, y), c in zip(pts, colors):
    ax.scatter([x], [y], s=90, color=c, edgecolor=SURFACE, linewidth=2, zorder=3)
  ax.set_xlim(lo, hi)
  ax.set_ylim(lo, hi)
  ax.set_xlabel("false refusal rate (refused a feasible item)", color=TEXT2)
  ax.set_ylabel("trap recall (refused an infeasible item)", color=TEXT2)
  ax.text(0.995, 0.012, "rates over all items; unparsed outputs count as non-refusals. Fisher p on parsed outputs only.",
          transform=ax.transAxes, ha="right", va="bottom", color=MUTED, fontsize=7.5)
  _title(ax, "Does the feasible flag carry information? refusal on traps vs on feasible items")
  fig.tight_layout()
  # labels last, in display space, so the hollow policy markers and the arm points never overprint
  _place_labels(ax, [(0.0, 0.0), (1.0, 1.0)] + pts, ["always feasible", "always infeasible"] + texts,
                [MUTED, MUTED] + colors)
  return fig


# ----------------------------------------------------------------------------------------------------
# 4. training-row token lengths
# ----------------------------------------------------------------------------------------------------
def fig_train_lengths(lens, max_len: int):
  """Histogram of tokenised training-row lengths against the sequence cap."""
  vals = np.asarray([v for v in (lens or []) if _finite(v)], dtype=float)
  if vals.size == 0:
    return None
  xmax = max(max_len + 64, int(vals.max()) + 64)
  bins = np.arange(0, xmax + 1, 32) if vals.size >= 50 else 8
  fig, ax = plt.subplots(figsize=(10, 3.8))
  fig.patch.set_facecolor(SURFACE)
  _style(ax, "y")
  n, _, _ = ax.hist(vals, bins=bins, color=SERIES[0], edgecolor=SURFACE, linewidth=2, zorder=3)
  top = float(n.max()) * 1.25 if n.size else 1.0
  ax.axvline(max_len, color=MUTED, linewidth=1, zorder=2)
  ax.text(max_len - 0.006 * xmax, top * 0.97, "sequence cap", ha="right", va="top", color=TEXT2, fontsize=8)
  ax.text(0.01, 0.97, f"n = {vals.size}  ·  median {np.median(vals):.0f} tokens  ·  max {int(vals.max())} tokens",
          transform=ax.transAxes, ha="left", va="top", color=TEXT2, fontsize=9)
  ax.set_xlim(0, xmax)
  ax.set_ylim(0, top)
  ax.yaxis.set_major_locator(MaxNLocator(integer=True))
  ax.set_xlabel("tokens per training row (prompt + completion)", color=TEXT2)
  ax.set_ylabel("training rows", color=TEXT2)
  _title(ax, "Training rows fit under the sequence cap")
  fig.tight_layout()
  return fig


# ----------------------------------------------------------------------------------------------------
# 5. LoRA training loss
# ----------------------------------------------------------------------------------------------------
def fig_loss_curve(log_history):
  """Training loss per logged optimizer step; log-scaled when the range spans more than 20x."""
  pts = sorted((float(e["step"]), float(e["loss"])) for e in (log_history or [])
               if isinstance(e, dict) and "loss" in e and "step" in e and _finite(e["loss"]) and _finite(e["step"]))
  if len(pts) < 2:
    return None
  xs, ys = zip(*pts)
  fig, ax = plt.subplots(figsize=(10, 3.8))
  fig.patch.set_facecolor(SURFACE)
  _style(ax, "y")
  ax.plot(xs, ys, color=SERIES[0], linewidth=2, marker="o", markersize=6.5, markeredgecolor=SURFACE,
          markeredgewidth=1.5, solid_joinstyle="round", solid_capstyle="round", zorder=3)
  if min(ys) > 0 and max(ys) / min(ys) > 20:
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=12))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))  # 0.01, 0.02, 0.05, 0.1, ... not 10^-2
    ax.yaxis.set_minor_formatter(NullFormatter())
  for k in (0, -1):  # first and last values
    ax.annotate(f"{ys[k]:.3f}", (xs[k], ys[k]), xytext=(10, 0), textcoords="offset points", ha="left", va="center",
                color=TEXT, fontsize=9)
  span = max(xs[-1] - xs[0], 1.0)
  ax.set_xlim(xs[0] - 0.03 * span, xs[-1] + 0.1 * span)
  ax.xaxis.set_major_locator(MaxNLocator(integer=True))
  ax.set_xlabel("optimizer step", color=TEXT2)
  ax.set_ylabel("training loss (completion tokens)", color=TEXT2)
  _title(ax, "LoRA training loss")
  fig.tight_layout()
  return fig


# ----------------------------------------------------------------------------------------------------
# 6. paired item transitions between two arms
# ----------------------------------------------------------------------------------------------------
def fig_transitions(grades_by_arm: dict, test, arm_a: str = "baseline", arm_b: str = "sft", families=None,
                    comparison: dict | None = None):
  """Diverging bars per family: items broken by arm_b (left, red) and fixed by arm_b (right, blue)."""
  if not test or arm_a not in grades_by_arm or arm_b not in grades_by_arm:
    return None
  ga, gb = grades_by_arm[arm_a], grades_by_arm[arm_b]
  fams = _families(test, families)
  fixed = {f: 0 for f in fams}
  broken = {f: 0 for f in fams}
  both_pass = {f: 0 for f in fams}
  both_fail = {f: 0 for f in fams}
  for it in test:
    if it.family not in fixed or it.id not in ga or it.id not in gb:
      continue
    a, b = bool(ga[it.id].accepted), bool(gb[it.id].accepted)
    if a and b:
      both_pass[it.family] += 1
    elif a:
      broken[it.family] += 1
    elif b:
      fixed[it.family] += 1
    else:
      both_fail[it.family] += 1

  fig, ax = plt.subplots(figsize=(10, 4.4))
  fig.patch.set_facecolor(SURFACE)
  _style(ax, "x")
  y = np.arange(len(fams))
  fx = np.array([fixed[f] for f in fams], dtype=float)
  bk = np.array([broken[f] for f in fams], dtype=float)
  ax.barh(y, fx, height=0.62, color=SERIES[0], edgecolor=SURFACE, linewidth=2, label=f"fixed ({arm_a} fail → {arm_b} pass)", zorder=3)
  ax.barh(y, -bk, height=0.62, color=RED, edgecolor=SURFACE, linewidth=2, label=f"broken ({arm_a} pass → {arm_b} fail)", zorder=3)
  ax.axvline(0, color=MUTED, linewidth=1, zorder=2)
  m = int(max(fx.max() if fx.size else 0, bk.max() if bk.size else 0, 1))
  pad = 0.04 * m
  for yi, f, b in zip(y, fx, bk):
    if f > 0:
      ax.text(f + pad, yi, f"{int(f)}", ha="left", va="center", color=TEXT, fontsize=9)
    if b > 0:
      ax.text(-b - pad, yi, f"{int(b)}", ha="right", va="center", color=TEXT, fontsize=9)
  for yi, f in zip(y, fams):
    ax.text(1.02, yi, f"both pass {both_pass[f]} · both fail {both_fail[f]}", transform=ax.get_yaxis_transform(),
            ha="left", va="center", color=MUTED, fontsize=8)
  lim = m + max(1, math.ceil(0.25 * m))
  ax.set_xlim(-lim, lim)
  ax.xaxis.set_major_locator(MaxNLocator(integer=True))
  ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(int(round(v)))}"))
  ax.set_yticks(y, [_fam_label(f, f in HELDOUT_FAMILIES) for f in fams])
  ax.invert_yaxis()
  ax.set_xlabel(f"← items broken by {arm_b}            items fixed by {arm_b} →", color=TEXT2)
  _title(ax, f"Paired item transitions {arm_a} → {arm_b}", pad=20 if comparison else 6)
  if comparison:
    sub = f"fixed {comparison.get('b_only', '?')}, broken {comparison.get('a_only', '?')}, McNemar {fmt_p(comparison.get('p_value'))}"
    ax.annotate(sub, xy=(0, 1), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points", ha="left",
                va="bottom", color=TEXT2, fontsize=9)
  _legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=1.2)
  fig.tight_layout(rect=(0, 0, 0.8, 1))  # the right margin holds the both-pass / both-fail text
  return fig


# ----------------------------------------------------------------------------------------------------
# 7. what the acceptances are: in-train vs held-out families
# ----------------------------------------------------------------------------------------------------
def fig_heldout_split(summary: dict, readings: dict):
  """Per arm, two stacked bars (in-train, held-out): refused traps below, solved feasible items above."""
  r_arms = (readings or {}).get("arms", {})
  arms = [a for a in summary.get("arms", {}) if a in r_arms and r_arms[a].get("split")]
  if not arms:
    return None
  fig, ax = plt.subplots(figsize=(10, 4.2))
  fig.patch.set_facecolor(SURFACE)
  _style(ax, "y")
  x = np.arange(len(arms))
  for i, a in enumerate(arms):
    split = r_arms[a]["split"]
    for off, key, lab in ((-0.2, "in_train", "in-train"), (0.2, "heldout", "held-out")):
      s = split.get(key) or {}
      n = s.get("n", 0)
      if not n:
        continue
      refused = s.get("trap_refusals", 0) / n
      solved = s.get("solved_feasible", 0) / n
      ax.bar(i + off, refused, 0.36, color=BLUE[250], edgecolor=SURFACE, linewidth=2, zorder=3)
      ax.bar(i + off, solved, 0.36, bottom=refused, color=BLUE[450], edgecolor=SURFACE, linewidth=2, zorder=3)
      acc = s.get("acc", refused + solved)
      acc = acc if _finite(acc) else refused + solved
      ax.text(i + off, refused + solved + 0.015, f"{acc:.3f}", ha="center", va="bottom", color=TEXT, fontsize=9)
      ax.text(i + off, -0.015, lab, ha="center", va="top", color=MUTED, fontsize=7.5)
  ax.set_xticks(x, arms)
  ax.tick_params(axis="x", pad=16)
  ax.set_xlim(-0.6, len(arms) - 0.4)
  ax.set_ylim(0, 1.05)
  ax.set_ylabel("accepted", color=TEXT2)
  first = r_arms[arms[0]]["split"]
  n_in = (first.get("in_train") or {}).get("n", 0)
  n_out = (first.get("heldout") or {}).get("n", 0)
  _title(ax, f"What the acceptances are: in-train (n = {n_in}) vs held-out (n = {n_out}) families")
  handles = [Patch(facecolor=BLUE[450], edgecolor="none", label="solved feasible items"),
             Patch(facecolor=BLUE[250], edgecolor="none", label="refused traps")]
  _legend(ax, handles=handles, loc="upper left", handlelength=1.2)
  fig.tight_layout()
  return fig
