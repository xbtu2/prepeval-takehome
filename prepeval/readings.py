"""Derived readings for the notebook's live narrative.

`derive_readings` turns the graded outputs of a run into a plain-JSON dict of the statistics the story needs
(how often the baseline reasoned before answering, whether refusals track infeasibility, what the accepted items
are, what happened on the held-out families, where few-shot wins landed, what the PoT sandbox saw). The `render_*`
functions turn that dict into Markdown sentences whose wording follows the data: a reading is asserted only when
the numbers support it, counts are stated otherwise, and small runs get a hedge. No number reaches the text
without passing through a formatter, so a NaN never prints.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter

import numpy as np
from scipy.stats import fisher_exact

from . import grading as G
from . import stats as ST
from .families import HELDOUT_FAMILIES

_LEAD_FENCE = re.compile(r"^\s*```(?:json)?\s*", re.IGNORECASE)
_NAME_ERR = re.compile(r"name '(true|false|null)'")


# ----------------------------------------------------------------------------------------------------
# formatting (every number in the narrative goes through one of these)
# ----------------------------------------------------------------------------------------------------
def _finite(x) -> bool:
  try:
    return x is not None and math.isfinite(float(x))
  except (TypeError, ValueError):
    return False


def fmt_num(x, nd: int = 3) -> str:
  return f"{float(x):.{nd}f}" if _finite(x) else "n/a"


def fmt_signed(x, nd: int = 3) -> str:
  return f"{float(x):+.{nd}f}" if _finite(x) else "n/a"


def pct(x, nd: int = 0) -> str:
  return f"{100 * float(x):.{nd}f} %" if _finite(x) else "n/a"


def frac(k, n) -> str:
  return f"{k}/{n} ({pct(k / n)})" if n else f"{k}/0 (n/a)"


def fmt_ci(ci, nd: int = 3) -> str:
  if not ci or len(ci) != 2 or not all(_finite(v) for v in ci):
    return ""
  return f"[{float(ci[0]):.{nd}f}, {float(ci[1]):.{nd}f}]"


def fmt_p(p) -> str:
  if not _finite(p):
    return "p n/a"
  p = float(p)
  return "p < 0.001" if p < 1e-3 else f"p = {p:.2g}"


def _list(d: dict) -> str:
  """{'a': 2, 'b': 1} -> 'a 2, b 1' (largest first)."""
  return ", ".join(f"{k} {v}" for k, v in sorted(d.items(), key=lambda kv: -kv[1])) if d else "none"


# ----------------------------------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------------------------------
def has_reasoning(raw: str | None) -> bool:
  """True when there is non-whitespace text before the first '{' (after one leading ```json fence)."""
  s = _LEAD_FENCE.sub("", raw or "", count=1)
  i = s.find("{")
  return bool((s if i < 0 else s[:i]).strip())


def fisher(a_yes: int, a_n: int, b_yes: int, b_n: int):
  """Two-sided Fisher exact p for refusal rates in two groups; None when any margin is zero."""
  table = [[a_yes, a_n - a_yes], [b_yes, b_n - b_yes]]
  if min(sum(r) for r in table) == 0 or min(sum(c) for c in zip(*table)) == 0:
    return None
  return float(fisher_exact(table, alternative="two-sided").pvalue)


def _split(items, grades) -> dict:
  acc = [it for it in items if grades[it.id].accepted]
  solved = sum(it.feasible for it in acc)
  return {"n": len(items), "accepted": len(acc), "solved_feasible": int(solved), "trap_refusals": len(acc) - int(solved),
          "n_feasible": sum(it.feasible for it in items), "n_traps": sum(not it.feasible for it in items),
          "acc": (len(acc) / len(items)) if items else float("nan")}


def _schema_exact(it, g) -> bool:
  if not isinstance(g.pred, dict):
    return False
  return {G.norm_key(k) for k in g.pred} == {G.norm_key(k) for k in it.schema}


def derive_readings(test, grades_by_arm, records_by_arm, summary, fewshot=None, caps=None) -> dict:
  """Plain-JSON statistics over the graded outputs; every value is int, float, None, str, list or dict."""
  n_traps = sum(not it.feasible for it in test)
  R: dict = {"n_items": len(test), "n_traps": n_traps, "n_feasible": len(test) - n_traps, "caps": dict(caps or {}),
             "arms": {}, "comparisons": {}, "fewshot": None, "pot_sandbox": None, "heldout_any_solved_feasible": False}

  for arm, grades in grades_by_arm.items():
    recs = records_by_arm.get(arm, [])
    tool_arm = bool(recs) and recs[0].get("graded_text") is not None
    items = [it for it in test if it.id in grades]
    traps = [it for it in items if not it.feasible]
    feas = [it for it in items if it.feasible]

    def refused(it):
      return grades[it.id].pred_feasible is False

    parsed_t = [it for it in traps if grades[it.id].pred_feasible is not None]
    parsed_f = [it for it in feas if grades[it.id].pred_feasible is not None]
    tr, fe = sum(map(refused, parsed_t)), sum(map(refused, parsed_f))
    acc_items = [it for it in items if grades[it.id].accepted]
    solved = int(sum(it.feasible for it in acc_items))

    # Would any refused feasible item have passed with the flag forced true? Re-grade exactly.
    would_pass = 0
    for it in feas:
      g = grades[it.id]
      if refused(it) and isinstance(g.pred, dict):
        try:
          would_pass += int(G.grade(it, json.dumps({**g.pred, "feasible": True}, default=str)).accepted)
        except Exception:
          pass

    in_train = [it for it in items if it.family not in HELDOUT_FAMILIES]
    heldout = [it for it in items if it.family in HELDOUT_FAMILIES]
    R["arms"][arm] = {
      "n": len(items), "accepted": len(acc_items),
      "n_with_reasoning": None if tool_arm else int(sum(has_reasoning(r.get("raw")) for r in recs)),
      "refusals": {"trap": int(tr), "trap_n": len(parsed_t), "feasible": int(fe), "feasible_n": len(parsed_f),
                   "unparsed": len(items) - len(parsed_t) - len(parsed_f), "fisher_p": fisher(tr, len(parsed_t), fe, len(parsed_f))},
      "accepted_split": {"solved_feasible": solved, "trap_refusals": len(acc_items) - solved},
      "forced_feasible": {"refused_feasible": int(sum(map(refused, feas))), "would_pass": int(would_pass),
                          "acc": ((solved + would_pass) / len(items)) if items else float("nan")},
      "split": {"in_train": _split(in_train, grades), "heldout": _split(heldout, grades)},
      "heldout_schema_exact": int(sum(_schema_exact(it, grades[it.id]) for it in heldout)),
      "cap_hits": int(sum(bool(r.get("hit_token_cap")) for r in recs)),
      "error_classes": dict(summary["arms"].get(arm, {}).get("error_classes", {})),
    }

  base = grades_by_arm.get("baseline")
  if base is not None:
    for arm, grades in grades_by_arm.items():
      if arm == "baseline":
        continue
      b_only, a_only, wins_split = Counter(), Counter(), Counter()
      for it in test:
        a = it.id in base and base[it.id].accepted
        b = it.id in grades and grades[it.id].accepted
        if b and not a:
          b_only[it.family] += 1
          wins_split["solved_feasible" if it.feasible else "trap_refusals"] += 1
        elif a and not b:
          a_only[it.family] += 1
      R["comparisons"][arm] = {"b_only_by_family": dict(b_only), "a_only_by_family": dict(a_only),
                               "b_only": sum(b_only.values()), "a_only": sum(a_only.values()),
                               "heldout_wins": sum(v for f, v in b_only.items() if f in HELDOUT_FAMILIES),
                               "wins_split": {"solved_feasible": wins_split["solved_feasible"], "trap_refusals": wins_split["trap_refusals"]}}

  if fewshot and "fewshot" in R["comparisons"]:
    cmp = R["comparisons"]["fewshot"]
    ex_fams = [ex.family for ex in fewshot]
    bof = cmp["b_only_by_family"]
    top_fam, top_n = (max(bof.items(), key=lambda kv: kv[1]) if bof else (None, 0))
    excl = [it for it in test if it.family not in ex_fams]
    fs = grades_by_arm["fewshot"]
    va = [int(it.id in base and base[it.id].accepted) for it in excl]
    vb = [int(it.id in fs and fs[it.id].accepted) for it in excl]
    mc = ST.mcnemar_exact(va, vb) if excl else {"b_only": 0, "a_only": 0, "discordant": 0, "p_value": float("nan")}
    R["fewshot"] = {"exemplar_families": ex_fams, "top_family": top_fam, "top_n": int(top_n),
                    "top_share": (top_n / cmp["b_only"]) if cmp["b_only"] else None, "b_only": cmp["b_only"], "a_only": cmp["a_only"],
                    "wins_split": cmp["wins_split"],
                    "excluding_exemplar_families": {"n": len(excl), "accepted_baseline": int(sum(va)), "accepted_fewshot": int(sum(vb)),
                                                    "b_only": mc["b_only"], "a_only": mc["a_only"], "p_value": mc["p_value"]}}

  if "pot" in records_by_arm:
    recs = records_by_arm["pot"]
    status = Counter((r.get("sandbox") or "no_code") for r in recs)
    status_cap = Counter(((r.get("sandbox") or "no_code"), bool(r.get("hit_token_cap"))) for r in recs)
    rejected, rej_cap = status["rejected"], status_cap[("rejected", True)]
    truefalse = sum(1 for r in recs if r.get("sandbox") == "runtime_error" and _NAME_ERR.search(r.get("sandbox_err") or ""))
    other = [v for a, v in (caps or {}).items() if a != "pot"]
    R["pot_sandbox"] = {"status": dict(status), "status_cap": {f"{s}|{'cap' if c else 'nocap'}": v for (s, c), v in status_cap.items()},
                        "rejected": int(rejected), "rejected_cap_hits": int(rej_cap), "rejected_all_cap_hits": bool(rejected and rej_cap == rejected),
                        "runtime_errors": int(status["runtime_error"]), "runtime_truefalse": int(truefalse),
                        "cap": (caps or {}).get("pot"), "other_cap": max(other) if other else None}

  R["heldout_any_solved_feasible"] = any(a["split"]["heldout"]["solved_feasible"] > 0 for a in R["arms"].values())
  return R


# ----------------------------------------------------------------------------------------------------
# narrative
# ----------------------------------------------------------------------------------------------------
def hedge(summary: dict, smoke: bool) -> str:
  n = summary["n_items"]
  if smoke:
    return (f"*Smoke run: n = {n} items on a small model with a short token cap. The sentences below exercise the code "
            f"path; the counts are too small to be results.*")
  if n < 100:
    return f"*n = {n}; differences under about 10 points are within noise.*"
  return ""


def _flag_verdict(ref: dict) -> str:
  p, tn, fn = ref["fisher_p"], ref["trap_n"], ref["feasible_n"]
  small = " (under 10 parsed flags on one side)" if min(tn, fn) < 10 else ""
  if p is None:
    return "too few parsed flags to test" + small
  rate_t = ref["trap"] / tn if tn else 0.0
  rate_f = ref["feasible"] / fn if fn else 0.0
  if p >= 0.1:
    return "the flag carries no information, so its trap refusals are refusals it would have made anyway" + small
  if p < 0.05 and rate_t > rate_f:
    return "the flag carries information" + small
  return "inconclusive" + small


def _arm_headline(summary: dict, arm: str) -> str:
  a = summary["arms"][arm]
  s = f"{arm} {fmt_num(a['acc'])} {fmt_ci(a['ci95'])}".rstrip()
  cmp = summary["comparisons"].get(arm)
  if cmp:
    s += f" ({fmt_signed(cmp['delta_acc'])} vs baseline, {fmt_p(cmp['p_value'])})"
  return s


def floor_lines(summary: dict, controls: dict) -> str:
  floor = controls["effective_floor"]
  rows = [f"Effective floor (strongest non-solving control): **{fmt_num(floor)}**. Margin over the floor is acc − floor; "
          f"the normalised position is (acc − floor)/(1 − floor)."]
  for arm, a in summary["arms"].items():
    acc = a["acc"]
    norm = (acc - floor) / (1 - floor) if _finite(acc) and floor < 1 else float("nan")
    rows.append(f"- `{arm}`: acc {fmt_num(acc)}, margin {fmt_signed(acc - floor if _finite(acc) else None)}, "
                f"normalised {fmt_signed(norm)}; errors {_list(a['error_classes'])}")
  return "\n".join(rows)


def render_items_line(test, train_all, manifest: dict) -> str:
  fams = Counter(it.family for it in test)
  n_trap = sum(not it.feasible for it in test)
  return (f"This run evaluates **{len(test)} items** ({_list(dict(fams))}; {n_trap} infeasible traps = {pct(n_trap / len(test))}). "
          f"The training split holds {len(train_all)} rows in {len({it.family for it in train_all})} families. "
          f"Build-time leak check: {'pass' if manifest['leak_check']['PASS'] else 'FAIL'} "
          f"({', '.join(f'{k} {v}' for k, v in manifest['leak_check'].items() if k != 'PASS')}).")


def render_rows_line(train_items, lens, max_len: int) -> str:
  fams = Counter(it.family for it in train_items)
  n_trap = sum(not it.feasible for it in train_items)
  over = sum(l > max_len for l in lens)
  return (f"**{len(train_items)} training rows** ({_list(dict(fams))}); infeasible {frac(n_trap, len(train_items))}. "
          f"Token length median {int(np.median(lens)) if lens else 'n/a'}, longest {max(lens) if lens else 'n/a'}, "
          f"cap {max_len}; {over} row{'s' if over != 1 else ''} would be truncated.")


def render_prompting(summary: dict, R: dict, controls: dict, timing: dict, caps: dict, batch_size: int, device_name: str,
                     smoke: bool = False) -> str:
  arms = [a for a in summary["arms"] if a in R["arms"]]
  n = summary["n_items"]
  floor = controls["effective_floor"]
  out: list[str] = []
  h = hedge(summary, smoke)
  if h:
    out.append(h)

  # headline + wall time (the latency argument from section 2)
  secs = ", ".join(f"{a} {timing[a] / max(1, n):.2f} s" for a in arms if a in timing)
  out.append(f"**This run.** {'; '.join(_arm_headline(summary, a) for a in arms)}; n = {n}. "
             f"Generation time per item at batch size {batch_size} on {device_name}: {secs}"
             + (" (PoT's sandbox execution is not included)." if "pot" in arms else "."))

  # (a) did the baseline reason before answering?
  if "baseline" in R["arms"]:
    b = R["arms"]["baseline"]
    nr, nb = b["n_with_reasoning"], b["n"]
    if nr is not None and nb:
      bare = nb - nr
      if bare / nb >= 0.9:
        s = (f"**The baseline answered without working.** {bare}/{nb} baseline completions are a bare JSON object with "
             f"nothing before it, although the system message invites the model to work through the arithmetic. The "
             f"comparison below is therefore direct answer against guided reasoning (fewshot), against code (pot), and "
             f"later against a fine-tune taught to write a trace.")
      elif bare / nb <= 0.1:
        s = f"**The baseline worked before answering** on {nr}/{nb} items."
      else:
        s = f"**The baseline mixed modes:** {bare}/{nb} completions are a bare JSON object and {nr} carry working before it."
      sp = b["accepted_split"]
      s += (f" Its {b['accepted']} accepted items are {sp['solved_feasible']} solved feasible items "
            f"({pct(sp['solved_feasible'] / R['n_feasible']) if R['n_feasible'] else 'n/a'} of {R['n_feasible']} feasible) "
            f"and {sp['trap_refusals']} trap refusals.")
      out.append(s)

  # (b) the floor and the feasible flag
  above = [a for a in arms if _finite(summary["arms"][a]["acc"]) and summary["arms"][a]["acc"] > floor]
  where = "above" if len(above) == len(arms) else ("below" if not above else "around")
  lines = [f"**Why the prompting arms sit {where} the always-refuse floor of {fmt_num(floor)}.** A policy that refuses "
           f"every item scores the trap share. To beat it a model has to solve feasible items faster than it loses traps."]
  for a in arms:
    ref = R["arms"][a]["refusals"]
    ff = R["arms"][a]["forced_feasible"]
    lines.append(f"- `{a}`: refuses {ref['trap']}/{ref['trap_n']} traps vs {ref['feasible']}/{ref['feasible_n']} feasible items "
                 f"({fmt_p(ref['fisher_p'])}): {_flag_verdict(ref)}. Forced `feasible = true` would score {fmt_num(ff['acc'])} "
                 f"(recorded {fmt_num(summary['arms'][a]['acc'])})"
                 + (f"; {ff['would_pass']} refused feasible item(s) carried correct values." if ff["would_pass"] else "."))
  out.append("\n".join(lines))

  # (d) few-shot: where did the wins land?
  if R.get("fewshot") and "fewshot" in summary["comparisons"]:
    f = R["fewshot"]
    cmp = summary["comparisons"]["fewshot"]
    ex = f["exemplar_families"]
    if f["b_only"] == 0:
      out.append(f"**Few-shot produced no item-level wins over the baseline on this run**; {f['a_only']} items went the other way "
                 f"({fmt_p(cmp['p_value'])}).")
    else:
      bof = R["comparisons"]["fewshot"]["b_only_by_family"]
      ws = f["wins_split"]
      ex_txt = ", ".join(f"`{e}`" for e in dict.fromkeys(ex))
      if f["top_share"] is not None and f["top_share"] >= 0.6 and f["top_family"] in ex:
        e = f["excluding_exemplar_families"]
        s = (f"**The few-shot gain is one family copying one example.** Of the {f['b_only']} items fewshot wins and the baseline "
             f"loses, {f['top_n']} are `{f['top_family']}`, the family of a worked example. Excluding the exemplar families "
             f"({ex_txt}; n = {e['n']}) fewshot {'wins' if e['accepted_fewshot'] > e['accepted_baseline'] else 'loses'}: "
             f"{e['accepted_fewshot']} against {e['accepted_baseline']} accepted, exact {fmt_p(e['p_value'])}. The gain is "
             f"confined to the exemplar's family; read the headline delta of {fmt_signed(cmp['delta_acc'])} as a property of these exemplars.")
      else:
        s = (f"**Few-shot wins were spread across families on this run** ({_list(bof)}); exemplar families {ex_txt}. "
             f"Delta against the baseline {fmt_signed(cmp['delta_acc'])}, {fmt_p(cmp['p_value'])}.")
      s += f" The wins are {ws['solved_feasible']} solved feasible items and {ws['trap_refusals']} trap refusals; losses {f['a_only']}."
      out.append(s)

  # (e) PoT: what the sandbox saw
  if R.get("pot_sandbox") and "pot" in arms:
    p = R["pot_sandbox"]
    st = p["status"]
    s = (f"**PoT's failures sit in the model's programs.** Sandbox outcomes: {st.get('ok', 0)} ok, {st.get('runtime_error', 0)} "
         f"runtime errors, {st.get('rejected', 0)} rejected, {st.get('timeout', 0)} timeouts, {st.get('no_code', 0)} without code.")
    if p["rejected"]:
      s += (f" {p['rejected_cap_hits']}/{p['rejected']} rejected programs hit the token cap, so "
            f"{'all of them are truncations rather than policy rejections' if p['rejected_all_cap_hits'] else 'most rejections are truncations'}.")
    if p["runtime_errors"]:
      s += (f" {p['runtime_truefalse']}/{p['runtime_errors']} runtime errors are `true`/`false`/`null` written as JSON inside "
            f"Python (a dict literal, not a program).")
    cap_hits = R["arms"]["pot"]["cap_hits"]
    if p["cap"] is not None and p["other_cap"] is not None and p["cap"] < p["other_cap"]:
      s += (f" PoT ran at {p['cap']} new tokens against {p['other_cap']} for the other arms and hit its cap on {cap_hits} items; "
            f"a PoT-below-baseline reading is partly a budget artefact.")
    else:
      s += f" PoT hit its {p['cap']}-token cap on {cap_hits} items."
    out.append(s)

  out.append(f"**{len(above)} of the {len(arms)} prompting arms clear the floor.** The next three sections try the weight-level lever.")
  return "\n\n".join(out)


def render_sft(summary: dict, R: dict, controls: dict, lora_prov: dict | None, smoke: bool = False) -> str:
  if "sft" not in summary["arms"] or "sft" not in R["arms"]:
    return "*The sft arm did not run (ARMS excludes it); nothing to read.*"
  arms = [a for a in summary["arms"] if a in R["arms"]]
  floor = controls["effective_floor"]
  out: list[str] = []
  h = hedge(summary, smoke)
  if h:
    out.append(h)

  cmp = summary["comparisons"].get("sft", {})
  above = [a for a in arms if _finite(summary["arms"][a]["acc"]) and summary["arms"][a]["acc"] > floor]
  out.append(f"**Headline.** baseline {fmt_num(summary['arms'].get('baseline', {}).get('acc'))} → sft "
             f"{fmt_num(summary['arms']['sft']['acc'])} {fmt_ci(summary['arms']['sft']['ci95'])} on n = {summary['n_items']} items; "
             f"delta {fmt_signed(cmp.get('delta_acc'))} {fmt_ci(cmp.get('delta_ci95'))}, McNemar {fmt_p(cmp.get('p_value'))}. "
             f"Arms above the always-refuse floor ({fmt_num(floor)}): {', '.join(f'`{a}`' for a in above) if above else 'none'}.")

  # (c) in-train vs held-out
  s_sft, s_bl = R["arms"]["sft"]["split"], R["arms"].get("baseline", {}).get("split")
  ho, tr = s_sft["heldout"], s_sft["in_train"]
  ho_share = (ho["solved_feasible"] / ho["n_feasible"]) if ho["n_feasible"] else None
  tr_share = (tr["solved_feasible"] / tr["n_feasible"]) if tr["n_feasible"] else 0.0
  if tr_share < 0.2:
    head = "**SFT did not learn the trained procedures on this run.**"
  elif ho["n"] == 0:
    head = "**SFT learned the trained procedures.**"
  elif ho_share == 0:
    head = "**SFT learned the trained procedures and transferred nothing.**"
  elif ho_share < 0.2:
    head = "**SFT learned the trained procedures and transferred little.**"
  else:
    head = "**SFT learned the trained procedures and part of it transferred.**"
  s = (f"{head} In the trained families SFT solves {tr['solved_feasible']}/{tr['n_feasible']} feasible items"
       + (f" where the baseline solved {s_bl['in_train']['solved_feasible']}." if s_bl else "."))
  if ho["n"]:
    if not R["heldout_any_solved_feasible"]:
      s += f" On the {ho['n']} held-out items no arm, SFT included, solves a single feasible item (0/{ho['n_feasible']})."
    else:
      s += (f" On the {ho['n']} held-out items SFT solves {ho['solved_feasible']}/{ho['n_feasible']} feasible items"
            + (f" (baseline {s_bl['heldout']['solved_feasible']})." if s_bl else "."))
    refs = ", ".join(f"{a} {R['arms'][a]['split']['heldout']['trap_refusals']}/{R['arms'][a]['split']['heldout']['n_traps']}" for a in arms)
    ho_floor = ho["n_traps"] / ho["n"]
    pos = ["below" if R["arms"][a]["split"]["heldout"]["acc"] < ho_floor else ("at" if R["arms"][a]["split"]["heldout"]["acc"] == ho_floor else "above")
           for a in arms]
    s += (f" Held-out trap refusals: {refs}. The always-refuse policy scores {fmt_num(ho_floor)} on the held-out items; "
          + ("every arm sits below it there." if all(p == "below" for p in pos) else
             "arms sit " + ", ".join(f"{a} {p}" for a, p in zip(arms, pos)) + " it there."))
    s += (f" SFT emits exactly the schema's keys on {R['arms']['sft']['heldout_schema_exact']}/{ho['n']} held-out items, so "
          f"{'the failure is content, not format' if R['arms']['sft']['heldout_schema_exact'] >= 0.9 * ho['n'] else 'format is also part of the failure'}.")
    if ho_share == 0 and tr_share >= 0.2:
      s += " The held-out design did its job: it separated procedure from template, and the answer on this run was template."
  out.append(s)

  # the flag after training
  ref = R["arms"]["sft"]["refusals"]
  out.append(f"**SFT's `feasible` flag:** refuses {ref['trap']}/{ref['trap_n']} traps against {ref['feasible']}/{ref['feasible_n']} "
             f"feasible items ({fmt_p(ref['fisher_p'])}); {_flag_verdict(ref)}.")

  # paired transitions
  if "sft" in R["comparisons"]:
    c = R["comparisons"]["sft"]
    out.append(f"**Paired against the baseline**, SFT wins {c['b_only']} items and loses {c['a_only']} ({fmt_p(cmp.get('p_value'))}); "
               f"wins by family: {_list(c['b_only_by_family'])}; losses by family: {_list(c['a_only_by_family'])}; "
               f"held-out wins: {c['heldout_wins']} ({c['wins_split']['trap_refusals']} of all wins are trap refusals).")

  if lora_prov:
    first = lora_prov.get("first_step_under_0_1")
    mean_loss = lora_prov.get("mean_train_loss", lora_prov.get("final_loss"))
    last = lora_prov.get("last_logged_loss")
    out.append(f"**Training:** {lora_prov.get('steps')} steps, mean training loss over the epoch {fmt_num(mean_loss)}"
               + (f", last logged step loss {fmt_num(last)}" if last is not None else "") + "; "
               + (f"the loss first fell under 0.1 at step {first}." if first else "the loss never fell under 0.1."))
  return "\n\n".join(out)
