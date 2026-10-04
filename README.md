# Liquid-handling plan reasoning: an eval and an improvement scheme for a 1.5B model

**A liquid handler executes whatever numbers and well addresses the planner emits, so "turn this protocol
intent into volumes, wells and containers" is a reasoning capability worth measuring, and a 1.5B model that
is bad at it can be made much better in under an hour by letting it write code and teaching it the
procedures.**

This is a take-home demo of two skills: building an evaluation and improving a small model on it. It is not
a production eval. Everything runs in one free-tier Google Colab session (T4) in about 35 minutes, or as a
plain Python script on a CPU in smoke mode.

## Quick start

| | |
|---|---|
| Colab | open `notebook/prep_eval_colab.ipynb`, *Runtime → Change runtime type → T4 GPU*, *Run all* |
| Local build and tests | `uv sync --all-extras && uv run pytest` |
| Local smoke run (CPU, 0.5B model, every code path) | `PREPEVAL_SMOKE=1 uv run python notebook/prep_eval_colab.py` |
| Rebuild the item files | `uv run python -m prepeval.dataset` (needs the `plr` extra once, for `labware.json`) |

## 1. The capability

**Reagent-prep and labware reasoning for liquid handling**: given a protocol intent in a scientist's words and
the facts of the bench (stock concentrations, container geometry, pipette limits, lab policies), produce the
exact volumes, concentrations, well lists and container choices a robot would execute, and recognise when the
request cannot be carried out as stated.

Why this one:

- It is the quantitative core of protocol automation: every `transfer`, `serial_dilute` and `normalize`
  step in a protocol compiler bottoms out in this arithmetic and this addressing.
- Small models fail it in characteristic ways (unit slips, inverted dilution factors, miscounted wells,
  "diluting" a sample that is already too dilute), so there is headroom and the failures are legible.
- No public dataset of wet-lab calculation problems with numeric ground truth exists (searched Hugging Face
  and the web; the nearest are college-chemistry sets such as SciBench and ChemistryQA, and LAB-Bench's
  ProtocolQA, which is troubleshooting multiple choice). A generated set is therefore the eval.

## 2. The eval

### Items

240 test items, six families × 40 seeds, generated deterministically by `prepeval/families.py`. Exactly every
fifth seed is an **infeasible trap** (20 %), so a model that always complies and one that always refuses both
fail. Each family has three or four phrasings.

| family | the model must produce | trap | ground truth |
|---|---|---|---|
| `dilute_stock` | stock µL and diluent µL for a target concentration and volume, across unit changes (mM→µM, mg/mL→ng/µL, X-fold) | target above stock; stock volume below the pipette minimum | arithmetic |
| `serial_dilution` | carry volume, diluent per well, discard from the last well, every well's concentration | wells would overflow | arithmetic + PyLabRobot well capacity |
| `master_mix` | per-component totals for N reactions with the stated excess, water to volume | recipe does not fit the reaction volume | arithmetic |
| `normalize_samples` **(held out)** | per-sample sample and diluent volumes to a common concentration and minimum volume | a sample already below target | outcome-graded arithmetic |
| `well_addressing` | expand `B2:D5`, well at column-major index k, the 8 wells under an 8-channel head, 96→384 interleaved-quadrant destination | off-plate addresses | **PyLabRobot** |
| `labware_fit` **(held out)** | plate vs deep-well vs trough for a volume and channel pattern, plates needed, trough load including unreachable volume | nothing fits | **PyLabRobot** labware + arithmetic |

**Prompt rule.** The prompt states the intent; a `World:` block carries every number (stock concentrations,
container facts, pipette minimum, lab policies); grading owns the answer key. The lever is never named: no
prompt says "account for dead volume" or "check feasibility". The only affordance is the JSON schema printed
in the prompt, which includes `"feasible": true|false` as a report channel.

**Ground truth.** Labware facts (rows × columns, well maximum volume) and all well addressing come from
[PyLabRobot](https://github.com/PyLabRobot/pylabrobot) at a pinned commit (`labware.json` caches them;
`tests/test_labware_plr.py` asserts every function agrees with the live PLR objects). The model never sees
PLR identifiers, only the facts. Trough "unreachable volume" is a stated lab policy, not a PLR number.

### Grading

Programmatic, no LLM judge (`prepeval/grading.py`). The last JSON object in the completion is extracted
(fenced, inline, or a Python dict literal). Volumes and concentrations must be within
`max(0.05 µL, 1 %)`; counts exact; well lists exact as sets (`A01` ≡ `A1`); container choice exact. An item is
**accepted** only if every required field passes. For infeasible items only the `feasible` flag is graded: a
specific refusal scores 1, acting anyway scores 0. Refusing a feasible item is a `feasibility` error. For
`normalize_samples` any (sample, diluent) pair that reaches the target within tolerance, meets the minimum
volume and respects the available volume passes (any correct construction is accepted). Every failure is
classed as `format`, `feasibility` or `wrong_value`.

### Controls (run before any model call, in `pytest` and in the notebook)

| control | expected |
|---|---|
| gold answers graded | 100 % accepted |
| each gold corrupted one field at a time (±3 % beyond tolerance, ×1000 unit slip, flipped flag, dropped well) | 100 % rejected |
| a *program* that prints the gold, through sandbox → extractor → grader | 100 % accepted |
| constant policy "always feasible, zeros" | ≈ 0 |
| constant policy "always infeasible" | = trap share (0.20) |
| leak check at build time | no test prompt in train; held-out families absent from train; no non-trivial gold value printed in its own prompt (items that would are resampled); no few-shot value in a test gold |

The strongest non-solving control is the **effective floor**; arms are also reported as position above it.

### Statistics

Acceptance per arm with a 95 % bootstrap CI (10 000 resamples); paired delta against the baseline with its
bootstrap CI and an exact McNemar test on discordant pairs; per-family acceptance (n = 40 each, directional)
with held-out families marked; trap recall and false-refusal rate; format-failure rate; seconds per item.
Decoding is greedy (`pass@1`) with one shared generation config for every arm. Harness failures are counted,
not scored as zero.

## 3. The improvement scheme

Four arms on the same items, same decoding, same model (`Qwen/Qwen2.5-1.5B-Instruct`, Apache-2.0, fp16 on T4).

| arm | what changes | why it should help |
|---|---|---|
| `baseline` | zero-shot: intent + world + schema; the model reasons as it likes and ends with the JSON | the honest reference: instruct models already reason step by step |
| `fewshot` | two worked examples from the training families (a dilution and a well block) | mostly format compliance; for Qwen2.5, worked examples add little reasoning |
| `pot` | Program-of-Thought: the model writes one Python block that prints the JSON; a subprocess sandbox executes it; text JSON is the fallback | arithmetic and well enumeration move into the interpreter; the model only has to set up the formula (PAL/PoT: GSM-Hard 23 → 61 for CoT → code) |
| `sft` | LoRA (r = 16, attention + MLP, one epoch, ~190 steps, ~6 min on T4) on ~1 440 generated items from four families, each with a programmatic reasoning trace and the gold JSON; evaluated with the baseline prompt | teaches the procedures; the two held-out families separate learned procedure from template memorisation |

The fine-tune is honest only because of the split: training seeds are disjoint from test seeds, no training
prompt equals a test prompt, and `normalize_samples` and `labware_fit` never appear in training. In-family
and held-out acceptance are reported separately.

## 4. Results

> **Fill in from the Colab run.** The notebook prints a README-ready block (the two tables below) at the end
> and writes `results/results.json`, `results/figure.png` and `results/raw_outputs.jsonl`.

Headline (n = 240, greedy, Qwen2.5-1.5B-Instruct, T4):

| arm | n | accepted | acc | 95% CI | in-train fams | held-out fams | trap recall | false refusal | format fail | Δ vs baseline [CI] | McNemar p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | | | | | | | | | | — | — |
| fewshot | | | | | | | | | | | |
| pot | | | | | | | | | | | |
| sft | | | | | | | | | | | |

Per family (n = 40 each; directional):

| family | baseline | fewshot | pot | sft |
|---|---|---|---|---|
| | | | | |

A CPU smoke run (0.5B model, 4 items per family, 3 LoRA steps) is kept in `results/smoke-cpu.json` only as
proof that every code path executes; its numbers are not results.

## 5. What this does and does not show

- **Do not claim** that frontier models would struggle here. Items of this shape (documented facts plus a
  goal-level derivation) are the kind the top tier clears on first contact; the point is the gap between a
  small model and the ceiling, and how much of it two cheap interventions close.
- A scripted solver tops this eval by construction. That is precisely why tool use is a legitimate
  intervention: the model's job becomes setting up the right formula from scientist language and world facts.
- Per-family numbers at n = 40 carry ±15 pp; the n = 240 aggregate carries the claim.
- A pass means "the bookkeeping came out right", never "the assay would have worked".
- Items are generated, not real protocol text. The vocabulary and operation set mirror a protocol
  intermediate representation (transfer, serial dilute, normalise, master-mix assembly, dead volume, equal
  final volumes), but no real protocol was used.
- `feasible` in the schema is a deliberate affordance so a small model has a report channel at all; a stricter
  design would make infeasibility discoverable only from the world.

## 6. Layout

```
prepeval/
  families.py    six seeded generators: prompt (intent + world + schema), gold, trace, grader spec
  labware.py     PyLabRobot wrappers + labware.json cache: catalogue, ranges, column-major index, quadrants
  grading.py     JSON extraction, per-field tolerance / set match, feasibility, error class, controls
  prompts.py     arm prompts (baseline, few-shot, PoT), SFT target
  sandbox.py     Program-of-Thought executor: subprocess, AST filter, builtins whitelist, 2 s timeout
  dataset.py     deterministic splits, leak checks, manifest
  stats.py       bootstrap CI, exact McNemar, per-family tables, results JSON
data/            test.jsonl (240), train.jsonl (~1 440), fewshot.jsonl (2), manifest.json (seeds, sha256, PLR commit)
notebook/        prep_eval_colab.py (jupytext percent source) and the built .ipynb
tests/           hand-worked golds, grader controls, extraction cases, sandbox safety, PLR cross-check, leak checks
results/         smoke-cpu.json (local); results.json, figure.png, raw_outputs.jsonl (Colab run)
```

## 7. What I would do differently with more time

- Grade by **replaying the model's plan** through PyLabRobot's volume trackers (a third-party bookkeeping
  oracle) instead of matching plan parameters, so any correct construction passes for every family.
- Add a `maj@5` self-consistency arm over executed programs, and SFT + PoT together.
- Add a small real-text split from CC-BY protocol recipe tables, hand-checked, as a distribution-shift probe.
- Run a frontier reference arm to measure the ceiling instead of asserting it, and a GSM8K slice before and
  after the LoRA to show no general regression.
- Make infeasibility discoverable from the world alone (drop the `feasible` key) and grade the refusal as a
  free-text report.

## References

LAB-Bench (Laurent et al., 2024) and LABBench2 (2026); Aviary (Narayanan et al., 2024: tools + fine-tuning took
Llama-3.1-8B from 1 % to frontier level on SeqQA); PAL (Gao et al., 2023) and Program-of-Thoughts (Chen et
al., 2023); "To CoT or not to CoT" (Sprague et al., 2025); GSM-Symbolic (Mirzadeh et al., 2024); "SFT
Memorizes, RL Generalizes" (Chu et al., 2025); "Adding Error Bars to Evals" (Miller, 2024); PyLabRobot
(Wierenga et al., 2023).
