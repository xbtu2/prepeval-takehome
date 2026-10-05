# %% [markdown]
# # Liquid-handling plan reasoning: measure a 1.5B model, then improve it
#
# A liquid handler is a benchtop robot that pipettes liquid between containers and executes whatever numbers and
# grid positions it is given, with no check of its own. In ML terms the task here is structured prediction with
# an exact oracle and an abstention option: each item is a short natural-language request plus every constant
# needed to answer it; the target is a small JSON of numbers and grid coordinates the robot would execute
# verbatim; one item in five is unsatisfiable by construction and the only correct output is `feasible: false`.
# This notebook evaluates a small open model (`Qwen/Qwen2.5-1.5B-Instruct`) on 240 such items, tries two
# prompting interventions, then fine-tunes a LoRA on 1 431 generated worked solutions and measures what it
# learned and what it did not.
#
# | arm | what changes |
# |---|---|
# | `baseline` | zero-shot: intent + world facts + JSON schema; the model may reason freely and ends with JSON |
# | `fewshot`  | two worked examples from the training families as prior turns |
# | `pot`      | Program-of-Thought: the model writes Python that prints the JSON; a sandbox executes it |
# | `sft`      | LoRA fine-tune on generated items with reasoning traces (4 families); 2 families are held out |
#
# Grading is programmatic (no LLM judge). Controls run before any model call. Every acceptance carries its n and
# a 95 % bootstrap CI; arms are compared with an exact McNemar test on paired items. **The numbers in the text
# below are computed from this run**; a callout before the provenance section quotes the reference run so a
# Colab result can be compared.
#
# **How to run.** Colab: *Runtime → Change runtime type → T4 GPU*, then *Run all* (estimated 35 min; not yet
# timed on a T4). The same file runs as a plain script (`python notebook/prep_eval_colab.py`);
# `PREPEVAL_SMOKE=1` runs a tiny version of every cell with a 0.5B model.
#
# **Reading it in 12 minutes.** (0) The primer below if you have never seen a well plate. (1) The family table
# in section 4, then one item with its gold and worked trace in 4.3. (2) The controls and the trivial-classifier
# floor, 4.4. (3) The few-shot and PoT readings, section 5. (4) The results table and Figure 4, section 8.
# (5) Figure 6, in-train versus held-out, section 8. (6) Future directions, section 9. Everything else is the
# evidence behind those stops.

# %% [markdown]
# ### For readers without a lab background
#
# **Containers.** A *plate* is a palm-sized tray with a fixed grid of small cups called *wells*: a 96-well plate
# is 8 rows (A to H) by 12 columns (1 to 12), about 360 µL per well; a 384-well plate is 16 by 24, about 70 µL;
# a *deep-well* plate has the 96 layout with about 2.4 mL wells. Wells are named like spreadsheet cells (A1 is top-left;
# `B2:D5` is the rectangle rows B to D by columns 2 to 5). A *trough* is one open basin that all pipette tips
# share. An *8-channel head* has eight tips in a column and serves a whole plate column in one move.
#
# **Liquids.** A *stock* is a concentrated solution; a *diluent* (water, or a named buffer: treat any buffer as
# water with a name) thins it; the result is a *working solution*. Diluting conserves the amount of substance,
# C1·V1 = C2·V2, so the stock volume is V_final × C_target / C_stock, the diluent is the remainder, and dilution
# can only lower a concentration. A *serial dilution* chains this into a geometric series by moving a fixed
# *carry* volume from each well into the next; an N-point M-fold series has N wells, each M times more dilute
# than the previous, and the carry out of the last well is *discarded* so every well ends at the same volume. A
# *master mix* is the shared ingredients of N identical reactions mixed once, scaled by N × (1 + excess).
# *Normalising* samples means bringing several samples to one common concentration (not z-scoring). The
# *pipette minimum* is the smallest volume the robot can transfer; *dead volume* is liquid a trough must hold
# that the tips cannot reach, so the *trough load* is what is aspirated plus the dead volume. Concentration
# units are molar (M, mM, µM, nM), mass per volume (mg/mL, µg/mL, ng/µL) or relative "X" (a 5X stock is five
# times the 1X working strength); no item ever converts between kinds, so only SI prefixes are needed.
#
# **No lab knowledge is needed to solve any item.** Every constant (concentrations, grid sizes, well capacities,
# the pipette minimum, policy percentages, dead volumes) and every naming or indexing convention is printed in
# the item. Reagent names are opaque labels. The capability tested is arithmetic, constraint checking and 2-D
# grid indexing from natural language. A *family* is one of the six task types. *Acceptance* is exact-match
# accuracy under tolerance over the whole output. A *trap* is an unsatisfiable item whose only correct answer is
# `feasible: false`; answering `feasible: false` is called *abstaining* here (the tables say *refusing*): *trap
# recall* is the share of traps abstained on, *false refusal* the share of feasible items abstained on.

# %% [markdown]
# ## 1. Why this capability matters
#
# Four primitive operations of lab automation (move a volume; build a geometric dilution series; bring samples
# to a common concentration; batch-mix shared ingredients for N reactions) all reduce to the same arithmetic
# and 2-D grid indexing: volumes, per-well concentrations, lists of grid cells, container choice. A protocol
# compiler, the software that turns an experiment's recipe into robot instructions, bottoms out in exactly
# these numbers. The consumer of the output is a machine: a 10× error is executed as readily as the right
# number and silently ruins every sample on the plate. So the metric is exact correctness of the whole output
# within tolerance; one wrong field fails the item.
#
# The task has a second half: the 20 % of items that are traps, where the stated constants contradict the
# request (a stock volume below the pipette minimum; a well that would overflow) and the model must abstain. A
# planner that complies with everything is as dangerous as one that refuses everything; section 4.4 turns that
# into a floor the arms must clear.
#
# Failures are legible in the raw outputs (a power-of-ten prefix slip, C_target/C_stock used where
# C_stock/C_target was needed, a rectangle of wells miscounted); the error class records which part failed
# (format, the feasibility flag, or a value). No public dataset of such calculations with numeric ground truth
# was found (the nearest are college-chemistry sets such as SciBench and ChemistryQA, and LAB-Bench ProtocolQA,
# which is multiple-choice troubleshooting), so the items are generated from seeds, leak-checked, and graded
# against a programmatic answer key.

# %% [markdown]
# ## 2. Why a small model
#
# A frontier model would be expected to clear these items on first contact (not measured here; section 9 lists
# it). The question here is whether a model small enough to live on the instrument can, and what it takes to
# get it there.
#
# 1. **Regulated edge deployment.** The planner runs on the PC attached to the robot, often with no network.
#    In regulated labs (GxP, the family of "good practice" quality regimes) every external dependency in the
#    execution loop must be formally validated and re-validated when it changes, so an API-served model is a
#    compliance burden as well as a latency cost. A 1.5B model in fp16 is about 3 GB.
# 2. **Latency and cost.** Hundreds of planning steps per run, times a fleet; each call has to be fast and
#    nearly free. Section 5 prints the measured seconds per item.
# 3. **Ownership.** Pinned, hashed, auditable weights you can fine-tune on your own procedures; no vendor drift
#    between validation and production. Section 7 trains and hashes a small adapter.
# 4. **Privacy and IP.** Protocols, reagents and sample metadata never leave the site.
#
# The price is capability. Sections 3 to 5 measure it; sections 6 to 8 test how much a short fine-tune buys back.

# %% [markdown]
# ## 3. What goes wrong with small models on this task
#
# Expectations the eval is built to count. Sections 5 and 8 report on five of them with counts; arithmetic and
# unit slips show up only in the raw outputs and the appendix, since the grader records which field failed, not
# why.
#
# - **Answering without working.** A bare JSON object instead of a trace: a guess with a format.
# - **Arithmetic and unit slips.** Inverted dilution factors, raw numerals divided across units, a factor of
#   1 000 lost between mg/mL and ng/µL.
# - **Abstention independent of the label.** If P(abstain | infeasible) equals P(abstain | feasible), the
#   model's refusals carry no information and it is dominated by the constant always-abstain predictor, which
#   scores the infeasible-class prior.
# - **Template over procedure.** A worked example's trace skeleton copied onto items where it is wrong; a
#   fine-tuned skeleton reproduced on task types never seen.
# - **Code that is not a program.** A JSON dict inside a Python fence with `true`/`false` spelled as JSON, or a
#   program cut off at the token budget.
# - **Format fragility.** Arithmetic inside JSON values, an unclosed bracket, a loop that runs to the cap.
#
# Per item the eval records the error class, the predicted `feasible` flag, the sandbox outcome and whether
# the token cap was hit, so each of these can be counted rather than guessed at.

# %% [markdown]
# ## 4. How the eval is built
#
# Each item states a task in a scientist's words. A `World:` block carries every number the task needs: stock
# concentrations, container geometry, pipette minimum, lab policies. The question asks for one JSON object
# with fixed keys. The prompt never names the check that makes an item infeasible; the only affordance is the
# `feasible` key in the schema.
#
# The grader holds the answer key: a programmatic gold per item, a kind-specific rule per field (volumes within
# max(0.05 µL, 1 %), concentrations within 1 %, counts exact, well lists exact as sets, container choice exact),
# and one error class per failure. An item is accepted only if every required field passes; an infeasible item
# is graded on the flag alone.
#
# Six task types ("families") × 40 parameter draws ("seeds"); every fifth draw is unsatisfiable, which fixes the
# trivial-classifier floor at 0.20 (always abstain) and about 0 (constant answer). Two task types
# (`normalize_samples`, `labware_fit`) receive zero training examples: "in-train" and "held-out" below are the
# in-distribution (same task type, new parameters) and out-of-distribution (unseen task type) splits. The
# held-out types share some primitives with the trained ones (C1·V1 = C2·V2 per sample; capacity comparisons)
# and add some no training trace shows (inequality constraints with a set-valued answer; ceiling division and a
# dead-volume offset), so transfer is a real test, not a formality.
#
# | family | the model must produce | the math | trap (20 % of items) |
# |---|---|---|---|
# | `dilute_stock` | stock and diluent µL for a target concentration and volume | one linear equation plus a unit-prefix conversion | target above stock; stock volume below the pipette minimum |
# | `serial_dilution` | carry volume, diluent per well, discard, every well's concentration | a geometric sequence under a fixed final volume, with a well-capacity check | a well would overflow |
# | `master_mix` | per-ingredient totals for N reactions with the stated excess, water as the remainder | scale a recipe vector by N(1 + ε); water = volume − ingredients | the recipe does not fit the reaction volume |
# | `normalize_samples` (held out) | per-sample sample and diluent volumes to a common concentration and minimum volume | one linear equation per sample under inequality constraints; set-valued answer | a sample already below the target |
# | `well_addressing` | expand `B2:D5`; well at column-major index k; the 8 wells of a column; copy a 96-well plate into one quadrant of a 384-well plate (every second row and column) | 2-D index conversions (rectangle, linear index, column slice, stride-2 sub-lattice) | off-grid addresses |
# | `labware_fit` (held out) | plate vs deep-well vs trough for a volume and pattern; plates needed; trough load (aspirated volume plus dead volume) | capacity comparisons, ceiling division, a dead-volume offset | nothing fits |
#
# Plate geometries (rows × columns, well capacity) and the indexing conventions come from PyLabRobot, the
# open-source library that drives these robots, at a pinned commit, so the grid facts are a third party's, not
# mine; the model never sees the library.

# %% [markdown]
# ### 4.1 Configuration

# %%
import os
import sys

SMOKE = os.environ.get("PREPEVAL_SMOKE", "0") == "1"  # tiny CPU-friendly run that exercises every code path
MODEL_ID = os.environ.get("PREPEVAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct" if SMOKE else "Qwen/Qwen2.5-1.5B-Instruct")
REPO_URL = "https://github.com/xbtu2/prepeval-takehome.git"
REPO_DIR = "prepeval-takehome"
N_PER_FAMILY = int(os.environ.get("PREPEVAL_N_PER_FAMILY", 4 if SMOKE else 40))  # 40 -> 240 test items
ARMS = ["baseline", "fewshot", "pot", "sft"]
BATCH_SIZE = 2 if SMOKE else 16  # 32 halves inference time on a T4 if the budget is tight
SFT_MAX_STEPS = int(os.environ.get("PREPEVAL_SFT_STEPS", 3)) if SMOKE else -1  # -1 = one epoch (179 steps at effective batch 8)
SFT_TRAIN_ITEMS = int(os.environ.get("PREPEVAL_SFT_ITEMS", 16)) if SMOKE else None  # None = all
MAX_NEW_TOKENS_CAP = int(os.environ["PREPEVAL_MAX_NEW"]) if os.environ.get("PREPEVAL_MAX_NEW") else (96 if SMOKE else None)  # None = the arm's own cap
SEED = 0
OUT_DIR = "results"
PROBE_N = 8 if SMOKE else 40

# %% [markdown]
# ### 4.2 Environment (Colab installs; local runs assume the repo venv)

# %%
import subprocess

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
  # Colab ships torch + transformers; never reinstall torch (it would replace the CUDA build).
  # Colab also preinstalls torchao 0.10, and peft >= 0.18 raises ImportError at get_peft_model() when a torchao
  # older than 0.16 is present; nothing here uses torchao, so remove it rather than upgrade it against Colab's torch.
  subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
  subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft>=0.14", "trl>=0.19", "datasets>=3.0", "accelerate>=1.0"], check=True)
  if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", "-q", REPO_URL, REPO_DIR], check=True)
  os.chdir(REPO_DIR)
elif not os.path.isdir("prepeval"):
  try:
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
  except NameError:  # interactive kernel: __file__ undefined; assume cwd is the repo root or notebook/
    if os.path.isdir(os.path.join("..", "prepeval")):
      os.chdir("..")
sys.path.insert(0, os.getcwd())
os.makedirs(OUT_DIR, exist_ok=True)

import gc
import hashlib
import json
import time
from collections import Counter, defaultdict

import numpy as np
import torch

from prepeval import dataset as D
from prepeval import families as F
from prepeval import grading as G
from prepeval import prompts as P
from prepeval import readings as RD
from prepeval import sandbox as S
from prepeval import stats as ST

torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
print(f"device={device} dtype={dtype} torch={torch.__version__} python={sys.version.split()[0]}")
if device == "cuda":
  print(device_name, f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")

# %%
# Display layer: rendered Markdown and inline figures in a kernel (Colab, Jupyter, %run); plain text and PNG files
# as a script. The backend decision has to happen before the first pyplot import, so prepeval.figures is imported here.
import io

try:
  from IPython import get_ipython

  _ip = get_ipython()
except ImportError:  # the plain-script venv has no IPython
  _ip = None
IN_KERNEL = _ip is not None and hasattr(_ip, "kernel")

import matplotlib

if not IN_KERNEL:
  matplotlib.use("Agg")  # headless; in a kernel leave the inline backend alone
import matplotlib.pyplot as plt

from prepeval import figures as FIG

if IN_KERNEL:
  from IPython.display import Image, Markdown, display


def show_md(text, code=False):
  """Rendered Markdown in a kernel, plain text in a script log; code=True keeps the raw Markdown copyable."""
  if IN_KERNEL:
    display(Markdown(f"```markdown\n{text}\n```" if code else text))
  else:
    print(text)


def out_name(stem):
  return os.path.join(OUT_DIR, ("smoke-" if SMOKE else "") + stem)


def show_fig(fig, path, width_px=None):
  """Save the figure to `path`, show the same PNG inline in a kernel, then close it (one render, no leak)."""
  if fig is None:
    show_md("_Figure skipped: nothing to draw for this run._")
    return None
  buf = io.BytesIO()
  fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor())
  with open(path, "wb") as fh:
    fh.write(buf.getvalue())
  if IN_KERNEL:
    display(Image(data=buf.getvalue(), width=width_px or int(fig.get_figwidth() * 80)))
  plt.close(fig)
  print("saved", path)
  return path


print("display:", "kernel (inline figures)" if IN_KERNEL else "script (figures saved to files)")

# %%
show_md("**Figure 1. The labware in one figure.** Left: a 96-well plate with a rectangular block, the column an 8-channel head "
        "serves, and one indexed well. Middle: a 5-point, 5-fold serial dilution with the numbers of a real test item. Right: one "
        "direct dilution and the pipette-minimum case that makes an item infeasible.")
show_fig(FIG.fig_plate_primer(), out_name("fig_primer.png"))

# %% [markdown]
# ### 4.3 The items
#
# The item files are committed and loaded, not regenerated, so the eval is pinned. `data/manifest.json` carries
# their SHA-256, the PyLabRobot commit and the build-time leak check (no test prompt in train, held-out families
# absent from train, no gold value printed in its own prompt except values the item gives by construction, no
# few-shot value in a test gold). Below: the composition of this run's test set and one item in full, with the
# generator's own worked solution (the "trace") that explains its gold.

# %%
manifest = json.load(open("data/manifest.json"))
test_all = D.load_split("test")
train_all = D.load_split("train")
fewshot = D.load_split("fewshot")


def first_n_per_family(items, n):
  seen = Counter()
  out = []
  for it in items:
    if seen[it.family] < n:
      out.append(it)
      seen[it.family] += 1
  return out


test = first_n_per_family(test_all, N_PER_FAMILY)
train = train_all if SFT_TRAIN_ITEMS is None else train_all[:: max(1, len(train_all) // SFT_TRAIN_ITEMS)][:SFT_TRAIN_ITEMS]
show_md(RD.render_items_line(test, train_all, manifest) + f" Held-out families (never in train): {', '.join(f'`{f}`' for f in F.HELDOUT_FAMILIES)}.")
print("\nExample item\n" + "-" * 80 + "\n" + test[0].prompt + "\n" + "-" * 80 + "\ngold: " + json.dumps(test[0].gold))
print("why:  " + test[0].trace)

# %% [markdown]
# ### 4.4 Grading and controls
#
# Before any model call the grader is checked against itself. Every gold must be accepted (the grader is
# self-consistent); every single-field corruption of a gold (+3 % beyond tolerance, ×1000 prefix slip, flipped
# flag, dropped or shifted well, other container) must be rejected (the grader discriminates); a program that
# prints the gold must pass through sandbox, extractor and grader (the tool path is lossless); two constant
# predictors set the floor; the build-time leak check rules out contamination.
#
# The strongest non-solving control is the **trivial-classifier floor** (`effective_floor` in the table below):
# the better of the two constant predictors, here always-abstain at the infeasible-class prior of 0.200. An arm
# below it has negative skill: its answers and abstentions are jointly worse than ignoring the input. This
# matters in section 5. Trap recall and false refusal in the results tables are computed over all items, with an
# unparseable output counted as not abstaining; the Fisher tests in the text use parsed outputs only.

# %%
controls = {}
controls["gold_accepted"] = sum(G.grade(it, G.gold_completion(it)).accepted for it in test) / len(test)
muts = [(it, n, t) for it in test for n, t in G.mutations(it).items()]
controls["mutations_rejected"] = sum(not G.grade(it, t).accepted for it, _, t in muts) / len(muts)
ok = 0
for it in test:
  prog = "```python\nimport json\nresult = " + repr(it.gold) + "\nprint(json.dumps(result))\n```"
  text, res = S.execute_completion(prog)
  ok += bool(res and res.ok and G.grade(it, text).accepted)
controls["gold_program_through_sandbox"] = ok / len(test)
controls["policy_always_feasible_zeros"] = sum(G.grade(it, G.policy_always_feasible_zeros(it)).accepted for it in test) / len(test)
controls["policy_always_infeasible"] = sum(G.grade(it, G.policy_always_infeasible(it)).accepted for it in test) / len(test)
controls["leak_check_pass"] = bool(manifest["leak_check"]["PASS"])
controls["effective_floor"] = max(controls["policy_always_feasible_zeros"], controls["policy_always_infeasible"])
exp = {"gold_accepted": "1.00", "mutations_rejected": "1.00", "gold_program_through_sandbox": "1.00",
       "policy_always_feasible_zeros": "0.00", "policy_always_infeasible": "= trap share", "leak_check_pass": "True",
       "effective_floor": "strongest non-solving control"}
rows = ["| control | value | expected |", "|---|---|---|"]
rows += [f"| {k} | {v if isinstance(v, bool) else f'{v:.3f}'} | {exp[k]} |" for k, v in controls.items()]
show_md("\n".join(rows))
print(f"({len(muts)} single-field corruptions tested)")
assert controls["gold_accepted"] == 1.0 and controls["mutations_rejected"] == 1.0 and controls["gold_program_through_sandbox"] == 1.0

# %% [markdown]
# ### 4.5 Model and decoding
#
# One `GenerationConfig` for every arm: greedy, no repetition penalty (the checkpoint's own generation config
# sets `repetition_penalty = 1.1`, which would penalise re-emitting digits and JSON keys), left padding, own
# system message, prompt tokens sliced off. Token budgets:
# 512 new tokens for baseline, few-shot and sft; 320 for PoT, which writes a short program. Whether an output
# hit its cap is recorded per item.

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "left"
assert tok.pad_token_id is not None and tok.pad_token_id != tok.eos_token_id


def load_base():
  try:
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
  except TypeError:  # transformers < 4.56
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype)
  return m.to(device).eval()


MAX_LEN = 1024  # SFT sequence cap; the longest training row is 782 tokens (checked here, before any GPU time is spent)
if "sft" in ARMS:
  train_lens = [len(tok(tok.apply_chat_template(P.messages_baseline(it), tokenize=False, add_generation_prompt=True) + P.sft_target(it))["input_ids"]) + 1 for it in train]
  assert max(train_lens) <= MAX_LEN, f"longest SFT row {max(train_lens)} tokens > MAX_LEN={MAX_LEN}"

model = load_base()
import copy

# A GenerationConfig passed to generate() replaces the model's own (which carries repetition_penalty=1.1
# and sampling defaults); greedy decoding with no repetition penalty is all we want.
GEN = GenerationConfig(do_sample=False, repetition_penalty=1.0, temperature=1.0, top_p=1.0, top_k=50,
                       pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
print(f"loaded {MODEL_ID}: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")


def render(messages, prefill=""):
  return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefill


@torch.no_grad()
def generate(model, messages_list, max_new_tokens, prefill="", stop_strings=None, batch_size=BATCH_SIZE):
  """Batched greedy decoding; items are processed in order of prompt length so padding stays small."""
  texts = [render(m, prefill) for m in messages_list]
  order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
  outs = [None] * len(texts)
  hit_cap = [False] * len(texts)
  generate.last_hit_cap = hit_cap
  for b in range(0, len(order), batch_size):
    idx = order[b : b + batch_size]
    enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True).to(device)
    cfg = copy.deepcopy(GEN)
    cfg.max_new_tokens = max_new_tokens
    kw = {"stop_strings": stop_strings, "tokenizer": tok} if stop_strings else {}
    gen = model.generate(**enc, generation_config=cfg, **kw)
    new = gen[:, enc["input_ids"].shape[1] :]
    for i, row in zip(idx, new):
      outs[i] = prefill + tok.decode(row, skip_special_tokens=True)
      hit_cap[i] = int((row != tok.pad_token_id).sum()) >= max_new_tokens
  return outs


def arm_cap(arm):
  spec = P.ARMS[arm]
  return spec["max_new_tokens"] if MAX_NEW_TOKENS_CAP is None else min(spec["max_new_tokens"], MAX_NEW_TOKENS_CAP)


def run_arm(arm, model, items, tag=None):
  spec = P.ARMS[arm]
  cap = arm_cap(arm)
  msgs = [spec["messages"](it, fewshot) for it in items]
  t0 = time.time()
  if spec["tool"] == "python":
    # prefill the opening fence so the model writes code immediately and the closing fence stops decoding
    raw = generate(model, msgs, cap, prefill="```python\n", stop_strings=["```"])
    raw = [r if r.rstrip().endswith("```") else r + "\n```" for r in raw]
  else:
    raw = generate(model, msgs, cap)
  hit_cap = list(generate.last_hit_cap)
  elapsed = time.time() - t0
  grades, records = {}, []
  for it, out in zip(items, raw):
    graded_text, run = (S.execute_completion(out) if spec["tool"] == "python" else (out, None))
    g = G.grade(it, graded_text)
    grades[it.id] = g
    records.append({"item_id": it.id, "family": it.family, "raw": out, "graded_text": graded_text if spec["tool"] else None,
                    "sandbox": (run.status if run else None), "sandbox_err": (run.stderr[:160] if run else None),
                    "accepted": g.accepted, "error_class": g.error_class, "hit_token_cap": bool(hit_cap[len(records)])})
  acc = np.mean([g.accepted for g in grades.values()])
  print(f"[{tag or arm}] n={len(items)} accepted={acc:.3f} time={elapsed:.0f}s "
        f"({elapsed / len(items):.2f}s/item) format_fail={np.mean([g.error_class == 'format' for g in grades.values()]):.2f} "
        f"hit_token_cap={np.mean(hit_cap):.2f}")
  return grades, records, elapsed


# %% [markdown]
# ### 4.6 Probe: is the eval in the model's working range?
#
# Arm `baseline` on the first few items. If this were > 0.7 the set would be too easy for the model and if it
# were ~0 with mostly `format` failures the problem would be the output channel, not the reasoning. (Sanity
# check only; the model is not swapped post hoc.)

# %%
probe_items = first_n_per_family(test, max(1, PROBE_N // len(F.FAMILIES)))
pg, pr, _ = run_arm("baseline", model, probe_items, tag="probe")
print("error classes:", dict(Counter(g.error_class for g in pg.values() if not g.accepted)))

# %% [markdown]
# ## 5. Eval results: the model as it ships, and two prompting levers
#
# Three arms on the pristine model, same items, same decoding. `baseline` is the direct question. `fewshot`
# prepends two solved items as prior turns: one single dilution and one rectangular-range expansion (`C2:F5`),
# both from trained task types and both feasible, so the demonstrations never show an abstention. `pot` asks
# for one Python block that computes the answer; a subprocess sandbox executes it and the text JSON is the
# fallback.
#
# Why each should help: worked examples give the model a trace to imitate; code moves arithmetic and well
# enumeration into the interpreter so the model only has to set up the formula (PAL and Program-of-Thought
# reported large gains on arithmetic word problems). The table and the paragraphs after it are computed from
# this run.

# %%
grades_by_arm, records_by_arm, timing = {}, {}, {}
for arm in [a for a in ARMS if a != "sft"]:
  grades_by_arm[arm], records_by_arm[arm], timing[arm] = run_arm(arm, model, test)
if "pot" in records_by_arm:
  print("sandbox outcomes (pot):", dict(Counter(r["sandbox"] for r in records_by_arm["pot"])))

# %%
caps = {a: arm_cap(a) for a in ARMS}
summary_prompt = ST.summarize(test, grades_by_arm, baseline="baseline")
R5 = RD.derive_readings(test, grades_by_arm, records_by_arm, summary_prompt, fewshot=fewshot, caps=caps)
show_md(ST.results_table(summary_prompt))
show_md(RD.render_prompting(summary_prompt, R5, controls, timing, caps, BATCH_SIZE, device_name, smoke=SMOKE))

# %%
fams = list(summary_prompt["arms"]["baseline"]["per_family"])
show_md("**Figure 2. Where the prompting arms fail.** One row per family, one panel per arm; each item is one of five outcomes.")
show_fig(FIG.fig_outcomes(records_by_arm, test, arms=list(grades_by_arm), families=fams), out_name("fig_outcomes.png"))

# %% [markdown]
# *Reference run (RTX 4090, fp16, see the callout before the provenance section): baseline 0.092, fewshot 0.154,
# pot 0.062, all under the 0.200 floor.*

# %% [markdown]
# ## 6. Training data
#
# The training rows come from the same generators as the test items, with disjoint seeds (1000 to 1374) and
# only the four training families. `normalize_samples` and `labware_fit` never appear. Items whose prompt
# duplicated a test prompt were dropped at build time.
#
# Each row is a prompt and a completion. The prompt is the baseline arm's chat-formatted prompt. The
# completion is a programmatic reasoning trace, a `Final answer:` line and the gold JSON. Traces are produced by
# the code that produces the gold, so they are correct by construction, and on infeasible rows they name the
# check that fails. One in five training rows is infeasible, the same share as the test set. Below: the
# composition and token lengths, and one feasible and one infeasible completion (the prompt half of each row
# is the baseline prompt shown in section 4.3). Reading them: "X" is the relative concentration unit from the
# primer, and C1·V1 = C2·V2 is conservation of the dissolved substance.

# %%
lora_prov, lens, log_history, rows = None, [], [], []
if "sft" in ARMS:
  # completion ends without EOS: TRL appends the EOS token itself for prompt-completion rows
  rows = [{"prompt": render(P.messages_baseline(it)), "completion": P.sft_target(it)} for it in train]
  lens = [len(tok(r["prompt"] + r["completion"])["input_ids"]) + 1 for r in rows]
  assert max(lens) <= MAX_LEN, f"longest training row {max(lens)} tokens > {MAX_LEN}: truncation would drop the JSON"
  show_md(RD.render_rows_line(train, lens, MAX_LEN))
  feas_row = next(r for it, r in zip(train, rows) if it.feasible)
  trap_row = next((r for it, r in zip(train, rows) if not it.feasible), None)
  print("A feasible training completion\n" + "-" * 80 + "\n" + feas_row["completion"][:600])
  if trap_row:
    print("\nAn infeasible training completion\n" + "-" * 80 + "\n" + trap_row["completion"][:600])

# %% [markdown]
# ## 7. Training design
#
# LoRA, r = 16, alpha = 32, dropout 0.05, on the attention and MLP projections (q, k, v, o, gate, up, down):
# about 18.5 M trainable parameters, 1.2 % of the model. Learning rate 2e-4 with a cosine schedule and 3 %
# warmup; effective batch 8 (2 per device × 4 accumulation); one epoch; sequences capped at 1 024 tokens; loss
# on the completion only; fp16 autocast with fp32 adapter weights.
#
# A fresh copy of the base model is loaded for training so the model that served section 5 is never mutated.
# After training the adapter is saved with its SHA-256 prefix, merged into the weights, and the merged model is
# evaluated with the baseline prompt and the baseline token budget. No prompt changes between the baseline and
# sft arms.
#
# Why this recipe: it fits a free T4 in a few minutes, the adapter is small enough to pin and audit (section 2,
# argument 3), and one epoch over 1 431 rows is enough to fit the training distribution; the loss curve below
# shows whether it plateaued before the epoch ended.

# %%
if "sft" in ARMS:
  from datasets import Dataset
  from peft import LoraConfig, PeftModel, get_peft_model
  from trl import SFTConfig, SFTTrainer

  ds = Dataset.from_list(rows)

  # a fresh base model for training: get_peft_model wraps in place, and the pristine model already served section 5
  del model
  gc.collect()
  if device == "cuda":
    torch.cuda.empty_cache()
  model = load_base()

  lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
  peft_model = get_peft_model(model, lora)
  peft_model.print_trainable_parameters()

  import dataclasses

  supported = {f.name for f in dataclasses.fields(SFTConfig)}
  per_device, accum = (1, 1) if SMOKE else (2, 4)  # effective batch 8; 2 x 4 keeps a 16 GB T4 comfortable
  n_steps = SFT_MAX_STEPS if SFT_MAX_STEPS > 0 else int(np.ceil(len(rows) / (per_device * accum)))
  cfg_kw = dict(output_dir=os.path.join(OUT_DIR, "smoke-sft_tmp" if SMOKE else "sft_tmp"), per_device_train_batch_size=per_device,
                gradient_accumulation_steps=accum, learning_rate=2e-4, num_train_epochs=1,
                max_steps=SFT_MAX_STEPS, lr_scheduler_type="cosine", warmup_steps=max(1, int(0.03 * n_steps)),
                logging_steps=1 if SMOKE else 10,
                fp16=(device == "cuda"), bf16=False, report_to="none", save_strategy="no", seed=SEED,
                gradient_checkpointing=False, max_length=MAX_LEN, packing=False, completion_only_loss=True,
                dataset_num_proc=1, remove_unused_columns=True)
  if "max_length" not in supported:  # older trl
    cfg_kw["max_seq_length"] = cfg_kw.pop("max_length")
  dropped = [k for k in cfg_kw if k not in supported]
  if dropped:
    print("SFTConfig keys not supported by this trl version (dropped):", dropped)
  cfg = SFTConfig(**{k: v for k, v in cfg_kw.items() if k in supported})
  trainer = SFTTrainer(model=peft_model, args=cfg, train_dataset=ds, processing_class=tok)
  t0 = time.time()
  train_out = trainer.train()
  timing["sft_train"] = time.time() - t0
  log_history = [dict(e) for e in trainer.state.log_history]
  losses = [(e["step"], e["loss"]) for e in log_history if "loss" in e and "step" in e]
  first_under = next((s for s, l in losses if l < 0.1), None)
  last_logged = losses[-1][1] if losses else None
  # train_out.training_loss is the Trainer's mean loss over the whole epoch, not the loss at the last step
  print(f"trained {train_out.global_step} steps in {timing['sft_train']:.0f}s; mean training loss {train_out.training_loss:.3f}"
        + (f" (last logged step loss {last_logged:.3f})" if last_logged is not None else ""))
  if not np.isfinite(train_out.training_loss):
    print("WARNING: non-finite training loss (fp16 overflow?) - the sft arm below is not trustworthy")
  lora_prov = {"r": 16, "alpha": 32, "targets": "attn+mlp", "train_items": len(rows), "steps": int(train_out.global_step),
               "mean_train_loss": float(train_out.training_loss), "last_logged_loss": last_logged, "lr": 2e-4,
               "effective_batch": per_device * accum, "max_length": MAX_LEN, "loss_points": len(losses), "first_step_under_0_1": first_under}

  adapter_dir = os.path.join(OUT_DIR, "smoke-lora_adapter" if SMOKE else "lora_adapter")
  trainer.model.save_pretrained(adapter_dir)
  lora_prov["adapter_sha256_16"] = hashlib.sha256(open(os.path.join(adapter_dir, "adapter_model.safetensors"), "rb").read()).hexdigest()[:16]
  del trainer
  gc.collect()
  if device == "cuda":
    torch.cuda.empty_cache()

  # Sanity: with the adapter disabled the wrapper should start its answers the way the baseline arm did.
  peft_model.eval()
  peft_model.config.use_cache = True
  check_items = test[:10]
  with peft_model.disable_adapter():
    off = generate(peft_model, [P.messages_baseline(it) for it in check_items], 48, batch_size=len(check_items))
  base_raw = {r["item_id"]: r["raw"] for r in records_by_arm["baseline"]}
  same = sum(base_raw[it.id].startswith(o.rstrip()[:40]) for it, o in zip(check_items, off))
  print(f"adapter disabled -> first tokens match the baseline arm on {same}/{len(off)} items (fp16 batch noise can cost one or two)")

  merged = peft_model.merge_and_unload().eval()
  merged.config.use_cache = True
  show_md(f"**Trained** {lora_prov['steps']} steps in {timing['sft_train']:.0f} s on {device_name}; mean training loss over the "
          f"epoch {RD.fmt_num(lora_prov['mean_train_loss'])}, last logged step loss {RD.fmt_num(last_logged)}; adapter "
          f"`{lora_prov['adapter_sha256_16']}` saved to `{adapter_dir}`; "
          + (f"the loss first fell under 0.1 at step {first_under}." if first_under else "the loss never fell under 0.1."))

# %%
show_md("**Figure 3. LoRA training loss** at the logged steps. A curve that is still falling at the end means one epoch was not enough.")
show_fig(FIG.fig_loss_curve(log_history), out_name("fig_loss_curve.png"))

# %% [markdown]
# ## 8. Training result
#
# The fine-tuned model is evaluated on the same items with the baseline prompt. This section holds the complete
# results table: all four arms, acceptance with 95 % bootstrap CI, in-train and held-out acceptance, trap recall
# (recall on the infeasible class), false-refusal rate (abstention rate on feasible items, the false-positive
# rate of the abstain decision), format-failure rate (unparseable output), the paired delta against the baseline
# and the exact McNemar p-value. Per-family cells (n = 40) are directional at about ±15 points; the n = 240
# aggregate carries the claim.
#
# Two questions decide what the fine-tune proved. In-train (in distribution): on the four trained task types
# with unseen parameter values, does it beat the 0.200 trivial-classifier floor? Held-out (out of distribution):
# on the two task types it never saw, does any skill transfer, or did it learn four trace skeletons?

# %%
if "sft" in ARMS:
  grades_by_arm["sft"], records_by_arm["sft"], timing["sft"] = run_arm("sft", merged, test)
  model = merged
summary = ST.summarize(test, grades_by_arm, baseline="baseline")
summary["controls"] = controls
summary["timing_s"] = timing
R = RD.derive_readings(test, grades_by_arm, records_by_arm, summary, fewshot=fewshot, caps=caps)
summary["readings"] = R
show_md(ST.results_table(summary))
show_md(ST.family_table(summary))
show_md(RD.floor_lines(summary, controls))
show_md(RD.render_sft(summary, R, controls, lora_prov, smoke=SMOKE))

# %%
show_md("**Figure 4. Acceptance by arm and by family.** Left: acceptance with 95 % bootstrap CI and the trivial-classifier floor "
        "(always abstain). Right: per family; the two held-out families are labelled (held-out).")
show_fig(FIG.fig_acceptance(summary, controls), out_name("figure.png"))

# %%
arm_b = "sft" if "sft" in grades_by_arm else ("fewshot" if "fewshot" in grades_by_arm else None)
if arm_b:
  show_md(f"**Figure 5. Paired item transitions baseline → {arm_b}.** Items the second arm fixes point right; items it breaks point "
          "left. These discordant pairs are what the McNemar test counts.")
  show_fig(FIG.fig_transitions(grades_by_arm, test, "baseline", arm_b, families=fams, comparison=summary["comparisons"].get(arm_b)),
           out_name(f"fig_transitions_{arm_b}.png"))

# %%
show_md("**Figure 6. What the acceptances are.** In-train and held-out acceptance per arm, split into solved feasible items and "
        "refused traps. Transfer to a new family would show up as a solved-feasible segment in a held-out bar.")
show_fig(FIG.fig_heldout_split(summary, R), out_name("fig_heldout_split.png"))

# %%
show_md("**Figure 7. Does the `feasible` flag carry information?** Abstention rate on feasible items (x) against abstention rate "
        "on traps (y), all four arms. A flag that tracks infeasibility sits above the diagonal; a flag that abstains at random sits on it.")
show_fig(FIG.fig_refusals(summary, R), out_name("fig_refusals.png"))

# %% [markdown]
# ### Reference run
#
# > **Reference run.** This notebook's code and data, RTX 4090, fp16, greedy, n = 240; timed phases 5 min 29 s
# > (inference 3 min 59 s over the four arms, LoRA training 90 s), 6 min 20 s end to end on the shell clock
# > including model loading and controls; artefacts committed under `results/reference/`.
# > baseline 0.092 [0.058, 0.129]; fewshot 0.154 (+0.063, McNemar p = 0.017); pot 0.062 (−0.029, p = 0.23);
# > sft 0.504 [0.442, 0.567] (+0.412, p = 1.1e-22).
# > In-train / held-out acceptance: baseline 0.100 / 0.075, fewshot 0.200 / 0.062, pot 0.075 / 0.037,
# > sft 0.725 / 0.062. Trivial-classifier floor 0.200 (always-abstain policy).
# > Accepted per family, of 40 (baseline / fewshot / pot / sft): dilute_stock 3 / 26 / 4 / 30;
# > serial_dilution 4 / 0 / 3 / 32; master_mix 0 / 0 / 0 / 29; well_addressing 9 / 6 / 5 / 25;
# > normalize_samples (held out) 6 / 5 / 3 / 5; labware_fit (held out) 0 / 0 / 0 / 0.
# > Token-cap hits 1 / 4 / 30 / 0. LoRA: 179 steps, 90 s, mean training loss over the epoch 0.074, last logged
# > step loss 0.011, adapter `a24b7ef6d2bc32f2`.
# > On that run all 240 baseline outputs were bare JSON; no arm solved a feasible held-out item; 24 of 25
# > few-shot wins were in `dilute_stock`; all 24 rejected PoT programs were SyntaxErrors at the 320-token cap
# > and 38 of 42 runtime errors were `true`/`false` written as JSON.
# > A second run on the same GPU (the previous notebook version, same items and decoding) reproduced the three
# > prompting arms exactly: identical acceptance, per-family and error-class counts. LoRA training did not
# > reproduce: that fit scored 0.463 [0.400, 0.525] with the same mean training loss, so read the sft number as
# > one draw; the two fits differ by about 0.04. No Colab T4 run has been made; one should land inside the
# > intervals above.

# %% [markdown]
# ### Provenance and results file

# %%
import peft
import transformers

try:
  import trl

  trl_v = trl.__version__
except Exception:
  trl_v = None
provenance = {
  "model_id": MODEL_ID, "device": device, "dtype": str(dtype),
  "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
  "transformers": transformers.__version__, "peft": peft.__version__, "trl": trl_v, "torch": torch.__version__,
  "dataset_sha256": manifest.get("sha256"), "generator_version": manifest["generator_version"], "plr_commit": manifest["plr_commit"],
  "prompt_sha256": {"system": hashlib.sha256(P.SYSTEM.encode()).hexdigest()[:16], "pot_system": hashlib.sha256(P.POT_SYSTEM.encode()).hexdigest()[:16]},
  "decoding": {"greedy": True, "repetition_penalty": 1.0, "max_new_tokens": caps, "batch_size": BATCH_SIZE},
  "lora": lora_prov,
  "n_test_items": len(test), "smoke": SMOKE, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
out_path = os.path.join(OUT_DIR, "smoke-cpu.json" if SMOKE else "results.json")
ST.dump_results(out_path, summary, grades_by_arm, provenance)
with open(os.path.join(OUT_DIR, "raw_outputs.jsonl" if not SMOKE else "smoke_raw_outputs.jsonl"), "w") as fh:
  for arm, recs in records_by_arm.items():
    for r in recs:
      fh.write(json.dumps({"arm": arm, **r}) + "\n")
print("wrote", out_path)
show_md("**README-ready block** (copy into README section 4):")
show_md(ST.results_table(summary) + "\n\n" + ST.family_table(summary), code=True)

# %% [markdown]
# ## 9. Future directions
#
# In the order they would change a conclusion above:
#
# 1. A zero-shot chain-of-thought baseline ("work step by step, then the JSON") so the reference arm reasons;
#    on the reference run the baseline answered without working.
# 2. One token budget for every arm (at least 512) and `true`, `false`, `null` defined in the PoT runner, so no
#    arm is measured against its cap or its spelling.
# 3. One infeasible few-shot exemplar, and few-shot reported across several exemplars on one item set; the
#    current gain is confined to the exemplar's family and has not been separated from the item change.
# 4. A second LoRA seed, and a five-train / one-held-out rotation so transfer is tested per family rather than
#    on two fixed families.
# 5. Grade by simulating the plan's end state (per-well volumes and concentrations) in a third-party simulator
#    (PyLabRobot's volume trackers) instead of matching the plan's parameters, so any plan that reaches the right
#    end state passes: an outcome-based, set-valued oracle, which today only `normalize_samples` has.
# 6. A frontier reference arm to measure the ceiling instead of asserting it, and a GSM8K slice before and
#    after the LoRA to show no general regression.
# 7. A small real-text split from CC-BY protocol recipe tables, hand-checked, as a distribution-shift probe.
# 8. A maj@5 self-consistency arm over executed programs, and SFT combined with PoT, once item 2 has made PoT a
#    fair arm.
# 9. Infeasibility discoverable from the world alone (drop the `feasible` key) with the refusal graded as free
#    text.

# %% [markdown]
# ## Appendix: failure examples
#
# Reviewers trust examples more than CIs. One per error class per arm; the full outputs are in
# `results/raw_outputs.jsonl`. Error classes: `format` means no parseable JSON object (arithmetic left inside a
# value, an unclosed bracket, a truncated output); `feasibility` means the `feasible` flag is wrong (a trap
# answered, or a feasible item refused); `wrong_value` means the flag is right and at least one field is outside
# tolerance. The grader lower-cases keys before matching, which is why `pred` shows `stock_ul` against the
# gold's `stock_uL`.

# %%
for arm, recs in records_by_arm.items():
  by_class = defaultdict(list)
  for r in recs:
    if not r["accepted"]:
      by_class[r["error_class"]].append(r)
  print("=" * 100 + f"\n{arm}")
  for cls, rs in by_class.items():
    for r in rs[:1]:
      it = next(i for i in test if i.id == r["item_id"])
      g = grades_by_arm[arm][it.id]
      shown = (r["graded_text"] or r["raw"]) if arm == "pot" else r["raw"]
      print(f"--- [{cls}] {it.id}\n  gold: {json.dumps(it.gold)[:200]}\n  pred: {json.dumps(g.pred, default=str)[:200] if g.pred else None}")
      print("  model output (tail): " + shown.strip().replace("\n", " ⏎ ")[-300:])
