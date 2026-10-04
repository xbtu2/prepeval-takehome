"""Sandbox safety and the gold-through-sandbox control (tests the Program-of-Thought path end to end)."""

from __future__ import annotations

import json

import pytest

from prepeval import families as F
from prepeval import grading as G
from prepeval import sandbox as S

GOLD_PROGRAM = """```python
import json
result = {"stock_uL": 40.0, "diluent_uL": 60.0, "feasible": True}
print(json.dumps(result))
```"""


def test_runs_a_gold_program_and_grades_it():
  text, res = S.execute_completion(GOLD_PROGRAM)
  assert res is not None and res.ok, res
  obj = G.extract_json(text)
  assert obj["stock_ul"] == 40.0 and obj["feasible"] is True


def test_result_without_print_is_still_captured():
  text, res = S.execute_completion("```python\nresult = {'a': 1, 'feasible': False}\n```")
  assert res.ok and json.loads(text.strip())["a"] == 1


@pytest.mark.parametrize(
  "code",
  [
    "import os\nprint(os.listdir('.'))",
    "from subprocess import run\nrun(['ls'])",
    "print(open('/etc/passwd').read())",
    "print(().__class__.__bases__[0].__subclasses__())",
    "print(__import__('os').getcwd())",
    "print(eval('1+1'))",
    "import math\nprint(math.__loader__)",
  ],
)
def test_blocks_escape_attempts(code):
  res = S.run_code(code)
  assert not res.ok and res.status in ("rejected", "runtime_error"), res


def test_times_out_on_infinite_loop():
  res = S.run_code("while True:\n  pass")
  assert res.status == "timeout"


def test_times_out_or_fails_on_huge_power():
  res = S.run_code("x = 10**10**8\nprint(1)")
  assert not res.ok and res.status in ("timeout", "runtime_error")


def test_runtime_error_is_reported_not_raised():
  res = S.run_code("print(1/0)")
  assert res.status == "runtime_error" and "ZeroDivisionError" in res.stderr


def test_math_and_json_available_without_import():
  res = S.run_code("result = {'v': math.ceil(2.1)}\nprint(json.dumps(result))")
  assert res.ok and json.loads(res.stdout.strip().splitlines()[0])["v"] == 3


def test_gold_programs_for_every_family_pass_the_grader():
  """Control 3: a program that emits the gold must be accepted through sandbox + extractor + grader."""
  items = [F.make_item(f, s, "test") for f in F.FAMILIES for s in range(0, 40, 5)]
  for it in items:
    prog = "```python\nimport json\nresult = " + repr(it.gold) + "\nprint(json.dumps(result))\n```"
    text, res = S.execute_completion(prog)
    assert res.ok, (it.id, res)
    assert G.grade(it, text).accepted, it.id
