"""Program-of-Thought executor.

Model-written Python runs in a separate interpreter (`python -I`, isolated mode) with a 2 s wall-clock
timeout and a 512 MB address-space limit. Before execution an AST pass rejects anything but `import math` /
`import json`, every dunder name or attribute, and calls to eval/exec/open/compile/getattr/setattr/globals/
locals/vars/input/breakpoint/__import__. Execution uses a builtins whitelist. If the program defines
`result` but never printed it, the runner prints it, so a missing `print` is not a format failure.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass

TIMEOUT_S = 2.0
MEM_BYTES = 512 * 1024 * 1024

_FENCE = re.compile(r"```(?:python|py|json)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

RUNNER = r'''
import ast, sys, json, math
SAFE_MODULES = {"math", "json"}
BANNED_CALLS = {"eval", "exec", "open", "compile", "getattr", "setattr", "delattr", "globals", "locals", "vars",
                "input", "breakpoint", "__import__", "exit", "quit", "help", "memoryview", "classmethod", "staticmethod"}
SAFE_BUILTINS = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k) for k in [
  "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "frozenset", "int", "isinstance",
  "len", "list", "map", "max", "min", "pow", "print", "range", "reversed", "round", "set", "sorted", "str", "sum",
  "tuple", "zip", "chr", "ord", "repr", "format", "iter", "next", "ValueError", "TypeError", "ZeroDivisionError",
  "Exception", "KeyError", "IndexError", "True", "False", "None"] if (k in __builtins__ if isinstance(__builtins__, dict) else hasattr(__builtins__, k))}

def check(tree):
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      for a in node.names:
        if a.name.split(".")[0] not in SAFE_MODULES:
          raise ValueError(f"import of {a.name} is not allowed")
    elif isinstance(node, ast.ImportFrom):
      if (node.module or "").split(".")[0] not in SAFE_MODULES:
        raise ValueError(f"import from {node.module} is not allowed")
    elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
      raise ValueError("dunder attribute access is not allowed")
    elif isinstance(node, ast.Name) and node.id.startswith("__"):
      raise ValueError("dunder names are not allowed")
    elif isinstance(node, ast.Call):
      f = node.func
      name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
      if name in BANNED_CALLS:
        raise ValueError(f"call to {name} is not allowed")

src = sys.stdin.read()
try:
  tree = ast.parse(src)
  check(tree)
except Exception as e:
  print(f"__SANDBOX_REJECTED__ {type(e).__name__}: {e}", file=sys.stderr)
  sys.exit(3)
import builtins as _b
def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
  if level == 0 and name.split(".")[0] in SAFE_MODULES:
    return _b.__import__(name, globals, locals, fromlist, level)
  raise ImportError(f"import of {name} is not allowed")
SAFE_BUILTINS["__import__"] = _safe_import
ns = {"__builtins__": SAFE_BUILTINS, "math": math, "json": json}
try:
  exec(compile(tree, "<model>", "exec"), ns)
except Exception as e:
  print(f"__RUNTIME_ERROR__ {type(e).__name__}: {e}", file=sys.stderr)
  sys.exit(4)
if "result" in ns:
  try:
    print(json.dumps(ns["result"], default=str))
  except Exception as e:
    print(f"__RESULT_NOT_SERIALISABLE__ {e}", file=sys.stderr)
'''


@dataclass
class RunResult:
  ok: bool
  stdout: str
  stderr: str
  status: str  # ok | rejected | runtime_error | timeout | no_code


def extract_code(text: str) -> str | None:
  """First fenced code block; else the whole text if it looks like code; else None."""
  if not text:
    return None
  m = _FENCE.search(text)
  if m:
    return m.group(1)
  if "result" in text and ("=" in text or "print" in text):
    return text
  return None


def _limit_memory():  # pragma: no cover - runs in the child
  try:
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (MEM_BYTES, MEM_BYTES))
  except Exception:
    pass


def run_code(code: str, timeout: float = TIMEOUT_S) -> RunResult:
  try:
    proc = subprocess.run(
      [sys.executable, "-I", "-c", RUNNER],
      input=code,
      capture_output=True,
      text=True,
      timeout=timeout,
      preexec_fn=_limit_memory if sys.platform != "win32" else None,
    )
  except subprocess.TimeoutExpired:
    return RunResult(False, "", "timeout", "timeout")
  if proc.returncode == 3:
    return RunResult(False, proc.stdout, proc.stderr.strip(), "rejected")
  if proc.returncode == 4:
    return RunResult(False, proc.stdout, proc.stderr.strip(), "runtime_error")
  return RunResult(proc.returncode == 0, proc.stdout, proc.stderr.strip(), "ok" if proc.returncode == 0 else "runtime_error")


def execute_completion(text: str) -> tuple[str, RunResult | None]:
  """Run the model's code; return the text that should be graded (program stdout, or the raw text as a
  fallback when there is no runnable code) plus the run record."""
  code = extract_code(text)
  if code is None:
    return text, None
  res = run_code(code)
  if res.ok and res.stdout.strip():
    return res.stdout, res
  # fall back to any JSON the model wrote in prose
  return text, res
