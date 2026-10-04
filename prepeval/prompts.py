"""Arm prompts. Every arm sees the same item prompt (intent + world + question + schema); arms differ only in
the system message and in what surrounds the item.

  baseline   zero-shot; the model may reason as it likes and must finish with the JSON object
  fewshot    two worked examples (train families only) before the item
  pot        Program-of-Thought: the model writes one Python block that prints the JSON (executed by sandbox.py)
  sft        the fine-tuned model is evaluated with the *baseline* messages (no prompt change)
"""

from __future__ import annotations

from .families import Item, completion_text

SYSTEM = (
  "You are a laboratory automation assistant planning liquid-handling steps. Work through the arithmetic and "
  "the labware carefully, then finish with exactly one JSON object as requested."
)

POT_SYSTEM = (
  "You are a laboratory automation assistant planning liquid-handling steps. Solve the task by writing ONE "
  "Python code block (```python ... ```). Only the modules `math` and `json` may be imported. The code must "
  "compute the answer into a dict named `result` with exactly the requested keys and end with "
  "`print(json.dumps(result))`. Output only the code block."
)


def messages_baseline(item: Item) -> list[dict]:
  return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": item.prompt}]


def messages_fewshot(item: Item, examples: list[Item]) -> list[dict]:
  msgs = [{"role": "system", "content": SYSTEM}]
  for ex in examples:
    msgs.append({"role": "user", "content": ex.prompt})
    msgs.append({"role": "assistant", "content": completion_text(ex)})
  msgs.append({"role": "user", "content": item.prompt})
  return msgs


def messages_pot(item: Item) -> list[dict]:
  return [{"role": "system", "content": POT_SYSTEM}, {"role": "user", "content": item.prompt}]


ARMS = {
  "baseline": {"messages": lambda item, fewshot: messages_baseline(item), "max_new_tokens": 384, "tool": None},
  "fewshot": {"messages": lambda item, fewshot: messages_fewshot(item, fewshot), "max_new_tokens": 384, "tool": None},
  "pot": {"messages": lambda item, fewshot: messages_pot(item), "max_new_tokens": 320, "tool": "python"},
  "sft": {"messages": lambda item, fewshot: messages_baseline(item), "max_new_tokens": 384, "tool": None},
}


def sft_target(item: Item) -> str:
  """Assistant text the fine-tune learns: the programmatic trace followed by the gold JSON."""
  return completion_text(item)
