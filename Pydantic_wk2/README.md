# Week 2: Concurrent LLM calls with Pydantic structured outputs

`week2_starter.py` sends five raw meal-log entries to Claude at the same time and gets each one back as a validated Pydantic `MealLog` object. For every call it prints the nutrition estimate, the tokens used and the cost.

It builds on the Week 1 AsyncIO tutorial (`asyncio.gather` + `Semaphore`) and adds structured outputs on top.

## What it does

1. Defines a `MealLog` Pydantic model: meal name, calories, protein, fiber, whether it contains wheat, and whether it's a heavy meal.
2. Sends all five entries in `MEALS` to `claude-sonnet-5-5` at once. An `asyncio.Semaphore(5)` caps how many run at the same time.
3. Calls `client.messages.parse(..., output_format=MealLog)`, so Claude's answer comes back as a `MealLog` that has already been checked against the model.
4. Works out the cost of each call from `resp.usage` and prints a summary line with totals.

Failures don't stop the run. `asyncio.gather(..., return_exceptions=True)` collects them, and each one is labelled `CALL` (the API request failed) or `VALIDATION` (the answer didn't fit the model).

## Setup

Requires Python 3.10+ (developed on 3.14 in the `ai-engineer-journey` conda environment).

```bash
conda activate ai-engineer-journey
python -m pip install anthropic pydantic python-dotenv
```

Put your API key in a `.env` file at the repo root:


## Run

```bash
python Pydantic_wk2/week2_starter.py
```

Example output (Oct 23, 2026 run, shortened):

```
[OK  1.9s] pumpkin ravioli ... almond-milk latte: 850 cal, 22g protein, 5g fiber (wheat=True, heavy=True) | 729+87 tok = $0.0023
[OK  2.0s] Chobani non-fat plain Greek yogurt (5.3 oz cup): 80 cal, 15g protein, 0g fiber (wheat=False, heavy=False) | 706+81 tok = $0.0022
...
5/5 validated | wall 5.6s | tokens in=3596 out=1345 | total cost $0.0206
```

The total time is close to the slowest single call, not the sum of all five. That's the benefit of running them concurrently.

