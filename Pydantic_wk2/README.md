# Week 2: Concurrent LLM calls with Pydantic structured outputs

`week2_starter.py` sends five raw meal-log entries to Claude at the same time and gets each one back as a validated Pydantic `MealLog` object. For every call it prints the nutrition estimate, the tokens used and the cost.

It builds on the Week 1 AsyncIO tutorial (`asyncio.gather` + `Semaphore`) and adds structured outputs on top.

## What it does

1. Defines a `MealLog` Pydantic model: meal name, calories, protein, fiber, iron, whether it contains wheat, and whether it's a heavy meal (see [The `MealLog` model](#the-meallog-model)).
2. Sends all five entries in `MEALS` to `claude-sonnet-5-5` at once. An `asyncio.Semaphore(5)` caps how many run at the same time.
3. Calls `client.messages.parse(..., output_format=MealLog)`, so Claude's answer comes back as a `MealLog` that has already been checked against the model.
4. Works out the cost of each call from `resp.usage` and prints a summary line with totals.

Failures don't stop the run. `asyncio.gather(..., return_exceptions=True)` collects them, and each one is labelled `CALL` (the API request failed) or `VALIDATION` (the answer didn't fit the model).

## The `MealLog` model

Every field is an estimate made by Claude. The `description` on each field is sent to Claude as part of the schema, so it doubles as an instruction.

| Field | Type | Rules | Meaning |
|---|---|---|---|
| `meal_name` | `str` | Required, can't be blank (whitespace is trimmed) | Short name for the meal |
| `calories` | `int` | Required, `>= 0` | Estimated total calories |
| `protein_g` | `float` | Required, `>= 0` | Estimated protein in grams |
| `fiber_g` | `float` | `>= 0`, defaults to `0.0` | Estimated fiber in grams |
| `iron_mg` | `float` | `>= 0`, defaults to `0.0` | Estimated iron in milligrams |
| `has_wheat` | `bool` | Defaults to `False` | `True` if the meal contains wheat. The system prompt asks Claude to flag wheat honestly because of a wheat intolerance. |
| `is_heavy_meal` | `bool` | Computed from `calories`, not sent by Claude | `True` if the meal has 700 calories or more (threshold is configurable) |

### `iron_mg`

Added to track iron intake alongside protein and fiber. Because it's part of `MealLog`, adding the field was all it took: the schema sent to Claude picks it up automatically, and the script prints it with two decimal places.

### `is_heavy_meal`

A quick flag for meals of 700 calories or more. It's a Pydantic `@computed_field`, calculated from `calories` whenever you read it, so it always agrees with `calories`.

```python
class MealLog(BaseModel):
    HEAVY_MEAL_CALORIES: ClassVar[int] = 700
    ...

    @computed_field(description=f"True if calories >= {HEAVY_MEAL_CALORIES} (computed, not estimated)")
    @property
    def is_heavy_meal(self) -> bool:
        return self.calories >= self.HEAVY_MEAL_CALORIES
```

- **Claude never sees it.** Computed fields aren't inputs, so they're left out of the schema `messages.parse` sends. They do appear in `model_dump()` and in the output schema (`model_json_schema(mode="serialization")`), where the description shows up.
- **The threshold is a class constant.** `ClassVar` tells Pydantic it's not a field. To use a different cut-off, subclass and override it:

  ```python
  class StrictMealLog(MealLog):
      HEAVY_MEAL_CALORIES: ClassVar[int] = 500
  ```

  The description text is filled in when `MealLog` is defined, so a subclass's description still says 700.

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

Example output (Oct 25, 2026 run, shortened):

```
[OK  4.3s] jowar rotis with beerakaya dal and Greek yogurt: 490 cal, 31g protein, 9g fiber 4.50mg iron (wheat=False, heavy=False) | 790+386 tok = $0.0054
[OK  1.8s] almond milk protein shake: 160 cal, 27g protein, 1g fiber 1.00mg iron (wheat=False, heavy=False) | 775+71 tok = $0.0023
...
5/5 validated | wall 6.3s | tokens in=3881 out=1404 | total cost $0.0218
```

The total time is close to the slowest single call, not the sum of all five. That's the benefit of running them concurrently.

