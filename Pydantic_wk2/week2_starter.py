"""Week 2 practice: async concurrent LLM calls -> Pydantic structured outputs.

Practice spec (learning schedule, Week 2, Oct 26-Nov 1):
  - send 5 prompts concurrently to the LLM API
  - parse every response into a Pydantic model (structured outputs)
  - log tokens used and cost per call
  - reuse the Week 1 pattern: asyncio + asyncio.gather + Semaphore

Setup:
    pip install anthropic pydantic
    export ANTHROPIC_API_KEY=...   # key in env ONLY -- never in code or in git
    python week2_starter.py

The 5 prompts are real meal logs from Oct 19-20. Swap in your own week of logs
for the stretch goal.
"""

import asyncio
from dotenv import load_dotenv
import os
import sys
import time

import anthropic
load_dotenv()
from pydantic import BaseModel, Field, ValidationError, field_validator


class MealLog(BaseModel):
    """Structured nutritional read of one raw meal-log entry."""

    meal_name: str = Field(description="Short name of the meal, e.g. 'perugu garelu'")
    calories: int = Field(ge=0, description="Estimated total calories")
    protein_g: float = Field(ge=0, description="Estimated protein in grams")
    fiber_g: float = Field(ge=0, default=0.0, description="Estimated fiber in grams")
    has_wheat: bool = Field(default=False, description="True if the meal contains wheat")
    is_heavy_meal: bool = Field(
        default=False, description="True if this is a heavy meal (calories >= 700)"
    )

    @field_validator("meal_name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("meal_name must not be empty")
        return v.strip()


# ---- cost model: verified 2026-10-23 against platform.claude.com/docs (Claude Sonnet 5.5) ----
PRICE_PER_MTOK_INPUT_USD = 2.00    # $2 / input MTok, verified 2026-10-23
PRICE_PER_MTOK_OUTPUT_USD = 10.00  # $10 / output MTok, verified 2026-10-23
MODEL = "claude-sonnet-5-5"  # current model ID, verified 2026-10-23 in the Anthropic docs quickstart

SYSTEM = (
    "You are a nutrition estimator. Given a raw meal-log entry, return "
    "your best estimates. Prefer round, defensible numbers over false precision. "
    "Flag wheat honestly -- the user is wheat intolerant."
)

MEALS = [
    "pumpkin ravioli with sage, brown butter, amaretti crumbles, plus a tall almond-milk latte (Locanda Verde)",
    "4 shallow-fried perugu garelu (urad dal vadas) in 3/4 cup plain Greek-yogurt perugu",
    "3 homemade onion dosas with peanut chutney",
    "5.3oz cup of Chobani non-fat plain Greek yogurt",
    "2 jowar rotis, 3/4 cup tamarind rice, 3/4 cup plain Greek yogurt, 3/4 cup cabbage and tomato curry",
]


async def structure_one(client: anthropic.AsyncAnthropic, sem: asyncio.Semaphore, raw: str):
    """One concurrent call: raw entry in, validated MealLog out."""
    async with sem:
        start = time.perf_counter()
        # Schema derived from the model itself: schema and model can never drift apart.
        resp = await client.messages.parse(
            model=MODEL,
            max_tokens=2000,  # room for adaptive thinking + the JSON answer
            system=SYSTEM,
            messages=[{"role": "user", "content": raw}],
            output_format=MealLog,  # structured output, validated into a MealLog
        )
        elapsed = time.perf_counter() - start

        meal = resp.parsed_output
        if meal is None:
            raise RuntimeError(f"no parsed output (stop_reason={resp.stop_reason}) for entry: {raw!r}")
        usage = resp.usage
        cost = (usage.input_tokens / 1e6) * PRICE_PER_MTOK_INPUT_USD + (
            usage.output_tokens / 1e6
        ) * PRICE_PER_MTOK_OUTPUT_USD
        return meal, usage, cost, elapsed


async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY in your environment first.")

    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(5)

    t0 = time.perf_counter()
    results = await asyncio.gather(
        *(structure_one(client, sem, m) for m in MEALS), return_exceptions=True
    )
    wall = time.perf_counter() - t0

    total_in = total_out = total_cost = 0
    ok = 0
    for raw, res in zip(MEALS, results):
        if isinstance(res, Exception):
            kind = "VALIDATION" if isinstance(res, ValidationError) else "CALL"
            print(f"[FAILED:{kind}] {raw[:60]}... -> {res}")
            continue
        meal, usage, cost, elapsed = res
        ok += 1
        total_in += usage.input_tokens
        total_out += usage.output_tokens
        total_cost += cost
        print(
            f"[OK {elapsed:4.1f}s] {meal.meal_name}: {meal.calories} cal, "
            f"{meal.protein_g:.0f}g protein, {meal.fiber_g:.0f}g fiber "
            f"(wheat={meal.has_wheat}, heavy={meal.is_heavy_meal}) | "
            f"{usage.input_tokens}+{usage.output_tokens} tok = ${cost:.4f}"
        )

    print(
        f"\n{ok}/{len(MEALS)} validated | wall {wall:.1f}s | "
        f"tokens in={total_in} out={total_out} | total cost ${total_cost:.4f}"
    )


if __name__ == "__main__":
    asyncio.run(main())
