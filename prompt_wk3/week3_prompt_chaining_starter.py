"""Week 3 practice, pattern 1: prompt chaining -- outline -> check -> write.

Practice spec (files/week3-practice-workflows.md, Week 3, Nov 2-8):
    raw_log -> [LLM: outline the day's meals + totals]
             -> [CODE gate: totals parse? calories >= 0? protein present?
                 any meal missing? totals == sum of meals?]
             -> [LLM: write the structured day summary, ONLY if the gate passed]

The gate is CODE, not another LLM call. If the outline fails the gate,
stop and name the failure -- a bad outline (e.g. a missing meal) never
reaches the writer. Same "failures are named, never silently dropped"
rule as Week 2.

Reuse Week 2 plumbing: asyncio, client.messages.parse with
output_format=<Pydantic model>, per-call token/cost logging.
Model and prices are the Week 2 values (claude-sonnet-5-5, $2/$10 per MTok).

Input is your own data: one raw day of meal logs. Swap RAW_LOG below for
a real day from this week -- one string per meal entry, in order.

Setup:
    pip install anthropic pydantic
    export ANTHROPIC_API_KEY=...   # key in env ONLY -- never in code or in git
    python week3_prompt_chaining_starter.py
    python week3_prompt_chaining_starter.py --test-gate   # no API call:
                                                          # proves the gate
                                                          # catches a bad outline

Note: this starter defines its own outline/summary models. It does NOT
copy MealLog from the workspace copy of week2_starter.py -- the GitHub
version of MealLog is the source of truth (it has iron_mg, and
is_heavy_meal as a @computed_field with HEAVY_MEAL_CALORIES = 700).
"""

import asyncio
from dotenv import load_dotenv
import os
import sys
import time

import anthropic
load_dotenv()

from pydantic import BaseModel, Field


# ---- cost model: Week 2 values, verified 2026-10-23 (Claude Sonnet 5.5) ----
PRICE_PER_MTOK_INPUT_USD = 2.00    # $2 / input MTok
PRICE_PER_MTOK_OUTPUT_USD = 10.00  # $10 / output MTok
MODEL = "claude-sonnet-5-5"


# ---- your data: one raw day, one string per meal entry, in order ----
RAW_LOG = [
    "2 eggs omelet with mozzarella, bell peppers, spinach and mushroom",
    "1 cup of almond milk with 1 scoop of whey protein",
    "Halloumi cheese salad with 1 cup of white beans 1 cucumber and 1 tomato",
    "Chai with 1 tsp sugar in almond milk.",
    "3 south indian dosas with 120g of peanut chutney"
]


class OutlineMeal(BaseModel):
    """One meal as extracted by the outline step."""

    meal_name: str = Field(description="Short name of the meal")
    calories: int = Field(ge=0, description="Estimated total calories")
    protein_g: float = Field(ge=0, description="Estimated protein in grams")


class DayOutline(BaseModel):
    """Step 1 output: the day's meals plus totals the LLM computed."""

    meals: list[OutlineMeal] = Field(
        min_length=1, description="One entry per meal in the raw log, in order"
    )
    total_calories: int = Field(ge=0, description="Sum of the meals' calories")
    total_protein_g: float = Field(ge=0, description="Sum of the meals' protein")


class DaySummary(BaseModel):
    """Step 3 output: the structured day summary, written from the outline only."""

    title: str = Field(description="Short title, e.g. 'Day summary'")
    meals: list[str] = Field(description="One short line per meal, with its numbers")
    total_calories: int = Field(ge=0)
    total_protein_g: float = Field(ge=0)
    notes: str = Field(description="1-2 sentences: what stood out about the day")


def cost_of(usage) -> float:
    """Cost of one call, in USD, from its token usage."""
    return (usage.input_tokens / 1e6) * PRICE_PER_MTOK_INPUT_USD + (
        usage.output_tokens / 1e6
    ) * PRICE_PER_MTOK_OUTPUT_USD


def gate_outline(outline: DayOutline, expected_meals: int) -> list[str]:
    """CODE gate, not an LLM call. Returns a list of failures (empty = pass).

    Checks: every meal present, calories >= 0, protein present (>= 0),
    and the outline totals equal the sum of the meals.
    """
    failures: list[str] = []

    if len(outline.meals) != expected_meals:
        failures.append(
            f"meal count mismatch: outline has {len(outline.meals)} meals, "
            f"raw log has {expected_meals} -- a meal is missing or invented"
        )
    for i, meal in enumerate(outline.meals):
        if meal.calories < 0:
            failures.append(f"meal {i} ({meal.meal_name!r}): calories < 0")
        if meal.protein_g < 0:
            failures.append(f"meal {i} ({meal.meal_name!r}): protein < 0")

    summed_cal = sum(m.calories for m in outline.meals)
    if outline.total_calories != summed_cal:
        failures.append(
            f"total calories mismatch: outline says {outline.total_calories}, "
            f"meals sum to {summed_cal}"
        )
    summed_protein = sum(m.protein_g for m in outline.meals)
    if abs(outline.total_protein_g - summed_protein) > 0.01:
        failures.append(
            f"total protein mismatch: outline says {outline.total_protein_g:.1f}g, "
            f"meals sum to {summed_protein:.1f}g"
        )
    return failures


async def outline_day(
    client: anthropic.AsyncAnthropic, raw_log: list[str]
) -> tuple[DayOutline, object, float, float]:
    """Step 1: LLM outlines the day's meals + totals."""
    start = time.perf_counter()
    log_text = "\n".join(f"- {entry}" for entry in raw_log)
    print(log_text)

    resp = await client.messages.parse(
        model=MODEL,
        max_tokens=800,
        system=(
            "You extract a structured outline from a raw day of meal logs. "
            "One outline entry per meal, in the same order as the log. "
            "Estimate calories and protein per meal, then compute the totals "
            "as the exact sum of the meals -- do not guess the totals."
        ),
        messages=[
            {
                "role": "user",
                "content": f"Outline this day of meal logs:\n{log_text}",
            }
        ],
        output_format=DayOutline,
    )
    elapsed = time.perf_counter() - start
    outline = resp.parsed_output
    if outline is None:
        raise RuntimeError("no parsed output returned for the outline step")
    return outline, resp.usage, cost_of(resp.usage), elapsed


async def write_summary(
    client: anthropic.AsyncAnthropic, outline: DayOutline
) -> tuple[DaySummary, object, float, float]:
    """Step 3: LLM writes the day summary -- from the checked outline ONLY."""
    start = time.perf_counter()
    outline_text = "\n".join(
        f"- {m.meal_name}: {m.calories} cal, {m.protein_g:.0f}g protein"
        for m in outline.meals
    )
    resp = await client.messages.parse(
        model=MODEL,
        max_tokens=800,
        system=(
            "You write a structured day summary from the outline you are given. "
            "Use only the meals and totals in the outline -- do not invent, "
            "drop, or re-estimate any meal or number."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    "Write the day summary for this checked outline:\n"
                    f"{outline_text}\n"
                    f"Totals: {outline.total_calories} cal, "
                    f"{outline.total_protein_g:.0f}g protein"
                ),
            }
        ],
        output_format=DaySummary,
    )
    elapsed = time.perf_counter() - start
    summary = resp.parsed_output
    if summary is None:
        raise RuntimeError("no parsed output returned for the write step")
    return summary, resp.usage, cost_of(resp.usage), elapsed


def test_gate() -> None:
    """Prove the gate works without spending an API call.

    Builds a deliberately bad outline -- one meal dropped, totals wrong --
    and shows the gate naming the failures before any writer runs.
    """
    bad = DayOutline(
        meals=[
            OutlineMeal(meal_name="omelet", calories=330, protein_g=38.0),
            OutlineMeal(meal_name="chickpea bowl", calories=520, protein_g=15.0),
            # yogurt snack dropped on purpose; dinner dropped on purpose
        ],
        total_calories=999,  # wrong on purpose: does not match the meals
        total_protein_g=53.0,
    )
    failures = gate_outline(bad, expected_meals=len(RAW_LOG))
    print("[GATE TEST] bad outline (2 of 4 meals, wrong totals):")
    for failure in failures:
        print(f"  - {failure}")
    assert failures, "gate should have caught the bad outline"
    print("[GATE TEST] gate caught it -- the writer would never see this outline.")


async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY in your environment first.")

    client = anthropic.AsyncAnthropic()
    total_cost = 0.0

    # Step 1: outline
    outline, usage, cost, elapsed = await outline_day(client, RAW_LOG)
    total_cost += cost
    print(f"[OUTLINE {elapsed:.1f}s] {len(outline.meals)} meals, "
          f"{outline.total_calories} cal, {outline.total_protein_g:.0f}g protein | "
          f"{usage.input_tokens}+{usage.output_tokens} tok = ${cost:.4f}")
    for meal in outline.meals:
        print(f"  - {meal.meal_name}: {meal.calories} cal, {meal.protein_g:.0f}g protein")

    # Step 2: CODE gate -- the writer never runs if this fails
    failures = gate_outline(outline, expected_meals=len(RAW_LOG))
    if failures:
        print("\n[GATE FAILED] outline rejected -- writer not called:")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    print("[GATE PASSED] outline totals check out -- proceeding to write")

    # Step 3: write, from the checked outline only
    summary, usage, cost, elapsed = await write_summary(client, outline)
    total_cost += cost
    print(f"\n[WRITE {elapsed:.1f}s] {summary.title} | "
          f"{usage.input_tokens}+{usage.output_tokens} tok = ${cost:.4f}")
    for line in summary.meals:
        print(f"  - {line}")
    print(f"  Totals: {summary.total_calories} cal, {summary.total_protein_g:.0f}g protein")
    print(f"  Notes: {summary.notes}")

    print(f"\nDone | total cost ${total_cost:.4f}")


if __name__ == "__main__":
    if "--test-gate" in sys.argv:
        test_gate()
    else:
        asyncio.run(main())
