"""Week 3 practice, pattern 2: routing -- classifier -> specialist.

Practice spec (files/week3-practice-workflows.md, Week 3, Nov 2-8):
    raw_log -> [LLM classifier -> Route: "home_cooked" | "restaurant" | "snack_only"]
             -> [specialist prompt per route]
             -> compare against a single generic prompt (the baseline)

One design choice, flagged rather than hidden: the spec sketches one route
per log, but a real day mixes routes -- breakfast at home, a restaurant
bowl at lunch, a yogurt snack. So this starter routes EACH ENTRY: one
classifier call returns one route per entry, in order, and each entry is
then estimated by its own route's specialist. If you would rather route
a whole day to one route, the change is one line: classify the joined
day instead of the entries.

Why routing, not one prompt (the essay's point): optimizing one prompt
for restaurant meals hurts home-cooked estimates, and vice versa. The
classifier keeps them separate.

Success check (from the spec): the same log estimated two ways -- routed
vs. the single-prompt baseline -- and routing visibly beats the baseline.
If it doesn't beat it on your logs, say so. That IS the result, not a
failure.

Reuse Week 2 plumbing: asyncio, client.messages.parse with
output_format=<Pydantic model>, per-call token/cost logging.
Model and prices are the Week 2 values (claude-sonnet-5-5, $2/$10 per MTok).

Input is your own data: one raw day of meal logs. Swap RAW_LOG below for
a real day from this week -- one string per meal entry, in order. The
default day is deliberately mixed so one run exercises all three routes.

Setup:
    pip install anthropic pydantic python-dotenv
    export ANTHROPIC_API_KEY=...   # key in env ONLY -- never in code or in git
    python week3_routing_starter.py
    python week3_routing_starter.py --test-routes  # no API call: proves
                                                   # each route maps to a
                                                   # distinct specialist and
                                                   # the route-count check
                                                   # catches a bad classifier
                                                   # output

Note: this starter defines its own estimate models. It does NOT copy
MealLog from the workspace copy of week2_starter.py -- the GitHub
version of MealLog is the source of truth (it has iron_mg, and
is_heavy_meal as a @computed_field with HEAVY_MEAL_CALORIES = 700).
"""

import asyncio
import os
import sys
import time
from typing import Literal

from dotenv import load_dotenv

import anthropic

load_dotenv()

from pydantic import BaseModel, Field


# ---- cost model: Week 2 values, verified 2026-10-23 (Claude Sonnet 5.5) ----
PRICE_PER_MTOK_INPUT_USD = 2.00    # $2 / input MTok
PRICE_PER_MTOK_OUTPUT_USD = 10.00  # $10 / output MTok
MODEL = "claude-sonnet-5-5"

Route = Literal["home_cooked", "restaurant", "snack_only"]


# ---- your data: one raw day, one string per meal entry, in order ----
# A deliberately mixed day: home, restaurant, snack, home -- one run
# exercises all three routes. Swap in a real day before you judge it.
RAW_LOG = [
    "2 eggs omelet with mozzarella, bell peppers, spinach and mushroom",
    "CHOPT Mediterranean Hummus Bowl: cauliflower rice base, falafel, "
    "white beans, fiery feta dressing on the side",
    "7.7oz plain Greek yogurt with pomegranate arils",
    "3 south indian dosas with 120g of peanut chutney",
]


class RouteDecision(BaseModel):
    """One classifier decision: which route this entry takes, and why."""

    entry_index: int = Field(ge=0, description="Index into RAW_LOG, in order")
    route: Route = Field(description="home_cooked, restaurant, or snack_only")
    reason: str = Field(description="One short sentence: why this route")


class DayRoutes(BaseModel):
    """Step 1 output: one route decision per entry, in order."""

    routes: list[RouteDecision] = Field(
        min_length=1, description="One decision per entry in the raw log"
    )


class MealEstimate(BaseModel):
    """One meal's estimate, from a specialist (or the baseline)."""

    meal_name: str = Field(description="Short name of the meal")
    route: Route = Field(description="Which route estimated this meal")
    calories: int = Field(ge=0, description="Estimated total calories")
    protein_g: float = Field(ge=0, description="Estimated protein in grams")
    basis: str = Field(description="One sentence: how this number was reached")


class DayEstimate(BaseModel):
    """A whole day's estimates, routed or baseline."""

    meals: list[MealEstimate] = Field(min_length=1)
    total_calories: int = Field(ge=0, description="Sum of the meals' calories")
    total_protein_g: float = Field(ge=0, description="Sum of the meals' protein")


# ---- the specialists: one system prompt per route ----
# Kept in a plain dict (not an LLM call) so the mapping itself is testable
# offline -- see test_routes(). Each prompt says what to estimate FROM,
# which is the whole point of routing.
SPECIALIST_SYSTEM_PROMPTS: dict[Route, str] = {
    "home_cooked": (
        "You estimate a home-cooked meal from its ingredients. "
        "Work from the listed ingredients and their amounts -- a home "
        "portion is what was actually cooked, not a restaurant serving. "
        "Estimate calories and protein, then say in one sentence which "
        "ingredients drove the number."
    ),
    "restaurant": (
        "You estimate a restaurant meal from its menu-style description. "
        "Use the dish as served -- restaurant portions, dressings, and "
        "oils included, not a home-cooked equivalent. Estimate calories "
        "and protein, then say in one sentence which menu components "
        "drove the number."
    ),
    "snack_only": (
        "You estimate a snack with small-portion logic. This is a snack, "
        "not a full meal -- use the weighed or stated portion exactly, "
        "do not scale it up to a meal-sized serving. Estimate calories "
        "and protein, then say in one sentence how the portion drove "
        "the number."
    ),
}

BASELINE_SYSTEM_PROMPT = (
    "You estimate calories and protein for each meal in a day of meal logs. "
    "Use one generic approach for every entry, whatever kind of meal it is."
)


def specialist_prompt_for(route: Route) -> str:
    """Pure function: route -> its specialist prompt. No LLM call."""
    return SPECIALIST_SYSTEM_PROMPTS[route]


def cost_of(usage) -> float:
    """Cost of one call, in USD, from its token usage."""
    return (usage.input_tokens / 1e6) * PRICE_PER_MTOK_INPUT_USD + (
        usage.output_tokens / 1e6
    ) * PRICE_PER_MTOK_OUTPUT_USD


def check_routes(routes: DayRoutes, expected_entries: int) -> list[str]:
    """CODE check, not an LLM call. Returns failures (empty = pass).

    Checks: one decision per entry, in order, each route in the allowed set
    (Pydantic's Literal already enforces the set -- this checks the count
    and the ordering, which Pydantic cannot).
    """
    failures: list[str] = []
    if len(routes.routes) != expected_entries:
        failures.append(
            f"route count mismatch: classifier returned {len(routes.routes)} "
            f"decisions, raw log has {expected_entries} entries"
        )
    for i, decision in enumerate(routes.routes):
        if decision.entry_index != i:
            failures.append(
                f"route {i}: entry_index is {decision.entry_index}, "
                f"expected {i} -- decisions are out of order or misaligned"
            )
    return failures


async def classify(
    client: anthropic.AsyncAnthropic, raw_log: list[str]
) -> tuple[DayRoutes, object, float, float]:
    """Step 1: LLM classifier -- one route per entry, in order."""
    start = time.perf_counter()
    log_text = "\n".join(f"{i}. {entry}" for i, entry in enumerate(raw_log))
    resp = await client.messages.parse(
        model=MODEL,
        max_tokens=800,
        system=(
            "You classify each meal-log entry by where and how it was eaten. "
            "home_cooked: made at home from ingredients. restaurant: bought "
            "from a restaurant, cafe, or salad shop, described menu-style. "
            "snack_only: a small snack, not a full meal. Return exactly one "
            "decision per entry, in the same order, with a one-sentence reason."
        ),
        messages=[
            {
                "role": "user",
                "content": f"Classify each entry:\n{log_text}",
            }
        ],
        output_format=DayRoutes,
    )
    elapsed = time.perf_counter() - start
    routes = resp.parsed_output
    if routes is None:
        raise RuntimeError("no parsed output returned for the classify step")
    return routes, resp.usage, cost_of(resp.usage), elapsed


async def estimate_specialist(
    client: anthropic.AsyncAnthropic, entry: str, route: Route
) -> tuple[MealEstimate, object, float, float]:
    """Step 2: the entry's own specialist estimates it."""
    start = time.perf_counter()
    resp = await client.messages.parse(
        model=MODEL,
        max_tokens=600,
        system=specialist_prompt_for(route),
        messages=[
            {"role": "user", "content": f"Estimate this meal:\n{entry}"}
        ],
        output_format=MealEstimate,
    )
    elapsed = time.perf_counter() - start
    estimate = resp.parsed_output
    if estimate is None:
        raise RuntimeError("no parsed output returned for the specialist step")
    return estimate, resp.usage, cost_of(resp.usage), elapsed


async def estimate_baseline(
    client: anthropic.AsyncAnthropic, raw_log: list[str]
) -> tuple[DayEstimate, object, float, float]:
    """The comparison: one generic prompt estimates the whole day."""
    start = time.perf_counter()
    log_text = "\n".join(f"- {entry}" for entry in raw_log)
    # NOTE: max_tokens must fit the WHOLE structured output in one call.
    # The baseline returns all 4 meals (each with a basis sentence) plus
    # totals as a single JSON object -- unlike the specialists, which return
    # one meal each at 600 tokens. At 800 tokens the baseline JSON was cut
    # off mid-string, and client.messages.parse then failed with
    # "EOF while parsing a string": a truncated output, not a wrong one.
    # 1500 leaves headroom for 4 meals; the extra output costs ~$0.01.
    resp = await client.messages.parse(
        model=MODEL,
        max_tokens=1500,
        system=BASELINE_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Estimate this day of meal logs:\n{log_text}",
            }
        ],
        output_format=DayEstimate,
    )
    elapsed = time.perf_counter() - start
    estimate = resp.parsed_output
    if estimate is None:
        raise RuntimeError("no parsed output returned for the baseline step")
    return estimate, resp.usage, cost_of(resp.usage), elapsed


def test_routes() -> None:
    """Prove the routing table works without spending an API call.

    Checks: the three routes map to three distinct specialist prompts,
    a well-formed classifier output validates, and a short classifier
    output is caught by the count check before any specialist runs.
    """
    prompts = [specialist_prompt_for(r) for r in ("home_cooked", "restaurant", "snack_only")]
    assert len(set(prompts)) == 3, "each route needs its own specialist prompt"
    print("[ROUTES TEST] 3 routes -> 3 distinct specialist prompts")

    good = DayRoutes(
        routes=[
            RouteDecision(entry_index=0, route="home_cooked", reason="cooked at home"),
            RouteDecision(entry_index=1, route="restaurant", reason="bought at CHOPT"),
            RouteDecision(entry_index=2, route="snack_only", reason="small yogurt snack"),
            RouteDecision(entry_index=3, route="home_cooked", reason="cooked at home"),
        ]
    )
    assert not check_routes(good, expected_entries=4), "good routes should pass"
    print("[ROUTES TEST] well-formed classifier output (4 of 4) passes the check")

    bad = DayRoutes(
        routes=[
            RouteDecision(entry_index=0, route="home_cooked", reason="cooked at home"),
            RouteDecision(entry_index=1, route="restaurant", reason="bought at CHOPT"),
            # snack and dinner decisions dropped on purpose
        ]
    )
    failures = check_routes(bad, expected_entries=4)
    print("[ROUTES TEST] short classifier output (2 of 4 decisions):")
    for failure in failures:
        print(f"  - {failure}")
    assert failures, "count check should have caught the short output"
    print("[ROUTES TEST] count check caught it -- no specialist would run on this.")


async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY in your environment first.")

    client = anthropic.AsyncAnthropic()
    total_cost = 0.0

    # Step 1: classify -- one route per entry
    routes, usage, cost, elapsed = await classify(client, RAW_LOG)
    total_cost += cost
    print(f"[CLASSIFY {elapsed:.1f}s] | "
          f"{usage.input_tokens}+{usage.output_tokens} tok = ${cost:.4f}")
    for decision in routes.routes:
        print(f"  - [{decision.entry_index}] {decision.route}: "
              f"{RAW_LOG[decision.entry_index]} ({decision.reason})")

    # CODE check -- a misaligned classifier output never reaches a specialist
    failures = check_routes(routes, expected_entries=len(RAW_LOG))
    if failures:
        print("\n[ROUTES FAILED] classifier output rejected:")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    print("[ROUTES OK] one decision per entry, in order")

    # Step 2: each entry's specialist estimates it
    routed_meals: list[MealEstimate] = []
    for decision in routes.routes:
        entry = RAW_LOG[decision.entry_index]
        estimate, usage, cost, elapsed = await estimate_specialist(
            client, entry, decision.route
        )
        total_cost += cost
        routed_meals.append(estimate)
        print(f"[SPECIALIST {decision.route} {elapsed:.1f}s] {estimate.meal_name}: "
              f"{estimate.calories} cal, {estimate.protein_g:.0f}g protein | "
              f"{usage.input_tokens}+{usage.output_tokens} tok = ${cost:.4f}")
        print(f"    basis: {estimate.basis}")
    routed_cal = sum(m.calories for m in routed_meals)
    routed_protein = sum(m.protein_g for m in routed_meals)

    # Step 3: the single-prompt baseline, same log, for comparison
    baseline, usage, cost, elapsed = await estimate_baseline(client, RAW_LOG)
    total_cost += cost
    print(f"\n[BASELINE {elapsed:.1f}s] {baseline.total_calories} cal, "
          f"{baseline.total_protein_g:.0f}g protein | "
          f"{usage.input_tokens}+{usage.output_tokens} tok = ${cost:.4f}")
    for meal in baseline.meals:
        print(f"  - {meal.meal_name}: {meal.calories} cal, "
              f"{meal.protein_g:.0f}g protein")

    print("\n--- routed vs baseline (same log) ---")
    print(f"  routed:   {routed_cal} cal, {routed_protein:.0f}g protein")
    print(f"  baseline: {baseline.total_calories} cal, "
          f"{baseline.total_protein_g:.0f}g protein")
    print(f"  delta:    {routed_cal - baseline.total_calories:+d} cal, "
          f"{routed_protein - baseline.total_protein_g:+.0f}g protein")
    print("  Judge it yourself: did routing beat the single prompt on this "
          "day? If not, that is the result -- write it down, don't force it.")

    print(f"\nDone | total cost ${total_cost:.4f}")


if __name__ == "__main__":
    if "--test-routes" in sys.argv:
        test_routes()
    else:
        asyncio.run(main())
