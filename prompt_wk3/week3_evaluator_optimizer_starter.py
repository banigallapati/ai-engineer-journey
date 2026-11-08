"""Week 3 practice, pattern 3: evaluator-optimizer -- generator + critic loop.

Practice spec (files/week3-practice-workflows.md, Week 3, Nov 2-8):
    draft MealLog -> [LLM critic: check calories/protein against the raw text,
                      flag wheat honestly, flag any number that looks invented]
                   -> [LLM generator revises, or CODE stops the loop]
                   -> stop after 2 rounds max, or when the critic passes it

The hard stop is CODE, not the critic's goodwill. MAX_ROUNDS = 2 is enforced
in run_loop(): no matter what the critic says, the generator revises at
most twice, and then the loop ends -- with any remaining issues printed,
never silently dropped. An unbounded critic loop is the classic way a
simple workflow turns into an expensive agent.

Success check (from the spec): at least one draft where the critic catches
something real -- e.g. a wheat flag the generator missed, or a number that
looks invented -- AND the loop still terminates.

Reuse Week 2 plumbing: asyncio, client.messages.parse with
output_format=<Pydantic model>, per-call token/cost logging.
Model and prices are the Week 2 values (claude-sonnet-5-5, $2/$10 per MTok).

Input is your own data: one raw day of meal logs. Swap RAW_LOG below for
a real day from this week -- one string per meal entry, in order. The
default day is deliberately seeded with two traps the critic should catch:
the omelet the models keep pricing at 300 cal (actual ~190), and a CHOPT
bowl WITH pita chips -- wheat the generator may fail to flag. Note the
Nov 5 actual bowl swapped pita chips for white beans, so if you swap in
your real day the wheat flag must NOT fire: honest flagging cuts both ways.

Setup:
    pip install anthropic pydantic python-dotenv
    export ANTHROPIC_API_KEY=...   # key in env ONLY -- never in code or in git
    python week3_evaluator_optimizer_starter.py
    python week3_evaluator_optimizer_starter.py --test-loop  # no API call:
                                                             # proves the loop
                                                             # stops at 2
                                                             # revisions even
                                                             # when the critic
                                                             # never passes,
                                                             # and stops at 0
                                                             # when it passes
                                                             # at once

Note: this starter defines its own draft/review models. It does NOT copy
MealLog from the workspace copy of week2_starter.py -- the GitHub
version of MealLog is the source of truth (it has iron_mg, and
is_heavy_meal as a @computed_field with HEAVY_MEAL_CALORIES = 700).
"""

import asyncio
import os
import sys
import time
from typing import Awaitable, Callable

from dotenv import load_dotenv

import anthropic

load_dotenv()

from pydantic import BaseModel, Field


# ---- cost model: Week 2 values, verified 2026-10-23 (Claude Sonnet 5.5) ----
PRICE_PER_MTOK_INPUT_USD = 2.00    # $2 / input MTok
PRICE_PER_MTOK_OUTPUT_USD = 10.00  # $10 / output MTok
MODEL = "claude-sonnet-5-5"

# ---- the hard stop: at most this many generator revisions, enforced in code ----
MAX_ROUNDS = 2


# ---- your data: one raw day, one string per meal entry, in order ----
# Seeded with traps: the 300-cal omelet, and pita chips (wheat) in the
# CHOPT bowl. Swap in a real day from this week before you judge it.
RAW_LOG = [
    "2 eggs omelet with mozzarella, bell peppers, spinach and mushroom",
    "CHOPT Mediterranean Hummus Bowl: cauliflower rice base, falafel, "
    "white beans, pita chips, fiery feta dressing on the side",
    "7.7oz plain Greek yogurt with pomegranate arils",
    "3 south indian dosas with 120g of peanut chutney",
]


class DraftMeal(BaseModel):
    """One meal in the generator's draft."""

    meal_name: str = Field(description="Short name of the meal")
    calories: int = Field(ge=0, description="Estimated total calories")
    protein_g: float = Field(ge=0, description="Estimated protein in grams")
    contains_wheat: bool = Field(
        description="True ONLY if the entry actually contains wheat "
        "(bread, pita, chapati, pasta, semiya, etc.)"
    )
    basis: str = Field(description="One sentence: how this number was reached")


class DayDraft(BaseModel):
    """The generator's draft of the whole day."""

    meals: list[DraftMeal] = Field(min_length=1)
    total_calories: int = Field(ge=0, description="Sum of the meals' calories")
    total_protein_g: float = Field(ge=0, description="Sum of the meals' protein")


class CriticIssue(BaseModel):
    """One thing the critic found wrong, naming the meal it belongs to."""

    meal_index: int = Field(ge=0, description="Index into the raw log / draft")
    text: str = Field(description="What is wrong, concretely")


class CriticReview(BaseModel):
    """The critic's verdict on a draft."""

    passed: bool = Field(description="True only if nothing is wrong")
    issues: list[CriticIssue] = Field(
        default_factory=list,
        description="Empty when passed; one entry per problem when failed",
    )


def cost_of(usage) -> float:
    """Cost of one call, in USD, from its token usage."""
    return (usage.input_tokens / 1e6) * PRICE_PER_MTOK_INPUT_USD + (
        usage.output_tokens / 1e6
    ) * PRICE_PER_MTOK_OUTPUT_USD


# Token budgets: lesson from the routing starter -- max_tokens must fit the
# WHOLE structured output in one call. A 4-meal draft is ~1500 tokens; the
# routing baseline failed at 800 for exactly this reason.
GENERATE_MAX_TOKENS = 1500
CRITIQUE_MAX_TOKENS = 1000
REVISE_MAX_TOKENS = 1500


async def generate_draft(
    client: anthropic.AsyncAnthropic, raw_log: list[str]
) -> tuple[DayDraft, object, float, float]:
    """Round 0: the generator drafts the day from the raw log."""
    start = time.perf_counter()
    log_text = "\n".join(f"- {entry}" for entry in raw_log)
    resp = await client.messages.parse(
        model=MODEL,
        max_tokens=GENERATE_MAX_TOKENS,
        system=(
            "You estimate calories and protein for each meal in a day of "
            "meal logs, from the entries as written. Estimate from the "
            "stated portions -- do not assume a standard restaurant portion "
            "or a tablespoon of oil unless the entry says so. Flag wheat "
            "honestly: contains_wheat is true ONLY when the entry actually "
            "contains wheat (bread, pita, chapati, pasta, semiya). Totals "
            "must equal the sum of the meals."
        ),
        messages=[
            {
                "role": "user",
                "content": f"Draft the day's estimates:\n{log_text}",
            }
        ],
        output_format=DayDraft,
    )
    elapsed = time.perf_counter() - start
    draft = resp.parsed_output
    if draft is None:
        raise RuntimeError("no parsed output returned for the generate step")
    return draft, resp.usage, cost_of(resp.usage), elapsed


async def critique(
    client: anthropic.AsyncAnthropic, raw_log: list[str], draft: DayDraft
) -> tuple[CriticReview, object, float, float]:
    """The critic: check the draft against the raw text, nothing else."""
    start = time.perf_counter()
    log_text = "\n".join(f"{i}. {entry}" for i, entry in enumerate(raw_log))
    draft_text = "\n".join(
        f"{i}. {m.meal_name}: {m.calories} cal, {m.protein_g:.0f}g protein, "
        f"wheat={m.contains_wheat} ({m.basis})"
        for i, m in enumerate(draft.meals)
    )
    resp = await client.messages.parse(
        model=MODEL,
        max_tokens=CRITIQUE_MAX_TOKENS,
        system=(
            "You are a critic, not an estimator. Check the draft against "
            "the raw log entries, one meal at a time: is each calorie and "
            "protein number plausible for the portion actually stated? Is "
            "the wheat flag right in BOTH directions -- wheat missed, and "
            "wheat claimed where the entry has none? Does any number look "
            "invented or double-counted (e.g. dressing counted twice)? Do "
            "the totals equal the sum of the meals? Pass ONLY if nothing "
            "is wrong; otherwise list every problem, naming the meal index."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f"Raw log:\n{log_text}\n\nDraft to check:\n{draft_text}\n"
                    f"Draft totals: {draft.total_calories} cal, "
                    f"{draft.total_protein_g:.0f}g protein"
                ),
            }
        ],
        output_format=CriticReview,
    )
    elapsed = time.perf_counter() - start
    review = resp.parsed_output
    if review is None:
        raise RuntimeError("no parsed output returned for the critique step")
    return review, resp.usage, cost_of(resp.usage), elapsed


async def revise(
    client: anthropic.AsyncAnthropic,
    raw_log: list[str],
    draft: DayDraft,
    review: CriticReview,
) -> tuple[DayDraft, object, float, float]:
    """The generator revises the draft, addressing the critic's issues."""
    start = time.perf_counter()
    log_text = "\n".join(f"- {entry}" for entry in raw_log)
    draft_text = "\n".join(
        f"{i}. {m.meal_name}: {m.calories} cal, {m.protein_g:.0f}g protein, "
        f"wheat={m.contains_wheat} ({m.basis})"
        for i, m in enumerate(draft.meals)
    )
    issues_text = "\n".join(
        f"- meal {issue.meal_index}: {issue.text}" for issue in review.issues
    )
    resp = await client.messages.parse(
        model=MODEL,
        max_tokens=REVISE_MAX_TOKENS,
        system=(
            "You revise a draft day's estimates to fix the critic's issues. "
            "Change ONLY what the issues name -- do not re-estimate meals "
            "the critic passed, and do not invent new numbers. Totals must "
            "equal the sum of the revised meals."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f"Raw log:\n{log_text}\n\nCurrent draft:\n{draft_text}\n\n"
                    f"Critic's issues:\n{issues_text}\n\n"
                    "Return the revised draft."
                ),
            }
        ],
        output_format=DayDraft,
    )
    elapsed = time.perf_counter() - start
    revised = resp.parsed_output
    if revised is None:
        raise RuntimeError("no parsed output returned for the revise step")
    return revised, resp.usage, cost_of(resp.usage), elapsed


async def run_loop(
    generate: Callable[[list[str]], Awaitable[tuple[DayDraft, object, float, float]]],
    critique_fn: Callable[
        [list[str], DayDraft], Awaitable[tuple[CriticReview, object, float, float]]
    ],
    revise_fn: Callable[
        [list[str], DayDraft, CriticReview],
        Awaitable[tuple[DayDraft, object, float, float]],
    ],
    raw_log: list[str],
) -> tuple[DayDraft, CriticReview, int, float]:
    """The loop itself. Returns (final draft, final review, revisions, cost).

    generate / critique_fn / revise_fn are injected so the cap can be
    tested offline with fakes (see test_loop). The cap lives here, in code:
    `revisions >= MAX_ROUNDS` ends the loop however the critic votes.
    """
    total_cost = 0.0
    draft, _, cost, _ = await generate(raw_log)
    total_cost += cost
    print(f"[DRAFT] {draft.total_calories} cal, "
          f"{draft.total_protein_g:.0f}g protein")

    revisions = 0
    review: CriticReview | None = None
    while True:
        review, _, cost, _ = await critique_fn(raw_log, draft)
        total_cost += cost
        if review.passed:
            print(f"[CRITIC] passed -- no issues")
            break
        print(f"[CRITIC] failed ({len(review.issues)} issues):")
        for issue in review.issues:
            print(f"  - meal {issue.meal_index}: {issue.text}")
        if revisions >= MAX_ROUNDS:
            print(f"[STOP] hit the {MAX_ROUNDS}-revision cap in code -- "
                  f"ending with issues still open, not looping forever")
            break
        draft, _, cost, _ = await revise_fn(raw_log, draft, review)
        total_cost += cost
        revisions += 1
        print(f"[REVISE {revisions}/{MAX_ROUNDS}] "
              f"{draft.total_calories} cal, {draft.total_protein_g:.0f}g protein")

    assert review is not None
    return draft, review, revisions, total_cost


async def test_loop() -> None:
    """Prove the cap works without spending an API call.

    Three fake runs through the same run_loop() the live run uses:
    a critic that never passes (must stop at exactly 2 revisions), a
    critic that passes at once (0 revisions), and one that fails once
    then passes (1 revision).
    """
    draft = DayDraft(
        meals=[
            DraftMeal(meal_name="omelet", calories=300, protein_g=15.0,
                      contains_wheat=False, basis="standard portion"),
            DraftMeal(meal_name="chopt bowl", calories=720, protein_g=24.0,
                      contains_wheat=False, basis="menu estimate"),
        ],
        total_calories=1020,
        total_protein_g=39.0,
    )
    never_pass = CriticReview(passed=False, issues=[
        CriticIssue(meal_index=1, text="pita chips are wheat, flag missed"),
    ])
    passes = CriticReview(passed=True, issues=[])

    async def fake_generate(log):
        return draft, None, 0.0, 0.0

    calls = {"revise": 0}

    async def fake_revise(log, d, review):
        calls["revise"] += 1
        return draft, None, 0.0, 0.0

    async def fake_critique_never(log, d):
        return never_pass, None, 0.0, 0.0

    _, review, revisions, _ = await run_loop(
        fake_generate, fake_critique_never, fake_revise, RAW_LOG
    )
    assert revisions == MAX_ROUNDS == 2, f"cap failed: {revisions} revisions"
    assert not review.passed, "never-passing critic should end unpassed"
    print(f"[LOOP TEST] critic never passes -> stopped at {revisions} "
          f"revisions, loop terminated with issues open (not infinite)")

    async def fake_critique_pass(log, d):
        return passes, None, 0.0, 0.0

    calls["revise"] = 0
    _, review, revisions, _ = await run_loop(
        fake_generate, fake_critique_pass, fake_revise, RAW_LOG
    )
    assert revisions == 0 and review.passed
    print("[LOOP TEST] critic passes at once -> 0 revisions, loop ended")

    state = {"n": 0}

    async def fake_critique_once(log, d):
        state["n"] += 1
        return never_pass if state["n"] == 1 else passes, None, 0.0, 0.0

    _, review, revisions, _ = await run_loop(
        fake_generate, fake_critique_once, fake_revise, RAW_LOG
    )
    assert revisions == 1 and review.passed
    print("[LOOP TEST] critic fails once, then passes -> 1 revision, loop ended")


async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY in your environment first.")

    client = anthropic.AsyncAnthropic()

    draft, review, revisions, total_cost = await run_loop(
        lambda log: generate_draft(client, log),
        lambda log, d: critique(client, log, d),
        lambda log, d, r: revise(client, log, d, r),
        RAW_LOG,
    )

    print("\n--- final draft ---")
    for meal in draft.meals:
        wheat = "WHEAT" if meal.contains_wheat else "no wheat"
        print(f"  - {meal.meal_name}: {meal.calories} cal, "
              f"{meal.protein_g:.0f}g protein, {wheat}")
    print(f"  Totals: {draft.total_calories} cal, "
          f"{draft.total_protein_g:.0f}g protein")
    if review.passed:
        print(f"  Verdict: critic passed after {revisions} revision(s)")
    else:
        print(f"  Verdict: stopped at the {MAX_ROUNDS}-revision cap with "
              f"{len(review.issues)} issue(s) still open:")
        for issue in review.issues:
            print(f"  - meal {issue.meal_index}: {issue.text}")
        print("  Those issues are reported, not silently dropped -- judge "
              "whether the critic was right.")
    print(f"\nDone | total cost ${total_cost:.4f}")


if __name__ == "__main__":
    if "--test-loop" in sys.argv:
        asyncio.run(test_loop())
    else:
        asyncio.run(main())
