"""Week 4 practice: hand-rolled tool-calling loop -- LLM picks, CODE executes.

Practice spec (files/week4-practice-agents.md, Week 4, Nov 9-15):
    question -> [LLM: pick a tool + arguments, or answer directly]
             -> [CODE: validate the arguments, run the tool, return the
                 observation -- the LLM never runs a tool itself]
             -> [LLM: pick the next tool, or give the final answer]
    ... repeat, max 5 iterations, then CODE stops the loop.

Tools are plain Python functions, no framework. The three tools:
    calculator(expression)  -- safe arithmetic, no eval()
    web_search(query)       -- DuckDuckGo, top 3 results
    current_datetime()      -- now in America/New_York + UTC

Reuse Week 2/3 plumbing: asyncio, Anthropic client, per-call token/cost
logging. Model and prices are the Week 2 values (claude-sonnet-5-5,
$2/$10 per MTok). Unlike Weeks 2-3 this starter uses
client.messages.create (not .parse), because tool use returns
tool_use blocks, not a Pydantic model.

What Week 3 carries over -- and what it does NOT:
    - Chaining gate, carried over: tool arguments are validated in CODE
      before the tool runs (calculator expression must parse; a bad
      expression returns an error observation, it never crashes the loop).
    - Routing lesson, carried over: a tool cannot fix a wrong assumption
      the model holds. If the model searches for the wrong thing, the
      observation will be wrong too -- say so, don't add more tools.
    - Evaluator-optimizer lesson, carried over: do NOT add a critic LLM
      to check tool results. The code checks are the check -- the
      calculator result is computed by code, so it is trustworthy by
      construction; a same-model critic would share the generator's
      blind spots, exactly as in Week 3.

Setup:
    pip install anthropic ddgs python-dotenv
    # NOTE (Nov 12, 2026): the search package was renamed -- duckduckgo-search
    # is frozen at 8.1.1 and the maintained package is now `ddgs` (same
    # author, same `from ddgs import DDGS` API). Install `ddgs`, not
    # `duckduckgo-search`.
    export ANTHROPIC_API_KEY=...   # key in env ONLY -- never in code or in git
    python week4_tool_calling_starter.py
    python week4_tool_calling_starter.py --test-tools   # no API call:
                                                         # proves the three
                                                         # tools run locally

Default question needs two tools (calculator + datetime), so one run
exercises the loop, not just a single call. Swap QUESTION below for a
search question from the practice doc to exercise web_search.
"""

import asyncio
import ast
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

import anthropic


# ---- cost model: Week 2 values, verified 2026-10-23 (Claude Sonnet 5.5) ----
PRICE_PER_MTOK_INPUT_USD = 2.00    # $2 / input MTok
PRICE_PER_MTOK_OUTPUT_USD = 10.00  # $10 / output MTok
MODEL = "claude-sonnet-5-5"
MAX_ITERATIONS = 5  # CODE stops the loop here, no matter what the LLM wants


# ---- your question: swap this for one from the practice doc ----
QUESTION_0 = "What is 347 * 28, and what day of the week is it today in New York?"
QUESTION_1 = "What is the current Hugging Face AI Agents Course Unit 1 about — give me the section headings?"
QUESTION = "Search for the population of Paris, then double it, and tell me what time it is in New York right now."

# ---- tool 1: calculator (safe arithmetic -- never eval()) ----
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression. Only numbers and + - * / % ** ( )."""
    allowed_nodes = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
        ast.USub, ast.UAdd,
    )
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        return f"ERROR: could not parse expression {expression!r}: {e}"
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            return f"ERROR: {type(node).__name__} is not allowed in a calculation"
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            return "ERROR: only numbers are allowed in a calculation"

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp):
            value = _eval(node.operand)
            return -value if isinstance(node.op, ast.USub) else +value
        if isinstance(node, ast.BinOp):
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow):
                return left ** right
        raise ValueError(f"unsupported node {type(node).__name__}")

    try:
        return str(_eval(tree))
    except ZeroDivisionError:
        return "ERROR: division by zero"
    except Exception as e:  # noqa: BLE001 -- surfaced as an observation, not a crash
        return f"ERROR: {e}"


# ---- tool 2: web search (DuckDuckGo, top 3) ----
def web_search(query: str) -> str:
    """Search the web and return the top 3 results as title + snippet + URL."""
    try:
        from ddgs import DDGS
    except ImportError:
        return (
            "ERROR: ddgs is not installed -- "
            "run: pip install ddgs"
        )
    try:
        results = list(DDGS().text(query, max_results=3))
    except Exception as e:  # noqa: BLE001 -- surfaced as an observation
        return f"ERROR: search failed: {e}"
    if not results:
        return "No results found."
    lines = []
    for r in results:
        lines.append(f"- {r.get('title', '')}: {r.get('body', '')} ({r.get('href', '')})")
    return "\n".join(lines)


# ---- tool 3: current datetime ----
def current_datetime() -> str:
    """Current date and time in New York and in UTC."""
    now_ny = datetime.now(ZoneInfo("America/New_York"))
    now_utc = datetime.now(ZoneInfo("UTC"))
    return (
        f"New York: {now_ny.strftime('%A, %Y-%m-%d %H:%M %Z')} | "
        f"UTC: {now_utc.strftime('%Y-%m-%d %H:%M %Z')}"
    )


TOOLS = {
    "calculator": calculator,
    "web_search": web_search,
    "current_datetime": current_datetime,
}

TOOL_SCHEMAS = [
    {
        "name": "calculator",
        "description": (
            "Evaluate an arithmetic expression with + - * / % ** and "
            "parentheses. Use this for any calculation instead of doing "
            "arithmetic in your head."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "e.g. '347 * 28'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the web for current or factual information. Returns "
            "the top 3 results with title, snippet and URL."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "current_datetime",
        "description": (
            "Get the current date, day of the week and time in New York "
            "and in UTC. Use this for any 'today / now' question instead "
            "of guessing the date."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]

SYSTEM_PROMPT = (
    "You answer questions using the tools provided. When a question needs "
    "a calculation, a web search, or the current date/time, call the right "
    "tool instead of guessing. Use the tool's observation in your final "
    "answer and say which tool each number came from. If a tool returns an "
    "ERROR, say what went wrong and try a corrected call once -- do not "
    "invent the result the tool would have given."
)


def run_tool(name: str, arguments: dict) -> str:
    """CODE executes the tool. The LLM never runs a tool itself.

    Validates arguments first: unknown tool or missing argument returns
    an error observation instead of crashing the loop.
    """
    func = TOOLS.get(name)
    if func is None:
        return f"ERROR: unknown tool {name!r}"
    try:
        if name == "calculator":
            if "expression" not in arguments:
                return "ERROR: calculator needs an 'expression' argument"
            return func(arguments["expression"])
        if name == "web_search":
            if "query" not in arguments:
                return "ERROR: web_search needs a 'query' argument"
            return func(arguments["query"])
        if name == "current_datetime":
            return func()
    except Exception as e:  # noqa: BLE001 -- surfaced as an observation
        return f"ERROR: {e}"
    return f"ERROR: no handler for tool {name!r}"


def cost_of(usage) -> float:
    """Cost of one call, in USD, from its token usage."""
    return (usage.input_tokens / 1e6) * PRICE_PER_MTOK_INPUT_USD + (
        usage.output_tokens / 1e6
    ) * PRICE_PER_MTOK_OUTPUT_USD


def test_tools() -> None:
    """Prove the three tools run locally, without spending an API call."""
    print("[TOOLS TEST] calculator('347 * 28') =", calculator("347 * 28"))
    assert calculator("347 * 28") == "9716", "calculator gave the wrong answer"
    assert calculator("1 / 0").startswith("ERROR"), "division by zero must error"
    assert calculator("__import__('os')").startswith("ERROR"), "code must be rejected"
    print("[TOOLS TEST] calculator: 9716, division-by-zero errors, code rejected")
    print("[TOOLS TEST] current_datetime() =", current_datetime())
    print("[TOOLS TEST] web_search('Anthropic') =", web_search("Anthropic")[:200])
    print("[TOOLS TEST] all three tools ran -- the loop can use them.")


async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY in your environment first.")

    client = anthropic.AsyncAnthropic()
    messages = [{"role": "user", "content": QUESTION}]
    total_cost = 0.0
    transcript = [f"Q: {QUESTION}"]

    for iteration in range(1, MAX_ITERATIONS + 1):
        start = time.perf_counter()
        resp = await client.messages.create(
            model=MODEL,
            max_tokens=800,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        elapsed = time.perf_counter() - start
        total_cost += cost_of(resp.usage)
        print(
            f"[LLM {iteration} {elapsed:.1f}s] stop={resp.stop_reason} | "
            f"{resp.usage.input_tokens}+{resp.usage.output_tokens} tok = "
            f"${cost_of(resp.usage):.4f}"
        )

        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        text_parts = [b.text for b in resp.content if b.type == "text"]
        if text_parts:
            print("  LLM:", " ".join(text_parts))
            transcript.append(f"LLM: {' '.join(text_parts)}")

        if not tool_uses:
            # No tool call: this is the final answer. The loop ends here.
            print(f"\nFinal answer: {' '.join(text_parts)}")
            transcript.append(f"Final: {' '.join(text_parts)}")
            break

        # Append the assistant turn, then the tool results as a user turn.
        messages.append({"role": "assistant", "content": resp.content})
        tool_results = []
        for block in tool_uses:
            observation = run_tool(block.name, block.input or {})
            print(f"  [TOOL] {block.name}({block.input}) -> {observation[:160]}")
            transcript.append(f"TOOL {block.name}({block.input}) -> {observation}")
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": observation,
                }
            )
        messages.append({"role": "user", "content": tool_results})
    else:
        print(f"\n[STOPPED] hit the {MAX_ITERATIONS}-iteration cap -- code stopped "
              "the loop, the LLM did not. Final answer not reached.")

    print(f"\nDone | {len(transcript)} transcript lines | total cost ${total_cost:.4f}")
    print("Transcript:")
    for line in transcript:
        print(f"  {line}")


if __name__ == "__main__":
    if "--test-tools" in sys.argv:
        test_tools()
    else:
        asyncio.run(main())
