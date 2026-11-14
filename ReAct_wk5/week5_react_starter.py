"""Week 5 practice: hand-rolled ReAct loop -- Thought -> Action -> Observation.

Practice spec (files/week5-practice-react.md, Week 5, Nov 16-22, started
early Nov 13): extend the Week 4 starter, don't rewrite it. The three tools,
the code-gate on arguments, and the per-call token/cost logging carry over
unchanged. The one structural change is the THOUGHT: at every step the model
first says what it knows, what it still needs, and what it will do next --
then it acts (a tool call) or finishes (a direct answer).

How Thought is implemented here, per the spec: the model's text response
before a tool call IS the Thought -- no separate Thought:/Action: parser.
Anthropic tool use already gives us text blocks (Thought) + tool_use blocks
(Action), and the tool result is the Observation. A hand-rolled parser would
be a new thing to debug for no gain.

    question -> [LLM: Thought + Action (tool call) or Finish (direct answer)]
             -> [CODE: validate the arguments, run the tool, return the
                 observation -- the LLM never runs a tool itself]
             -> [LLM: Thought (what the observation changes) + next Action,
                 or Finish]
    ... repeat, max 8 iterations, then CODE stops the loop.

    Every step is written to a transcript file (react_qN_transcript.txt)
    as it happens, so a wrong run can be read back step by step: was it
    the Thought, the Action, or the Observation that went wrong?

Tools are plain Python functions, no framework. The three core tools are
the Week 4 three:
    calculator(expression)  -- safe arithmetic, no eval()
    web_search(query)       -- DuckDuckGo (ddgs), top 3 results
    current_datetime()      -- now in America/New_York + UTC
Stretch (optional, --fetch): a fourth tool, fetch_page(url), for the
Week 4 Q2 lesson -- search snippets don't carry page content, so a
question that needs what's ON a page needs a fetch tool, not more
search queries.

Reuse Week 2/3/4 plumbing: asyncio, Anthropic client, per-call token/cost
logging. Model and prices are the Week 2 values (claude-sonnet-5-5,
$2/$10 per MTok).

What Week 3/4 carry over -- and what they do NOT:
    - Chaining gate, carried over: tool arguments are validated in CODE
      before the tool runs. A bad expression returns an error observation,
      it never crashes the loop.
    - Routing lesson, carried over: a tool cannot fix a wrong assumption
      the model holds. In ReAct terms: a wrong Thought produces a wrong
      Action, and the transcript will show you exactly where.
    - Evaluator-optimizer lesson, carried over: do NOT add a critic LLM
      to check tool results. The code checks are the check.
    - Week 4 Q2 lesson, carried over: search != fetch. See --fetch.
    - Week 4 note, carried over (his principle): static knowledge the
      model can answer directly; dynamic knowledge needs the agent to
      interact with the environment -- tools are a must. The system
      prompt below says exactly that: if the answer is static, answer
      directly with no tool call at all.

Setup:
    pip install anthropic ddgs python-dotenv
    export ANTHROPIC_API_KEY=...   # key in env ONLY -- never in code or in git
    python week5_react_starter.py --test-tools   # no API call: proves the
                                                 # tools run locally
    python week5_react_starter.py --question 1   # run one question
    python week5_react_starter.py                # run all five questions
    python week5_react_starter.py --question 2 --fetch  # Q2 with fetch_page
"""

import argparse
import asyncio
import ast
import os
import re
import sys
import time
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

import anthropic


# ---- cost model: Week 2 values, verified 2026-10-23 (Claude Sonnet 5.5) ----
PRICE_PER_MTOK_INPUT_USD = 2.00    # $2 / input MTok
PRICE_PER_MTOK_OUTPUT_USD = 10.00  # $10 / output MTok
MODEL = "claude-sonnet-5-5"
MAX_ITERATIONS = 8  # CODE stops the loop here, no matter what the LLM wants


# ---- the five test questions from the practice spec ----
QUESTIONS = {
    1: "What was the population of the city where the 2024 Olympics were held, doubled?",
    2: "What is the population of Tokyo, and what is it divided by the population of Paris?",
    3: "How many days are there between today and Christmas Day 2026, and what is that number times 7?",
    4: "In what year was the Eiffel Tower completed, and how many years ago was that?",
    5: "Search for the current population of Mumbai, then tell me whether it is more than double the population of Paris, and by how much.",
}


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


# ---- stretch tool 4: fetch a page (only with --fetch) ----
def fetch_page(url: str) -> str:
    """Fetch a URL and return its text, tags stripped, first ~3000 chars.

    Week 4 Q2 proved search snippets don't carry page content -- a
    question that needs what's ON a page needs this tool, not more
    search queries. Errors come back as observations, never crashes.
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
    except Exception as e:  # noqa: BLE001 -- surfaced as an observation
        return f"ERROR: could not fetch {url}: {e}"
    text = re.sub(r"<script.*?</script>", " ", raw, flags=re.S | re.I)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "No readable text on this page."
    return text[:3000]


TOOLS = {
    "calculator": calculator,
    "web_search": web_search,
    "current_datetime": current_datetime,
    "fetch_page": fetch_page,
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
            "the top 3 results with title, snippet and URL. Snippets do "
            "NOT carry a page's full content -- if you need what is ON a "
            "page, use fetch_page instead."
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

FETCH_PAGE_SCHEMA = {
    "name": "fetch_page",
    "description": (
        "Fetch a web page and return its readable text (first ~3000 "
        "characters). Use this when you need content that lives ON a "
        "page -- section headings, a table, a paragraph -- that search "
        "snippets do not carry."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Full URL to fetch"},
        },
        "required": ["url"],
    },
}

SYSTEM_PROMPT = (
    "You are a ReAct agent: you solve questions by interleaving reasoning "
    "and acting. At every step, first write your Thought in plain text: "
    "what you know so far, what information you still need, and what you "
    "will do next. Then either call a tool (Action) or give your final "
    "answer. "
    "If the answer is static knowledge you are confident about, answer "
    "directly with no tool call at all. If anything is dynamic -- a "
    "current fact, a population, today's date, any arithmetic -- you must "
    "interact with the environment through a tool: never do arithmetic in "
    "your head (use the calculator), never guess the date (use "
    "current_datetime), and never state a live fact from memory (use "
    "web_search). Use each tool's observation in your final answer and "
    "say which tool each number came from. If a tool returns an ERROR, "
    "say what went wrong in your next Thought and try a corrected call "
    "once -- do not invent the result the tool would have given."
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
        if name == "fetch_page":
            if "url" not in arguments:
                return "ERROR: fetch_page needs a 'url' argument"
            return func(arguments["url"])
    except Exception as e:  # noqa: BLE001 -- surfaced as an observation
        return f"ERROR: {e}"
    return f"ERROR: no handler for tool {name!r}"


def cost_of(usage) -> float:
    """Cost of one call, in USD, from its token usage."""
    return (usage.input_tokens / 1e6) * PRICE_PER_MTOK_INPUT_USD + (
        usage.output_tokens / 1e6
    ) * PRICE_PER_MTOK_OUTPUT_USD


def test_tools(with_fetch: bool = False) -> None:
    """Prove the tools run locally, without spending an API call."""
    print("[TOOLS TEST] calculator('347 * 28') =", calculator("347 * 28"))
    assert calculator("347 * 28") == "9716", "calculator gave the wrong answer"
    assert calculator("1 / 0").startswith("ERROR"), "division by zero must error"
    assert calculator("__import__('os')").startswith("ERROR"), "code must be rejected"
    print("[TOOLS TEST] calculator: 9716, division-by-zero errors, code rejected")
    print("[TOOLS TEST] current_datetime() =", current_datetime())
    print("[TOOLS TEST] web_search('Anthropic') =", web_search("Anthropic")[:200])
    if with_fetch:
        fetched = fetch_page("https://huggingface.co/learn/agents-course/unit1/introduction")
        print("[TOOLS TEST] fetch_page(HF Unit 1) =", fetched[:200])
        assert not fetched.startswith("ERROR"), "fetch_page failed on a real page"
    print("[TOOLS TEST] all tools ran -- the loop can use them.")


async def run_react(client, question: str, qnum: int, use_fetch: bool) -> float:
    """Run one ReAct question. Returns the total cost in USD.

    The transcript is written to react_q{qnum}_transcript.txt as the run
    goes, so a wrong run can be read back step by step afterwards.
    """
    transcript_path = f"react_q{qnum}_transcript.txt"
    schemas = TOOL_SCHEMAS + ([FETCH_PAGE_SCHEMA] if use_fetch else [])
    messages = [{"role": "user", "content": question}]
    total_cost = 0.0

    with open(transcript_path, "w") as log:
        def write(line: str) -> None:
            print(line)
            log.write(line + "\n")
            log.flush()

        write(f"Q{qnum}: {question}")
        write(f"Tools: calculator, web_search, current_datetime"
              + (", fetch_page" if use_fetch else ""))

        for iteration in range(1, MAX_ITERATIONS + 1):
            start = time.perf_counter()
            resp = await client.messages.create(
                model=MODEL,
                max_tokens=1000,
                system=SYSTEM_PROMPT,
                tools=schemas,
                messages=messages,
            )
            elapsed = time.perf_counter() - start
            total_cost += cost_of(resp.usage)
            write(
                f"\n[Step {iteration} | {elapsed:.1f}s | "
                f"{resp.usage.input_tokens}+{resp.usage.output_tokens} tok = "
                f"${cost_of(resp.usage):.4f}]"
            )

            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            text_parts = [b.text for b in resp.content if b.type == "text"]
            thought = " ".join(text_parts).strip()
            if thought:
                # The model's text before a tool call IS the Thought.
                write(f"Thought: {thought}")

            if not tool_uses:
                # No tool call: this is the final answer. The loop ends here.
                write(f"Finish: {thought}")
                break

            # Append the assistant turn, then the tool results as a user turn.
            messages.append({"role": "assistant", "content": resp.content})
            tool_results = []
            for block in tool_uses:
                observation = run_tool(block.name, block.input or {})
                write(f"Action: {block.name}({block.input})")
                write(f"Observation: {observation[:500]}")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": observation,
                    }
                )
            messages.append({"role": "user", "content": tool_results})
        else:
            write(
                f"\n[STOPPED] hit the {MAX_ITERATIONS}-iteration cap -- code "
                "stopped the loop, the LLM did not. Final answer not reached."
            )

        write(f"\nDone | {iteration} step(s) | total cost ${total_cost:.4f}")
        write(f"Transcript: {transcript_path}")
    return total_cost


async def main(args) -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY in your environment first.")

    client = anthropic.AsyncAnthropic()
    grand_total = 0.0
    if args.question:
        grand_total += await run_react(
            client, QUESTIONS[args.question], args.question, args.fetch
        )
    else:
        for qnum, question in QUESTIONS.items():
            grand_total += await run_react(client, question, qnum, args.fetch)
    print(f"\nAll done | grand total ${grand_total:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Week 5: hand-rolled ReAct loop")
    parser.add_argument("--test-tools", action="store_true",
                        help="check the tools with no API call")
    parser.add_argument("--question", type=int, choices=range(1, 6),
                        help="run one question (1-5); default runs all five")
    parser.add_argument("--fetch", action="store_true",
                        help="enable the stretch fetch_page tool")
    parsed = parser.parse_args()
    if parsed.test_tools:
        test_tools(with_fetch=parsed.fetch)
    else:
        asyncio.run(main(parsed))
