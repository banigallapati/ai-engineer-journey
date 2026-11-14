# Week 5: A hand-rolled ReAct loop

`week5_react_starter.py` answers questions with a ReAct loop (Thought → Action → Observation) written by hand, with no agent framework. Claude decides which tool to call, and plain Python code runs the tool and sends the result back. It extends the Week 4 tool-calling starter: the tools, the argument checks and the per-call cost logging carry over unchanged.

## How the loop works

1. The question goes to `claude-sonnet-5-5` along with the tool definitions.
2. Claude replies with a short note saying what it's about to look up or calculate (the **Thought**), and either a tool call (the **Action**) or its final answer (**Finish**).
3. For a tool call, the code checks the arguments, runs the tool and sends back the result (the **Observation**). Claude never runs a tool itself.
4. This repeats until Claude answers without calling a tool. `MAX_ITERATIONS = 8` is a hard cap enforced in code, so a run can't loop forever.

There's no `Thought:`/`Action:` text parser. The API's tool use already returns text blocks (the Thought) and `tool_use` blocks (the Action) separately.

## Tools

| Tool | What it does |
|---|---|
| `calculator(expression)` | Safe arithmetic, parsed with `ast` (no `eval()`) |
| `web_search(query)` | DuckDuckGo search via `ddgs`, top 3 results |
| `current_datetime()` | The current time in America/New_York and UTC |
| `fetch_page(url)` | Optional, enabled with `--fetch`: downloads a page and returns its text (HTML stripped, first ~3000 characters), for questions that need what's *on* a page, not just search snippets |

The system prompt tells Claude to answer static knowledge directly, and to use a tool for anything dynamic: today's date, current facts, and all arithmetic.

## Questions

Six built-in questions in `QUESTIONS`, each needing a different mix of tools, for example: *"In what year was the Eiffel Tower completed, and how many years ago was that?"* Claude answers the year (1889) from memory, since it's static, then uses `current_datetime` and `calculator` for the "how many years ago" part.

## Running it

```bash
pip install anthropic ddgs python-dotenv
export ANTHROPIC_API_KEY=...       # or put it in a .env file; never in code

python week5_react_starter.py --test-tools           # checks the tools, no API call
python week5_react_starter.py --question 4           # run one question (1-6)
python week5_react_starter.py                        # run all six
python week5_react_starter.py --question 6 --fetch   # also enable fetch_page
```

## What you see

Each step prints a header with its time, `stop_reason`, tokens and cost, followed by:

- `Update:` lines: Claude's short progress notes between tool calls. They come from `thinking={"type": "adaptive", "display": "updates"}`, a beta feature, which is why the request uses `client.beta.messages.create`.
- `Thought:`, `Action:` and `Observation:` lines, then `Finish:` with the final answer.

Every run is also written to `react_qN_transcript.txt`, so you can read back afterwards whether a wrong answer came from the Thought, the Action or the Observation. These transcripts are ignored by git (see `.gitignore`).

## Failure handling

- **Bad tool arguments** (an unknown tool, a missing argument, an invalid expression) come back to Claude as an `ERROR:` observation instead of crashing the loop. Claude is asked to say what went wrong and try one corrected call.
- **An empty response** is never treated as a final answer. The step is retried once; if it's still empty, the run stops with an `[ERROR]` line.
- **A refusal** (`stop_reason="refusal"`) also comes back empty, so it is handled the same way. The log line includes `stop_details`, which names the safety filter that fired.
