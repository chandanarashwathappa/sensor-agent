"""Run the 20-question evaluation against the real agent (Mistral API).

    python -m evaluation.run_eval

Needs MISTRAL_API_KEY in .env. Writes evaluation/results.md and results.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from agent.agent import build_default_agent
from agent.analytics import DATA_PATH

from .checks import grade
from .questions import build_questions

OUT = Path(__file__).resolve().parent
PAUSE_S = 1.5          # stay below the API rate limit


def main() -> None:
    agent = build_default_agent()
    questions = build_questions(DATA_PATH)
    rows = []
    for q in questions:
        for attempt in range(3):
            try:
                res = agent.ask(q["question"])
                break
            except Exception as exc:            # rate limit / network: retry
                print(f"  retry {attempt + 1}: {exc}")
                time.sleep(5 * (attempt + 1))
        else:
            raise SystemExit(f"question {q['id']} failed three times")
        ok = grade(q, res.answer)
        tools = ", ".join(s.tool for s in res.steps) or "-"
        rows.append({**q, "answer": res.answer, "tools": tools, "correct": ok})
        print(f"{'PASS' if ok else 'FAIL'}  Q{q['id']:2d}  [{tools}]  {q['question']}\n        -> {res.answer[:150]}")
        time.sleep(PAUSE_S)

    score = sum(r["correct"] for r in rows)
    print(f"\nScore: {score}/{len(rows)} correct")
    (OUT / "results.json").write_text(json.dumps(rows, indent=2, default=str))
    lines = [f"# Evaluation results\n\n**{score}/{len(rows)} correct** "
             f"(model: mistral-small-latest, temperature 0)\n",
             "| # | Question | Expected | Tools used | Result |", "|---|---|---|---|---|"]
    for r in rows:
        exp = f"{r['expected']:.2f}" if r["kind"] == "number" else "keywords: " + " + ".join("/".join(g) for g in r["expected"])
        lines.append(f"| {r['id']} | {r['question']} | {exp} | {r['tools']} | {'✅' if r['correct'] else '❌'} |")
    (OUT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
