"""Run the 20-question evaluation against the real agent.

    python -m evaluation.run_eval

The backend comes from LLM_PROVIDER (see agent.agent.model_config):
    LLM_PROVIDER=ollama   local model via Ollama, no API key (used in CI)
    LLM_PROVIDER=mistral  Mistral API, needs MISTRAL_API_KEY in .env
Writes evaluation/results.md and results.json.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from agent.agent import build_default_agent, model_config
from agent.analytics import DATA_PATH

from .checks import grade
from .questions import build_questions

OUT = Path(__file__).resolve().parent


def main() -> None:
    cfg = model_config()
    model = f"{cfg['chat_model']} via {cfg['provider']}"
    pause_s = 1.5 if cfg["provider"] == "mistral" else 0.0      # API rate limit only
    agent = build_default_agent()
    questions = build_questions(DATA_PATH)
    rows = []
    for q in questions:
        for attempt in range(3):
            try:
                t0 = time.monotonic()
                res = agent.ask(q["question"])
                seconds = time.monotonic() - t0
                break
            except Exception as exc:            # rate limit / network: retry
                print(f"  retry {attempt + 1}: {exc}")
                time.sleep(5 * (attempt + 1))
        else:
            raise SystemExit(f"question {q['id']} failed three times")
        ok = grade(q, res.answer)
        tools = ", ".join(s.tool for s in res.steps) or "-"
        rows.append({**q, "answer": res.answer, "tools": tools, "correct": ok, "seconds": round(seconds, 1)})
        print(f"{'PASS' if ok else 'FAIL'}  Q{q['id']:2d}  [{tools}]  ({seconds:.1f} s)  {q['question']}\n"
              f"        -> {res.answer[:200]}", flush=True)
        time.sleep(pause_s)

    score = sum(r["correct"] for r in rows)
    failed = [f"Q{r['id']}" for r in rows if not r["correct"]]
    print(f"\nScore: {score}/{len(rows)} correct ({model})")
    (OUT / "results.json").write_text(json.dumps({"model": model, "score": score, "total": len(rows),
                                                  "rows": rows}, indent=2, default=str))
    lines = [f"# Evaluation results\n\n**{score}/{len(rows)} correct** (model: {model}, temperature 0)\n",
             "| # | Question | Expected | Tools used | Result |", "|---|---|---|---|---|"]
    for r in rows:
        exp = f"{r['expected']:.2f}" if r["kind"] == "number" else "keywords: " + " + ".join("/".join(g) for g in r["expected"])
        lines.append(f"| {r['id']} | {r['question']} | {exp} | {r['tools']} | {'✅' if r['correct'] else '❌'} |")
    report = "\n".join(lines) + "\n"
    (OUT / "results.md").write_text(report, encoding="utf-8")

    if os.getenv("GITHUB_ACTIONS") == "true":          # show the result on the CI run page
        print(f"::notice title=Evaluation score::{score}/{len(rows)} correct ({model})")
        print(f"::notice title=Failed questions::{', '.join(failed) if failed else 'none'}")
        summary = os.getenv("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a", encoding="utf-8") as f:
                f.write(report)


if __name__ == "__main__":
    main()
