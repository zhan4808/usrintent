Intent Refinement Tool: concise spec grounded in 3 sources

Goal
Build a developer-focused tool that tightens user intent into durable, testable prompts, then closes the loop using automated feedback + user edits.

Chosen research foundations (3)
1) Self-Refine (selfrefine.md)
   - Mechanism: generate -> feedback -> refine loop on outputs.
   - Why: simplest closed-loop; maps directly to "revise intent prompt after output review."
2) Evoke (evoke.md)
   - Mechanism: author/reviewer dual-agent with memory of edits and scores.
   - Why: natural "intent editor" + "intent critic" split.
3) SIPDO (sipdo.md)
   - Mechanism: synthetic, hard-case data to stress prompts.
   - Why: supports automated evaluation and progressive challenge.

How these map to the product
- Self-Refine: baseline refine loop for prompt revisions.
- Evoke: reviewer/author roles for structured critique + edits.
- SIPDO: synthetic stress tests to reveal prompt weaknesses.

MVP features (split)
Core features
- CLI wrapper that stores intent, prompts, outputs, and feedback.
- Local store (SQLite) with simple JSON export.
- LLM-as-a-judge scoring with a small rubric.
- One refine strategy (reviewer/author loop).
- Basic diff capture of user edits.

Refined features
- Synthetic stress tests (small batch) for regression checks.
- Prompt ensemble strategies (optional, later).
- UI for diff review + intent checklist.
- Telemetry aggregation and opt-in cloud sync.
- Requirements-engineering module for software-only flows.

Data model (minimal)
Intent
{
  "id": "intent_001",
  "description": "Add CSV parser with error handling",
  "task_type": "code_gen",
  "constraints": ["no new deps", "keep public API"],
  "done_criteria": ["tests pass", "no lint errors"],
  "status": "draft"
}

PromptVersion
{
  "id": "prompt_v001",
  "intent_id": "intent_001",
  "content": "...",
  "metrics": {
    "judge_score": 0.0,
    "user_rating": null
  }
}

Feedback
{
  "id": "fb_001",
  "prompt_id": "prompt_v001",
  "user_edits": "Added error line numbers",
  "judge_scores": {
    "relevance": 0.84,
    "correctness": 0.77,
    "coherence": 0.90
  }
}

CLI (minimal)
- tool intent new "..." --type=code --constraints="no deps"
- tool run intent_001 --model=...
- tool feedback prompt_v001 --rating=4 --edit-file=diff.patch
- tool refine intent_001
- tool history intent_001

Evaluation rubric (starter)
- Relevance: output aligns with intent.
- Correctness: tests pass, builds, lint clean.
- Coherence: output is structured and readable.
- Risk: avoid unsafe or hallucinated behavior.

Stop criteria
- User accepts output.
- No improvement after N iterations.
- Regression detected (judge score down or tests fail).

Roadmap (broader)
- Add prompt ensembles if needed later.
- Add requirements-engineering module for software-only flows.
- Add UI for diff review + intent checklist.
- Add telemetry aggregation and opt-in cloud sync.