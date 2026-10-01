---
name: crisp-dm
description: Orchestrator for the full CRISP-DM cycle (Provost & Fawcett, Data Science for Business, pp. 26-35). Detects where a project stands from its crisp/ reports, runs the six stage skills (crisp-1 ... crisp-6) in order, enforces each exit gate, routes loop-backs to earlier stages, and keeps an iteration log. Use when starting a data-science project end to end, resuming one, or asking "what's the next CRISP step?". Args - optional: a stage number to jump to (1-6), "status" to only report state, or "new" to start a fresh iteration.
---

# CRISP-DM Orchestrator

> "Iteration is the rule rather than the exception." The process is cycles
> within a cycle, with shortcuts back from each stage to any earlier one.

This skill does not do stage work itself. It decides **which stage runs next**,
invokes that stage's skill via the Skill tool, checks the stage's exit gate,
and records the outcome.

| # | Stage | Skill | Report |
|---|---|---|---|
| 1 | Business Understanding | `crisp-1-business-understanding` | `crisp/01_business_understanding.md` |
| 2 | Data Understanding | `crisp-2-data-understanding` | `crisp/02_data_understanding.md` |
| 3 | Data Preparation | `crisp-3-data-preparation` | `crisp/03_data_preparation.md` |
| 4 | Modeling | `crisp-4-modeling` | `crisp/04_modeling.md` |
| 5 | Evaluation | `crisp-5-evaluation` | `crisp/05_evaluation.md` |
| 6 | Deployment | `crisp-6-deployment` | `crisp/06_deployment.md` |

## State file — `crisp/STATE.md`
Create it on first run; update it after every stage. It holds:
- **Project:** one-line problem statement
- **Iteration:** N (starts at 1; +1 each time the cycle returns to Stage 1)
- **Current stage:** 1–6, and status `in-progress | gate-passed | gate-failed | blocked`
- **Stage table:** stage · report path · last updated · gate result
- **Transition log** (append-only): date · iteration · from → to · reason
  (e.g. `5 → 1: false-alarm cost exceeds analyst capacity; reformulate as ranking`)
- **Open human checkpoints** awaiting the user

## Procedure
1. **Locate project root.** Use the current working directory unless the user
   names another; reports live in `<root>/crisp/`.
2. **Resolve state.**
   - `status` arg → print the STATE.md summary and next recommended step; stop.
   - `new` arg, or no `crisp/` directory → create `crisp/STATE.md`, iteration 1
     (or N+1 if a prior iteration exists; archive old reports to
     `crisp/iter<N>/`), start at Stage 1.
   - Stage-number arg → jump there, but first verify every earlier stage's
     report exists and its gate passed; if not, say which is missing and start
     at the earliest incomplete stage instead.
   - Otherwise → resume at the first stage whose gate has not passed.
3. **Run the stage.** Invoke its skill with the Skill tool. Pass it the paths
   of all earlier reports. The stage skill writes its report.
4. **Check the exit gate.** Re-read the stage report and evaluate every gate
   checkbox from that stage's skill against evidence in the report (not
   against intent). Record `gate-passed` or `gate-failed` with the failing
   items.
5. **Route.**
   - Gate passed → advance to the next stage.
   - Gate failed → apply the stage's loop-back rule, e.g.
     2 → 1 (no viable data for the task) ·
     3 → 2 (features need unsourced data) ·
     4 → 3 or 1 (no model beats baseline) ·
     5 → 1 (results not good enough; the diagram's Evaluation→Business
     Understanding shortcut) or to whichever stage the evaluation implicates.
     Any stage may return to any earlier stage when a discovery warrants it.
     Log the transition with a concrete reason. When re-entering a stage,
     revise its report in place and add an entry to its iteration log —
     downstream reports become stale and must be re-run in order.
   - After Stage 6 → always return to Stage 1 with lessons learned
     (iteration +1), whether or not deployment succeeded.
6. **Stop at human checkpoints** (do not proceed on your own):
   - End of Stage 1: user confirms the formulation and success criteria.
   - Stage 2: any decision to purchase data or start a data-collection project.
   - Stage 5: stakeholder sign-off — the model may not advance to Stage 6
     without it recorded in `05_evaluation.md`.
   - Stage 6: any action that touches a production system, external service,
     or real users/patients (rollout, randomized live trial) needs explicit
     user approval for that specific action.
   Record the pending checkpoint in STATE.md, tell the user what is needed,
   and end the turn.
7. **Guard against loops.** If the same backward transition happens 3 times in
   one iteration, stop and summarize the recurring blocker for the user rather
   than cycling again.
8. **Report.** After each run, tell the user: stages run, gate results, any
   loop-back and why, and the next step or pending checkpoint.

## Cross-stage invariants (check at every gate)
- The use scenario in `01` (what is known at decision time) is still the one
  the features in `03` and the evaluation in `05` assume.
- The holdout fixed in `04` has not been used for training or tuning.
- Numbers in reports come from saved output files, never typed in by hand.
- No licensed raw data inside the repo; large intermediates on bulk scratch.

## Team-management reminder (book pp. 34-35)
CRISP is exploratory, closer to R&D than to software engineering. Prefer cheap
information first — pilot studies, throwaway prototypes, literature review —
before committing to engineering for deployment.
