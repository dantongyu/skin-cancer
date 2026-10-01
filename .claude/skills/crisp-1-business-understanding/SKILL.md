---
name: crisp-1-business-understanding
description: CRISP-DM stage 1 of 6 (Provost & Fawcett, Data Science for Business, pp. 26-28). Turn a vague business/research problem into one or more well-posed data-mining tasks with an explicit use scenario and value framing. Use at the START of any data-science project, or whenever a later stage loops back because the problem definition was wrong.
---

# CRISP-DM Stage 1 — Business Understanding

> "Business projects seldom come pre-packaged as clear and unambiguous data mining
> problems." Recasting the problem is itself an iterative process of discovery —
> "cycles within a cycle."

Iteration is the rule, not the exception: a first pass that does not solve the
problem is not a failure if it leaves the team better informed.

## Inputs
- The stakeholder's problem statement (however vague).
- If re-entering: the prior `crisp/0N_*.md` reports that triggered the loop-back.

## Procedure
1. **Restate the decision.** What decision will be made differently because of
   this work? Who makes it, when, and how often? Data science for business exists
   to support decision making — no decision, no project.
2. **Write the use scenario concretely** (the book's most-emphasized concept):
   - What exactly do we want to do? How exactly would we do it?
   - At the moment of decision, what is known about the individual/case?
   - What action follows from a model output, and who carries it out?
   - Which parts of this scenario could be data-mining models?
3. **Decompose into canonical data-mining tasks** (book p. 19). Map each
   sub-problem to one or more of:
   classification / class-probability estimation · regression · similarity
   matching · clustering · co-occurrence grouping · profiling · link prediction ·
   data reduction · causal modeling.
   A business problem often contains several tasks of different types whose
   solutions must be combined.
4. **Frame in expected value.** Sketch benefit/cost of each outcome (e.g. true
   positive, false positive, missed case) so success criteria are business
   quantities, not only accuracy. Rough numbers are fine; state assumptions.
5. **Supervised or unsupervised?** Note whether a reliable target label is
   *likely* to exist (verified in Stage 2). Superficially similar problems
   can differ here (credit-card fraud has labels; Medicare fraud does not).
6. **Define success and stop criteria** the Evaluation stage will test against,
   including qualitative ones (catastrophic-error tolerance, comprehensibility
   required by stakeholders, sign-off owners).
7. **Involve deployment early.** Name the engineers/operators who will build or
   run the result; invite them as advisors now (see Stage 6).

## Deliverable — `crisp/01_business_understanding.md`
- Decision supported; decision maker; cadence
- Use scenario (narrative + what is known at decision time)
- Task decomposition table: sub-problem → data-mining task → candidate target
- Expected-value sketch and success criteria (quantitative + qualitative)
- Stakeholders who must sign off; deployment partners
- Open questions / assumptions to verify in Stage 2
- Iteration log: version, date, what changed and why

## Exit gate → Stage 2 (`crisp-2-data-understanding`)
- [ ] Every sub-problem maps to a named data-mining task
- [ ] Use scenario states what information is available *at decision time*
- [ ] Success criteria are measurable and tied to the business goal
- [ ] Sign-off owners identified

## Pitfalls
- Jumping to an algorithm before the use scenario is written.
- Optimizing a proxy (accuracy) the business does not care about.
- Treating the first formulation as final — expect to revise it.
