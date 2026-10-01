---
name: crisp-6-deployment
description: CRISP-DM stage 6 of 6 (Provost & Fawcett, Data Science for Business, pp. 32-34). Put data-mining results into real use to realize return on investment - integrate a model into an information system or business process, deploy the mining system itself when appropriate, or deploy non-technical insights - with engineering hand-off, instrumentation, fail-safes, and a return to Business Understanding. Use once Evaluation approves deployment.
---

# CRISP-DM Stage 6 — Deployment

> Results of data mining — and increasingly the techniques themselves — are
> put into real use to realize some return on investment.
> "Your model is not what the data scientists design, it's what the engineers build."

## Inputs
- `crisp/05_evaluation.md` (approval, monitoring plan, sign-offs)
- Selected model / pattern / insight and its prep pipeline

## Procedure
1. **Choose the deployment form:**
   - **Model in a process/system** — e.g. churn scores driving targeted
     offers; a fraud model creating cases for analysts.
   - **The mining system itself** — automatically (re)builds and tests models
     in production. Warranted when (i) the world changes faster than the team
     can adapt (fraud, intrusion) or (ii) there are too many modeling tasks to
     curate by hand. Requires instrumentation that alerts the team to
     anomalies plus fail-safe operation.
   - **Non-technical** — rules taped to a printer; a change to data
     acquisition procedures, strategy, marketing, or operations.
2. **Plan the re-implementation.** Production often requires re-coding for
   speed or compatibility — budget for it. Hand-off package: working
   prototype, its evaluation, the exact prep pipeline, data dictionary, and
   test cases with expected outputs so engineers can verify parity.
3. **Avoid "over the wall" transfer.** Developers (ideally "data science
   engineers") should already be advising since Stage 1; they gradually take
   ownership. Data scientists stay involved through final deployment.
4. **Verify parity.** Confirm the production implementation reproduces the
   prototype's outputs on the test cases and that every input feature is
   genuinely available at decision time in production.
5. **Instrument and monitor** per the Stage-5 plan: input-data drift/format
   changes, output distribution, business KPIs, behavior changes provoked by
   the model; define rollback / fail-safe behavior.
6. **Roll out cautiously** (pilot, staged, or randomized per Stage 5).
7. **Close the loop.** Whether or not deployment succeeds, return to
   Stage 1 (`crisp-1-business-understanding`) with what was learned — new
   ideas, better formulations, even new lines of business.

## Deliverable — `crisp/06_deployment.md`
- Deployment form and integration point
- Hand-off package contents and owners (data science vs. engineering)
- Parity-test results
- Monitoring/alerting and fail-safe/rollback spec
- Rollout plan and status
- Lessons learned and proposals for the next CRISP iteration

## Exit gate
- [ ] Parity verified; monitoring live; rollback tested; owner named
- [ ] Lessons fed back into a new `crisp/01_business_understanding.md` revision

## Managing the team (book pp. 34-35)
Data mining is exploratory, closer to R&D than software engineering; outcomes
are uncertain and each step can change the understanding of the problem.
Engineering directly for deployment can be an expensive premature commitment —
invest in information first (pilot studies, throwaway prototypes, literature
review, experimental testbeds). Value people who formulate problems well,
prototype fast, make reasonable assumptions, design good experiments, and
analyze results.
