---
name: crisp-2-data-understanding
description: CRISP-DM stage 2 of 6 (Provost & Fawcett, Data Science for Business, pp. 28-29). Inventory and interrogate available data sources against the business problem - coverage, reliability, cost, label availability - and confirm or re-route the task formulation. Use after Business Understanding, or when a dataset is handed over and you must judge whether it fits the problem.
---

# CRISP-DM Stage 2 — Data Understanding

> The data are the raw material from which the solution is built. "Rarely is
> there an exact match with the problem."

## Inputs
- `crisp/01_business_understanding.md` (task decomposition, use scenario).
- Access to candidate data sources.

## Procedure
1. **Inventory sources.** For each source record: what it contains, why and
   how it was originally collected (often for unrelated purposes or none), the
   population it covers, time span, granularity, owner, access terms.
2. **Assess fit and limitations.** Different databases (customer, transaction,
   response…) contain different information, cover different *intersecting*
   populations, and vary in reliability. Note gaps relative to the use scenario.
3. **Estimate cost vs. benefit of each source.** Some data are free, some need
   effort, some must be purchased, some don't exist and need an ancillary
   collection project. Decide explicitly whether further investment is merited.
4. **Check the target variable.** Is there a reliable label for each supervised
   task from Stage 1? Who assigns it, and do they have an incentive to get it
   right? (Credit-card fraud: the victim reliably reports it → labels exist.
   Medicare fraud: perpetrators are legitimate users, no disinterested labeler
   → no reliable target; use profiling, clustering, anomaly detection,
   co-occurrence grouping instead.)
5. **Look beneath superficial similarity.** Two problems with the same name
   can need different task types. Match the *structure* of problem + data to
   tasks with established science behind them.
6. **Assess linkage/collation effort.** Entity resolution (one record per
   customer/patient, noisy product IDs) is itself a hard analytics problem;
   scope it.
7. **Profile the data** (counts, missingness, distributions, label base rates,
   time coverage) — enough to inform Stage 3, not a full cleaning effort.
8. **Let the solution path move.** If findings change the formulation, record
   it and loop back to Stage 1; team efforts may fork into parallel paths.

## Deliverable — `crisp/02_data_understanding.md`
- Source inventory table: source · contents · origin/purpose · population ·
  period · reliability · cost to obtain · decision (use / buy / collect / drop)
- Target-label assessment per task (exists? reliable? base rate?)
- Linkage/collation plan and risks
- Profiling summary
- Changes to the Stage-1 formulation (if any) and why

## Exit gate → Stage 3 (`crisp-3-data-preparation`)
- [ ] Each task has a data source and (if supervised) a credible target
- [ ] Cost/benefit decision recorded for each source
- [ ] Known coverage/reliability gaps written down for Evaluation to revisit
- Loop back to **Stage 1** if no viable data exists for the formulated task.

## Pitfalls
- Assuming historical data were collected for your purpose.
- Assuming labels exist because the problem "sounds supervised".
- Underestimating record matching/cleaning effort.
- Never copy licensed raw data into a repo — commit pull scripts instead.
