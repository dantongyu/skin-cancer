---
name: crisp-3-data-preparation
description: CRISP-DM stage 3 of 6 (Provost & Fawcett, Data Science for Business, pp. 29-30). Convert raw data into a modeling-ready table - format conversion, missing values, type conversion, scaling, feature (variable) engineering - and audit for leakage. Use after Data Understanding and before Modeling, or whenever building/changing features.
---

# CRISP-DM Stage 3 — Data Preparation

> Analytic technologies impose requirements on the data they use. Preparation
> often proceeds *alongside* data understanding.

## Inputs
- `crisp/01_business_understanding.md` (use scenario: what is known at decision time)
- `crisp/02_data_understanding.md` (sources, targets, gaps)

## Procedure
1. **Define the unit of analysis and target.** One row per instance (customer,
   patient, claim, visit…), one column per variable, target column per Stage 1.
   Fix the prediction time point for each instance.
2. **Convert to tabular form**; join sources per the Stage-2 linkage plan.
3. **Handle missing values** — remove or infer, and record which and why
   (missingness can itself be informative; consider an indicator).
4. **Convert types to suit candidate techniques**: some methods want
   symbolic/categorical data, others only numeric (encode accordingly).
5. **Normalize / scale** numeric values where comparability matters
   (distance-based or regularized methods).
6. **Craft the variables.** This is where human creativity, common sense, and
   domain knowledge matter most; solution quality often rests on how well the
   problem is structured and the variables crafted. Document each feature's
   definition and rationale.
7. **Leakage audit (mandatory).** A leak is a variable in historical data that
   carries information about the target but is *not available when the decision
   must be made*. For every feature ask: "Would I know this value at decision
   time in the use scenario?" Classic leaks: total pages visited in a session
   when predicting session end; item categories or tax paid when predicting
   "big spender". Check timestamps relative to the prediction point; treat
   suspiciously strong predictors as leaks until proven otherwise.
8. **Make it reproducible.** Preparation is a script, not manual edits; outputs
   are regenerable. Put large intermediates on bulk scratch, keep the code and
   a small data dictionary with the repo.

## Deliverable — `crisp/03_data_preparation.md` + prep script
- Unit of analysis, prediction time point, target definition
- Data dictionary: feature · definition · source · type/encoding · missing-value
  treatment · available-at-decision-time (Y/N, evidence)
- Leakage audit table with verdict for every feature
- Row counts through each filtering step
- Path to the generating script and output dataset

## Exit gate → Stage 4 (`crisp-4-modeling`)
- [ ] Every feature passes the decision-time availability check
- [ ] Prep is fully scripted and re-runnable
- [ ] Data dictionary complete
- Loop back to **Stage 2** if features need data not yet sourced.

## Pitfalls
- Leakage — preparation is done after the fact from historical data, which is
  exactly why leaks slip in.
- Silent row loss in joins/filters.
- Fitting scalers/imputers on data that includes the evaluation holdout.
