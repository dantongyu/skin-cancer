---
name: crisp-4-modeling
description: CRISP-DM stage 4 of 6 (Provost & Fawcett, Data Science for Business, p. 31). Apply data-mining techniques to the prepared data to produce models or patterns capturing regularities, chosen to match the task type from Business Understanding. Use after Data Preparation, when fitting/comparing models.
---

# CRISP-DM Stage 4 — Modeling

> The output of modeling is some sort of model or pattern capturing
> regularities in the data. This is the part of the craft where the most
> science and technology can be brought to bear.

## Inputs
- `crisp/01_business_understanding.md` (task types, success criteria)
- `crisp/03_data_preparation.md` + the prepared dataset

## Procedure
1. **Match technique to task.** Pick model families appropriate to each task
   from Stage 1 (classification/probability estimation, regression, clustering,
   co-occurrence, profiling, etc.) and to the data types prepared in Stage 3.
2. **Set aside evaluation data first.** Before fitting anything, fix a holdout
   (or cross-validation scheme, time-ordered if the use scenario is
   forward-looking) that the Evaluation stage will use. Do not tune on it.
3. **Establish a simple baseline** (majority class, mean, a simple rule, or
   the current business practice) and record its score before trying complex
   models.
4. **Fit candidate models**, from simple to complex. Control complexity /
   overfitting (regularization, tree depth, etc.) — "if we look hard enough at
   any dataset we will find patterns."
5. **Record every run**: data version, features, hyperparameters, seed,
   validation score. Log all selection steps so Evaluation can state what its
   inference is conditional on.
6. **Prefer comprehensible models when comparable.** Stakeholders will need to
   understand behavior for sign-off (Stage 5); if the best model is opaque,
   plan how to explain its behavior.
7. **Loop back when stuck.** Poor performance often means a preparation or
   formulation problem, not an algorithm problem.

## Deliverable — `crisp/04_modeling.md` + run log
- Candidate models per task and why chosen
- Baseline score(s)
- Run log table: run id · model · features · hyperparameters · validation score
- Selected model(s) and the selection rule used
- Known weaknesses / overfitting concerns for Evaluation

## Exit gate → Stage 5 (`crisp-5-evaluation`)
- [ ] Holdout untouched by training and tuning
- [ ] Selected model beats the baseline on validation data
- [ ] All selection steps logged
- Loop back to **Stage 3** (features) or **Stage 1** (formulation) if not.

## Pitfalls
- Tuning against the test set; reporting the best of many tries without
  disclosure.
- Mismatch between model output (e.g. a score) and what the use scenario needs
  (e.g. a ranking, a probability, a yes/no at a threshold).
