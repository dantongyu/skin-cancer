---
name: crisp-5-evaluation
description: CRISP-DM stage 5 of 6 (Provost & Fawcett, Data Science for Business, pp. 31-32). Rigorously assess whether model results are valid, reliable, and actually meet the original business goals - quantitative holdout tests, business-context costs (e.g. false-alarm burden), stakeholder comprehensibility and sign-off, testbed or randomized in-vivo evaluation, and monitoring plans. Use before any deployment decision.
---

# CRISP-DM Stage 5 — Evaluation

> Assess results rigorously and gain confidence they are valid and reliable
> before moving on. It is usually far easier, cheaper, quicker, and safer to
> test a model in a controlled laboratory setting first.

## Inputs
- `crisp/01_business_understanding.md` (success criteria, sign-off owners)
- `crisp/04_modeling.md` (selected model, untouched holdout, run log)

## Procedure
1. **Validity check (lab).** Score the selected model once on the untouched
   holdout. Ask: are these true regularities, or idiosyncrasies / sample
   anomalies? Compare with the baseline; report uncertainty, not just a point
   estimate, and state what the result is conditional on (selection steps).
2. **Business-goal check.** Evaluate against the Stage-1 expected-value
   framing, not only accuracy. The model is usually only one piece of a larger
   solution — evaluate it as such. Example: a >99%-accurate detector may still
   produce too many false alarms to be economically feasible. Ask: what staff
   cost do the false alarms imply? What customer/patient dissatisfaction?
3. **External/practical constraints.** Anything that makes a lab-passing model
   impractical (latency, data availability in production, regulation, ethics,
   operational capacity).
4. **Qualitative assessment & sign-off.** Present to the stakeholders who must
   sign off. They mainly want to know: will it do more good than harm, and is
   it unlikely to make catastrophic mistakes? (Phone-network example:
   stakeholders required exceptions for hospitals.) Make the model's behavior
   comprehensible to them — if the model itself is opaque, explain its
   behavior (examples, rules of thumb, sensitivity analyses).
5. **Testbed evaluation.** Where possible, evaluate in an environment that
   mirrors production data as closely as possible — production access is
   limited and deployed systems have many moving parts.
6. **In-vivo evaluation (if warranted).** Design a randomized experiment: the
   live system applies the model to a random subset, the rest is a control
   group. Design carefully (see Kohavi et al.).
7. **Plan ongoing monitoring.** The world changes — sometimes *because* of the
   model (fraud, spam adapt); input data change format or substance without
   warning. Specify what will be instrumented and what triggers an alert.
8. **Decide:** deploy / iterate (which stage?) / stop.

## Deliverable — `crisp/05_evaluation.md`
- Holdout results vs. baseline, with uncertainty and conditioning statement
- Business-value assessment (expected value, false-alarm burden, capacity)
- Practical/external constraints found
- Stakeholder review notes and sign-off status; requested exceptions
- Testbed / in-vivo experiment design or results
- Monitoring plan (metrics, drift checks, alert thresholds)
- Decision and rationale

## Exit gates
- **→ Stage 6 (`crisp-6-deployment`)** only if: holdout result holds up,
  business criteria met, stakeholders signed off, monitoring plan exists.
- **→ Stage 1 shortcut** if results are not good enough to deploy and the
  problem definition or data must change (the diagram's Evaluation→Business
  Understanding link). Any earlier stage may be revisited.

## Pitfalls
- Declaring success on lab accuracy alone.
- Re-using the holdout repeatedly until it "passes".
- Treating non-rejection as proof; overstating what was actually measured.
