# Uplift modeling for employee retention

**Who to keep — not who will leave.**

A flight-risk model predicts *who will leave*. A retention decision needs *who will stay because of what you do* — the causal effect of an intervention on an individual's choice to stay, their **uplift**. Those are different questions, and they point at different people. A risk score does not contain uplift, and no amount of predictive accuracy will put it there.

This repo demonstrates the method: fit an uplift model, show it recovers the true individual effect, and show that targeting by estimated uplift keeps substantially more people than targeting by predicted flight risk on the same budget.

Companion pieces: the explainer *"Predicting who leaves isn't knowing who to keep"* and an interactive [demo](https://amira-ghazy.github.io). The prediction half — done well — lives in [`attrition-fairness`](https://github.com/amira-ghazy/attrition-fairness); this is the half it can't reach.

## Why semi-synthetic data

Estimating uplift requires variation in *who was treated*. **Validating** an uplift estimate requires knowing the *true* per-person effect — which real data never gives you, because you only ever observe each person in one world (treated or not). No public HR dataset carries a randomized retention intervention, so this repo simulates one:

- realistic-looking HR covariates (satisfaction, tenure, overtime, pay gap, commute, manager quality, performance, age);
- a **true baseline leave risk** `p0(x)`;
- a **true heterogeneous uplift** `tau(x)` — the reduction in leave probability if treated;
- a **randomized** intervention `T`, and an observed outcome drawn from `p0 - T·tau`.

The model never sees `tau`. Because we *do*, we can score whether it recovered the truth. Randomization is what makes the effect identifiable here — observational data would require adjusting for confounding, and that burden is the real-world version of this problem, not a footnote.

The design deliberately makes **risk ≠ uplift**: baseline risk is driven partly by an *unfixable* cause (a long commute, standing in for relocation), while uplift comes only from *fixable* causes the intervention addresses (pay gap, overtime, a weak manager). A small group of content high performers have **negative** uplift — for them, a "retention check-in" signals they're being managed out and pushes them toward the door.

## Method

A **T-learner** (two-model approach): a gradient-boosted model of `P(leave)` fit on the control group, another on the treated group. Estimated uplift is the difference, `P(leave | control) − P(leave | treated)`. Baseline flight risk is the control model's prediction.

## Results

\`\`\`
=== Did the model recover the true effect? ===
corr(estimated uplift, TRUE uplift) = 0.733
corr(TRUE uplift, baseline risk)    = 0.481   (modest -> who's at risk is NOT who you can move)

=== Policy comparison: expected TRUE stays saved at each budget ===
 budget |  by risk | by uplift | uplift / risk
     5% |     58.0 |      82.3 |        1.42x
    10% |     87.9 |     145.5 |        1.65x
    20% |    143.4 |     237.8 |        1.66x
    40% |    276.4 |     336.6 |        1.22x

(random targeting @10% saves 33.1)

=== The four types in this workforce (by true risk & true uplift) ===
  sure thing              1212  (37.9%)
  persuadable             1004  (31.4%)
  lost cause               558  (17.4%)
  persuadable (low-risk)   277  ( 8.7%)
  sleeping dog             149  ( 4.7%)
\`\`\`

Two things to read here. First, the model recovers the true individual effect well (0.73) without ever seeing it. Second, the payoff: at a realistic, tight budget, **uplift targeting keeps 1.4–1.7× as many people as flight-risk targeting for the same spend** — because flight-risk targeting pours the budget into *lost causes* (high risk, no uplift) while uplift targeting finds the *persuadables*. The advantage is widest when the budget is tight and narrows as it grows, because once you can afford to treat almost everyone worth treating, ranking stops mattering.

## Run it

\`\`\`bash
pip install -r requirements.txt
python uplift_retention.py
\`\`\`

Deterministic (fixed seeds); the numbers above reproduce exactly.

## Caveats

- **The numbers are simulated**, by construction. They demonstrate that the *method* works and what the *decision* gains; they are not estimates about any real workforce.
- **Randomization does the heavy lifting.** On observational HR data you would not get a clean treated/untreated comparison, and the honest version of this analysis would lead with confounding adjustment (propensity weighting, doubly-robust estimation) and sensitivity checks. Without a design that supports it, uplift is an assumption, not a measurement.
- **One intervention, one outcome.** Real retention programs are bundles of actions with effects that drift over time; a single `tau` is a simplification.

## License

MIT — see [LICENSE](LICENSE).
