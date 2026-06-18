"""
Uplift modeling for employee retention — a method demonstration on semi-synthetic data.

THE QUESTION
------------
A flight-risk model predicts WHO WILL LEAVE. A retention decision needs to know WHO WILL
STAY BECAUSE OF WHAT YOU DO — the causal effect of an intervention on an individual's
choice to stay (their "uplift"). A risk score does not contain that, and accuracy will not
put it there. See the companion explainer: "Predicting who leaves isn't knowing who to keep."

WHY SEMI-SYNTHETIC
------------------
Estimating uplift requires variation in who was treated; *validating* the estimate requires
knowing the true per-person effect. No public HR dataset carries a randomized retention
intervention, so we simulate one: realistic covariates, a randomized treatment, and a known
heterogeneous effect tau(x). We then fit an uplift model that never sees tau and check (a)
whether it recovers tau, and (b) whether targeting by estimated uplift beats targeting by
predicted flight risk. Randomization is what makes the effect identifiable here; observational
data would require confounding adjustment. That caveat is the point, not a footnote.

METHOD
------
T-learner (two-model): a gradient-boosted model of P(leave) on the control group and another
on the treated group; estimated uplift = P(leave | control) - P(leave | treated).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

RNG = np.random.default_rng(7)
N = 8000
sigmoid = lambda z: 1 / (1 + np.exp(-z))


def make_workforce(rng=RNG, n=N):
    """Realistic-ish HR covariates, a TRUE baseline leave risk p0(x), and a TRUE
    heterogeneous uplift tau(x). Risk is driven partly by an UNFIXABLE cause (long commute,
    a relocation proxy); uplift comes only from FIXABLE causes the intervention addresses
    (pay gap, overtime, weak manager). So high risk != high uplift, by construction."""
    satisfaction    = rng.beta(5, 4, n)
    tenure          = rng.gamma(2, 2, n).clip(0, 25)
    overtime        = (rng.random(n) < 0.28).astype(int)
    underpaid       = rng.beta(2, 5, n)                 # higher = further below market
    commute_long    = (rng.random(n) < 0.25).astype(int)
    manager_quality = rng.beta(5, 3, n)
    performance     = rng.beta(5, 3, n)
    age             = rng.normal(38, 9, n).clip(21, 64)
    X = pd.DataFrame(dict(satisfaction=satisfaction, tenure=tenure, overtime=overtime,
                          underpaid=underpaid, commute_long=commute_long,
                          manager_quality=manager_quality, performance=performance, age=age))

    p0 = sigmoid(-0.4 - 2.1*(satisfaction-0.5) + 1.0*overtime + 1.7*(underpaid-0.3)
                 + 1.9*commute_long - 1.4*(manager_quality-0.5) - 0.10*tenure)

    fixable = 1.5*underpaid + 0.5*overtime + 1.0*(1-manager_quality) - 0.85
    tau = 0.7 * np.clip(fixable, 0, None)
    sleeping = ((satisfaction > 0.7) & (performance > 0.7)).astype(int)  # content stars: outreach backfires
    tau = np.clip(tau - 0.18*sleeping, -0.25, 0.7)
    return X, p0, tau


def main():
    X, p0, tau = make_workforce()
    # randomized intervention; observed outcome (1 = left)
    T = (RNG.random(N) < 0.5).astype(int)
    y = (RNG.random(N) < np.clip(p0 - T*tau, 0.01, 0.99)).astype(int)

    tr, te = train_test_split(np.arange(N), test_size=0.4, random_state=1)

    # --- T-learner: never sees tau ---
    m0 = GradientBoostingClassifier(random_state=0).fit(X.iloc[tr][T[tr]==0], y[tr][T[tr]==0])
    m1 = GradientBoostingClassifier(random_state=0).fit(X.iloc[tr][T[tr]==1], y[tr][T[tr]==1])
    risk_hat   = m0.predict_proba(X.iloc[te])[:, 1]               # baseline P(leave) = flight risk
    uplift_hat = risk_hat - m1.predict_proba(X.iloc[te])[:, 1]    # estimated reduction in P(leave)
    tau_te = tau[te]

    print("=== Did the model recover the true effect? ===")
    print(f"corr(estimated uplift, TRUE uplift) = {pearsonr(uplift_hat, tau_te)[0]:.3f}")
    print(f"corr(TRUE uplift, baseline risk)    = {pearsonr(tau_te, p0[te])[0]:.3f}   "
          f"(modest -> who's at risk is NOT who you can move)")

    print("\n=== Policy comparison: expected TRUE stays saved at each budget ===")
    print(f"{'budget':>7} | {'by risk':>8} | {'by uplift':>9} | {'uplift / risk':>13}")
    order_risk, order_up = np.argsort(-risk_hat), np.argsort(-uplift_hat)
    for frac in (0.05, 0.10, 0.20, 0.40):
        k = int(len(te) * frac)
        sr = tau_te[order_risk[:k]].sum()
        su = tau_te[order_up[:k]].sum()
        print(f"{int(frac*100):>6}% | {sr:8.1f} | {su:9.1f} | {su/sr:>11.2f}x")
    k = int(len(te)*0.10)
    print(f"\n(random targeting @10% saves {tau_te[RNG.permutation(len(te))[:k]].sum():.1f})")

    print("\n=== The four types in this workforce (by true risk & true uplift) ===")
    hi_risk = p0[te] >= np.median(p0[te])
    hi_up   = tau_te >= 0.10
    seg = np.where(tau_te < 0, "sleeping dog",
          np.where(hi_risk & hi_up, "persuadable",
          np.where(hi_risk & ~hi_up, "lost cause",
          np.where(~hi_risk & ~hi_up, "sure thing", "persuadable (low-risk)"))))
    for name, cnt in pd.Series(seg).value_counts().items():
        print(f"  {name:<22} {cnt:>5}  ({cnt/len(te)*100:4.1f}%)")


if __name__ == "__main__":
    main()
