"""
Estimators from Scratch: MLE, MAP, Posteriors and the Bootstrap on One Coin scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Estimators from Scratch: MLE, MAP, Posteriors and the Bootstrap on One Coin.
A gaming regulator has seized the coin from a casino's coin-toss table. Every estimator in chapter 4 of
Murphy's Probabilistic Machine Learning (sections 4.2 and 4.5-4.7) is run on that one coin."""
import math
import numpy as np

THETA_TRUE = 0.7  # hidden from the inspector: used only to simulate flips and to score the estimators


def jeffreys_table(N):
    """95% central credible interval of Beta(N1 + 1/2, N0 + 1/2) for every possible head count."""
    return {k: (beta_quantile(0.025, k + 0.5, N - k + 0.5), beta_quantile(0.975, k + 0.5, N - k + 0.5))
            for k in range(N + 1)}


def main():
    rng = np.random.default_rng(2026)
    table_log = (rng.random(100) < THETA_TRUE).astype(int)
    first10 = table_log[:10]

    # --- Part 1: the likelihood and the MLE ---
    N1, N0 = coin_counts(first10)
    print(f"First 10 flips from the table log: N1={N1} heads, N0={N0} tails")
    print(f"  MLE N1/N = {N1 / (N1 + N0):.4f}   golden-section minimum of the NLL = "
          f"{golden_section_min(lambda t: bernoulli_nll(t, N1, N0), 0.0, 1.0):.4f}")
    landings = rng.choice(3, size=500, p=[0.4985, 0.4985, 0.003])
    counts, theta_cat = categorical_mle(landings, 3)
    print(f"500 tosses onto the felt (tails, heads, edge): counts {counts.tolist()}  MLE {np.round(theta_cat, 4).tolist()}")
    weighings = 5.67 + 0.05 * rng.standard_normal(8)
    mu, var_mle, var_unb = gaussian_mle(weighings)
    ratio = np.mean([gaussian_mle(5.67 + 0.05 * rng.standard_normal(8))[1] for _ in range(20000)]) / 0.05 ** 2
    print(f"8 weighings of the coin: mean {mu:.4f} g, variance MLE {var_mle:.6f}, unbiased {var_unb:.6f}")
    print(f"  over 20000 repeats E[var MLE] / sigma^2 = {ratio:.4f}  (theory (N-1)/N = {7 / 8:.4f})")

    # --- Part 2: the zero-count problem and add-one smoothing ---
    print("Predicting 20 future flips after n observed ones (400 repeats, true theta 0.7), log loss in nats per flip:")
    for n in (1, 3, 10, 30):
        r = black_swan_risk(THETA_TRUE, n, 20, 400, np.random.default_rng(n))
        print(f"  n={n:3d}  MLE infinite loss in {100 * r['mle_infinite']:5.1f}% of repeats   add-one {r['add_one_loss']:.4f}"
              f"   oracle {r['oracle_loss']:.4f}")

    # --- Part 3: the beta-Bernoulli posterior ---
    for a, b in ((1, 1), (2, 2)):
        post = beta_posterior(N1, N0, a, b)
        print(f"Beta({a},{b}) prior -> Beta({post['a']},{post['b']}): mean {post['mean']:.4f}  mode {post['mode']:.4f}  "
              f"sd {math.sqrt(post['var']):.4f}  prior weight {post['prior_weight']:.4f}")
    p_seq = sequential_predictive(first10, 1, 1)
    chain = float(np.sum(np.where(first10 == 1, np.log(p_seq), np.log(1 - p_seq))))
    print(f"Sum of log one-step-ahead predictions {chain:.6f} = log evidence {log_evidence(N1, N0, 1, 1):.6f}")
    bayes, plugin = predictive_counts(10, 4, 1, 1, 1)
    print(f"After 4 heads and 1 tail, 10 more flips: P(10 heads) Bayes {bayes[10]:.4f} vs plug-in {plugin[10]:.4f};"
          f"  P(at most 5 heads) Bayes {bayes[:6].sum():.4f} vs plug-in {plugin[:6].sum():.4f}")
    casino = ([0.5, 0.5], [(20, 20), (30, 10)])
    w, comps = mixture_posterior(*casino, 20, 10)
    p_biased = sum(wk * (1 - beta_cdf(0.5, a, b)) for wk, (a, b) in zip(w, comps))
    print(f"Casino prior, 20 heads 10 tails: weights {np.round(w, 4).tolist()}  Pr(theta > 0.5) {p_biased:.4f}"
          f"  (single Beta(20,20) prior: {1 - beta_cdf(0.5, 40, 30):.4f})")

    # --- Part 4: credible intervals ---
    ci = credible_intervals(3, 9)
    print(f"Beta(3,9): central 95% {tuple(round(v, 4) for v in ci['central'])}  HDI {tuple(round(v, 4) for v in ci['hdi'])}")
    ci = credible_intervals(N1 + 1, N0 + 1)
    print(f"Seized coin after 10 flips: central 95% {tuple(round(v, 4) for v in ci['central'])}"
          f"  HDI {tuple(round(v, 4) for v in ci['hdi'])}")

    # --- Part 5: approximate inference on 10 heads and 1 tail (Figure 4.22) ---
    exact_mean = 11 / 13
    grid, probs = grid_posterior(10, 1, 1, 1, 20)
    print(f"10 heads, 1 tail, uniform prior: exact mean {exact_mean:.4f}, 20-point grid mean {float(grid @ probs):.4f}")
    m, sd, logz_t = laplace_theta(10, 1, 1, 1)
    am, asd, logz_a = laplace_logit(10, 1, 1, 1)
    gauss_theta = lambda t: -0.5 * ((t - m) / sd) ** 2 - math.log(sd * math.sqrt(2 * math.pi))

    def gauss_logit(t):
        al = np.log(t) - np.log1p(-t)
        return -0.5 * ((al - am) / asd) ** 2 - math.log(asd * math.sqrt(2 * math.pi)) - np.log(t) - np.log1p(-t)

    log_target = lambda t: (-bernoulli_nll(t, 10, 1) + beta_logpdf(t, 1, 1)) if 0 < t < 1 else -math.inf
    samples, acc = metropolis(log_target, 0.5, 12000, 0.2, np.random.default_rng(7))
    print(f"  total variation to the exact posterior: Laplace in theta {density_tv(gauss_theta, 10, 1, 1, 1):.4f}"
          f"   Laplace in logit {density_tv(gauss_logit, 10, 1, 1, 1):.4f}"
          f"   Metropolis 10000 samples {sample_tv(samples[2000:], 10, 1, 1, 1):.4f} (acceptance {acc:.2f})")
    print(f"  log evidence: exact {log_evidence(10, 1, 1, 1):.4f}  Laplace theta {logz_t:.4f}  Laplace logit {logz_a:.4f}")
    print(f"  95% HDI: exact {tuple(round(v, 4) for v in credible_intervals(11, 2)['hdi'])}"
          f"  from Metropolis samples {tuple(round(v, 4) for v in hdi_from_samples(samples[2000:]))}")

    # --- Part 6: the frequentist view ---
    for n in (10, 100):
        y = table_log[:n]
        k1, k0 = coin_counts(y)
        boot = bootstrap_mle(y, 10000, np.random.default_rng(11))
        post = np.random.default_rng(12).beta(1 + k1, 1 + k0, 10000)
        print(f"N={n:3d}, MLE {k1 / n:.2f}: bootstrap mean {boot.mean():.4f} sd {boot.std():.4f}"
              f" | posterior samples mean {post.mean():.4f} sd {post.std():.4f}")
    idx = np.random.default_rng(13).integers(0, 100, size=(2000, 100))
    uniq = np.mean([np.unique(r).size for r in idx]) / 100
    print(f"Unique flips in a bootstrap resample of 100: {uniq:.4f}  (1 - 1/e = {1 - math.exp(-1):.4f})")
    print("Exact coverage of nominal 95% intervals for N=10 flips:")
    jt = jeffreys_table(10)
    uniform = {k: credible_intervals(k + 1, 10 - k + 1)["central"] for k in range(11)}
    for theta in (0.05, 0.1, 0.3, 0.5, 0.7):
        cw = interval_coverage(wald_interval, theta, 10)
        cj = interval_coverage(lambda a, b: jt[a], theta, 10)
        cu = interval_coverage(lambda a, b: uniform[a], theta, 10)
        print(f"  theta={theta:.2f}  Wald {cw:.4f}   Jeffreys {cj:.4f}   uniform-prior credible {cu:.4f}")
    thetas = np.linspace(0.01, 0.99, 99)
    jt100 = jeffreys_table(100)
    avg_w = np.mean([interval_coverage(wald_interval, t, 100) for t in thetas])
    avg_j = np.mean([interval_coverage(lambda a, b: jt100[a], t, 100) for t in thetas])
    print(f"  N=100, averaged over theta in 0.01..0.99: Wald {avg_w:.4f}   Jeffreys {avg_j:.4f}")
    print("Mean squared error at N=10: MLE vs posterior mean (uniform prior) vs posterior mean (Beta(10,10)):")
    for theta in (0.05, 0.3, 0.5, 0.7, 0.95):
        row = [bias_variance(est, theta, 10)[2] for est in
               (lambda a, b: a / (a + b), lambda a, b: (a + 1) / (a + b + 2), lambda a, b: (a + 10) / (a + b + 20))]
        print(f"  theta={theta:.2f}  MLE {row[0]:.4f}   uniform {row[1]:.4f}   Beta(10,10) {row[2]:.4f}")

    # --- Verdict: the casino prior updated as the table log grows ---
    print("Verdict on the seized coin under the casino prior 0.5 Beta(20,20) + 0.5 Beta(30,10):")
    for n in (10, 30, 100):
        k1, k0 = coin_counts(table_log[:n])
        w, comps = mixture_posterior(*casino, k1, k0)
        p_biased = sum(wk * (1 - beta_cdf(0.5, a, b)) for wk, (a, b) in zip(w, comps))
        hdi = credible_intervals(k1 + 1, k0 + 1)["hdi"]
        lo, hi = wald_interval(k1, k0)
        print(f"  {n:3d} flips, {k1:2d} heads: weight on 'biased' {w[1]:.4f}  Pr(theta > 0.5) {p_biased:.4f}"
              f"  95% HDI (uniform prior) ({hdi[0]:.4f}, {hdi[1]:.4f})  Wald ({lo:.4f}, {hi:.4f})")


if __name__ == "__main__":
    main()

