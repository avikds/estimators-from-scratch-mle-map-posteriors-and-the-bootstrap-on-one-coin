# Estimators from Scratch: MLE, MAP, Posteriors and the Bootstrap on One Coin

A gaming regulator has seized the coin from a casino's coin-toss table, along with the table's log of flips. Run every estimator from chapter 4 of Murphy's Probabilistic Machine Learning on that one coin, in NumPy and the standard library only. Write the likelihood and find the MLE by hand and by golden-section search, watch the MLE assign probability zero to outcomes it has not seen yet, and fix it with a Beta prior and add-one smoothing. Build the conjugate beta-Bernoulli posterior with its predictive distribution, marginal likelihood and a mixture-of-Betas casino prior, then compute central and highest-density credible intervals from your own incomplete-beta function. Approximate the posterior with a grid, two Laplace approximations and a Metropolis sampler and measure how far each is from the truth. Finish on the frequentist side: the bootstrap as a poor man's posterior, the exact coverage of Wald and Jeffreys intervals, and the bias-variance tradeoff of shrinkage estimators, before you deliver the verdict on the coin.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** coin_counts
- [x] **2.** bernoulli_nll
- [x] **3.** golden_section_min
- [x] **4.** categorical_mle
- [x] **5.** gaussian_mle
- [x] **6.** map_bernoulli
- [x] **7.** log_loss
- [x] **8.** black_swan_risk
- [x] **9.** beta_logpdf

---

Built on Deep-ML.
