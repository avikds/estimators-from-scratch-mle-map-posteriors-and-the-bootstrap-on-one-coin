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
- [x] **10.** beta_posterior
- [x] **11.** log_evidence
- [x] **12.** sequential_predictive
- [x] **13.** predictive_counts
- [x] **14.** beta_cdf
- [x] **15.** mixture_posterior
- [x] **16.** beta_quantile
- [x] **17.** credible_intervals
- [x] **18.** hdi_from_samples
- [x] **19.** grid_posterior
- [x] **20.** laplace_theta
- [x] **21.** laplace_logit
- [x] **22.** metropolis
- [x] **23.** density_tv
- [x] **24.** sample_tv
- [x] **25.** bootstrap_mle
- [x] **26.** sampling_distribution
- [x] **27.** wald_interval
- [x] **28.** interval_coverage
- [x] **29.** bias_variance

## Results

```
First 10 flips from the table log: N1=8 heads, N0=2 tails
  MLE N1/N = 0.8000   golden-section minimum of the NLL = 0.8000
500 tosses onto the felt (tails, heads, edge): counts [245, 253, 2]  MLE [0.49, 0.506, 0.004]
8 weighings of the coin: mean 5.6610 g, variance MLE 0.000707, unbiased 0.000808
  over 20000 repeats E[var MLE] / sigma^2 = 0.8729  (theory (N-1)/N = 0.8750)
Predicting 20 future flips after n observed ones (400 repeats, true theta 0.7), log loss in nats per flip:
  n=  1  MLE infinite loss in 100.0% of repeats   add-one 0.7023   oracle 0.6166
  n=  3  MLE infinite loss in  37.2% of repeats   add-one 0.6801   oracle 0.6100
  n= 10  MLE infinite loss in   4.0% of repeats   add-one 0.6509   oracle 0.6139
  n= 30  MLE infinite loss in   0.0% of repeats   add-one 0.6244   oracle 0.6105
Beta(1,1) prior -> Beta(9,3): mean 0.7500  mode 0.8000  sd 0.1201  prior weight 0.1667
Beta(2,2) prior -> Beta(10,4): mean 0.7143  mode 0.7500  sd 0.1166  prior weight 0.2857
Sum of log one-step-ahead predictions -6.204558 = log evidence -6.204558
After 4 heads and 1 tail, 10 more flips: P(10 heads) Bayes 0.1250 vs plug-in 0.1074;  P(at most 5 heads) Bayes 0.2150 vs plug-in 0.0328
Casino prior, 20 heads 10 tails: weights [0.3458, 0.6542]  Pr(theta > 0.5) 0.9604  (single Beta(20,20) prior: 0.8858)
Beta(3,9): central 95% (0.0602, 0.5178)  HDI (0.0406, 0.4837)
Seized coin after 10 flips: central 95% (0.4822, 0.9398)  HDI (0.5163, 0.9594)
10 heads, 1 tail, uniform prior: exact mean 0.8462, 20-point grid mean 0.8415
  total variation to the exact posterior: Laplace in theta 0.2014   Laplace in logit 0.0807   Metropolis 10000 samples 0.0302 (acceptance 0.46)
  log evidence: exact -4.8828  Laplace theta -4.8776  Laplace logit -4.9253
  95% HDI: exact (0.6585, 0.9945)  from Metropolis samples (0.6531, 0.9902)
N= 10, MLE 0.80: bootstrap mean 0.7997 sd 0.1269 | posterior samples mean 0.7514 sd 0.1199
N=100, MLE 0.75: bootstrap mean 0.7496 sd 0.0432 | posterior samples mean 0.7449 sd 0.0432
Unique flips in a bootstrap resample of 100: 0.6346  (1 - 1/e = 0.6321)
Exact coverage of nominal 95% intervals for N=10 flips:
  theta=0.05  Wald 0.4002   Jeffreys 0.9885   uniform-prior credible 0.9139
  theta=0.10  Wald 0.6497   Jeffreys 0.9872   uniform-prior credible 0.9298
  theta=0.30  Wald 0.8401   Jeffreys 0.9244   uniform-prior credible 0.9244
  theta=0.50  Wald 0.8906   Jeffreys 0.9785   uniform-prior credible 0.9785
  theta=0.70  Wald 0.8401   Jeffreys 0.9244   uniform-prior credible 0.9244
  N=100, averaged over theta in 0.01..0.99: Wald 0.9269   Jeffreys 0.9496
Mean squared error at N=10: MLE vs posterior mean (uniform prior) vs posterior mean (Beta(10,10)):
  theta=0.05  MLE 0.0047   uniform 0.0089   Beta(10,10) 0.0905
  theta=0.30  MLE 0.0210   uniform 0.0157   Beta(10,10) 0.0201
  theta=0.50  MLE 0.0250   uniform 0.0174   Beta(10,10) 0.0028
  theta=0.70  MLE 0.0210   uniform 0.0157   Beta(10,10) 0.0201
  theta=0.95  MLE 0.0048   uniform 0.0089   Beta(10,10) 0.0905
Verdict on the seized coin under the casino prior 0.5 Beta(20,20) + 0.5 Beta(30,10):
   10 flips,  8 heads: weight on 'biased' 0.8200  Pr(theta > 0.5) 0.9647  95% HDI (uniform prior) (0.5163, 0.9594)  Wald (0.5521, 1.0479)
   30 flips, 24 heads: weight on 'biased' 0.9649  Pr(theta > 0.5) 0.9995  95% HDI (uniform prior) (0.6388, 0.9137)  Wald (0.6569, 0.9431)
  100 flips, 75 heads: weight on 'biased' 0.9798  Pr(theta > 0.5) 1.0000  95% HDI (uniform prior) (0.6601, 0.8274)  Wald (0.6651, 0.8349)
```
