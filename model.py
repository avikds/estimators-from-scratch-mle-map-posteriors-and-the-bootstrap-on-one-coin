"""
Estimators from Scratch: MLE, MAP, Posteriors and the Bootstrap on One Coin

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - coin_counts
import numpy as np

def coin_counts(y):
    """
    Convert a coin-flip log into sufficient statistics.

    Parameters:
        y: One-dimensional array or list containing only 0s and 1s.
           1 represents heads and 0 represents tails.

    Returns:
        Tuple (N1, N0), where:
            N1 = number of heads
            N0 = number of tails

    Raises:
        ValueError: If y is not one-dimensional or contains
                    values other than 0 and 1.
    """

    # Convert the input into a NumPy array.
    y = np.asarray(y)

    # Validate that the input is one-dimensional.
    if y.ndim != 1:
        raise ValueError("Input must be a one-dimensional array.")

    # Validate that every entry is either 0 or 1.
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("All entries must be either 0 or 1.")

    # Count heads (1) and tails (0).
    N1 = int(np.count_nonzero(y == 1))
    N0 = int(np.count_nonzero(y == 0))

    return (N1, N0)

# Step 2 - bernoulli_nll
def bernoulli_nll(theta, N1, N0):
    """
    Compute the negative log-likelihood of Bernoulli observations.

    Parameters:
        theta: Heads probability (scalar or NumPy array).
        N1: Number of heads.
        N0: Number of tails.

    Returns:
        Python float for scalar theta.
        NumPy array for array-valued theta.
    """

    # Convert theta to a NumPy array for vectorized calculations.
    theta = np.asarray(theta, dtype=float)

    # Initialize the negative log-likelihood with zeros.
    nll = np.zeros_like(theta, dtype=float)

    # Calculate the heads contribution.
    # Skip the term when N1 = 0 to implement 0 * log(0) = 0.
    if N1 != 0:
        with np.errstate(divide='ignore', invalid='ignore'):
            nll -= N1 * np.log(theta)

    # Calculate the tails contribution.
    # Skip the term when N0 = 0 to implement 0 * log(0) = 0.
    if N0 != 0:
        with np.errstate(divide='ignore', invalid='ignore'):
            nll -= N0 * np.log1p(-theta)

    # Return a Python float for scalar input.
    if theta.ndim == 0:
        return float(nll)

    return nll

# Step 3 - golden_section_min
import math

def golden_section_min(f, lo, hi, tol=1e-10):
    """
    Find the minimum of a unimodal function using golden-section search.

    Parameters:
        f: Objective function to minimize.
        lo: Lower bound of the search interval.
        hi: Upper bound of the search interval.
        tol: Stopping tolerance (default: 1e-10).

    Returns:
        Python float representing the midpoint of the final interval.
    """

    # Golden-section ratio.
    r = (math.sqrt(5.0) - 1.0) / 2.0

    # Initialize the two interior points.
    c = hi - r * (hi - lo)
    d = lo + r * (hi - lo)

    # Evaluate the function at both interior points.
    fc = f(c)
    fd = f(d)

    # Continue until the interval is sufficiently small.
    while (hi - lo) > tol:

        if fc < fd:
            # The minimum lies in [lo, d].
            hi = d

            # Reuse the previous evaluation at c.
            d = c
            fd = fc

            # Calculate and evaluate the new interior point.
            c = hi - r * (hi - lo)
            fc = f(c)

        else:
            # The minimum lies in [c, hi].
            lo = c

            # Reuse the previous evaluation at d.
            c = d
            fc = fd

            # Calculate and evaluate the new interior point.
            d = lo + r * (hi - lo)
            fd = f(d)

    # Return the midpoint of the final search interval.
    return float((lo + hi) / 2.0)

# Step 4 - categorical_mle
def categorical_mle(y, K):
    """
    Estimate categorical probabilities using maximum likelihood estimation.

    Parameters:
        y: One-dimensional integer array of observed outcomes (0 to K-1).
        K: Total number of possible outcomes.

    Returns:
        counts: Integer NumPy array containing the count of each outcome.
        theta: Float NumPy array containing the MLE probabilities.

    Raises:
        ValueError: If y is empty, not one-dimensional, contains
                    non-integer values, or has outcomes outside 0 to K-1.
                    Also raised if K is not a positive integer.
    """

    # Validate K.
    if not isinstance(K, (int, np.integer)) or isinstance(K, (bool, np.bool_)) or K <= 0:
        raise ValueError("K must be a positive integer.")

    K = int(K)

    # Convert the observations into a NumPy array.
    y = np.asarray(y)

    # Ensure that y is one-dimensional.
    if y.ndim != 1:
        raise ValueError("Input y must be a one-dimensional array.")

    # Reject an empty observation array.
    if y.size == 0:
        raise ValueError("Input y must not be empty.")

    # Ensure that all observations are integers.
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("All outcomes must be integers.")

    # Ensure that every observation belongs to the range 0 to K-1.
    if np.any(y < 0) or np.any(y >= K):
        raise ValueError("Outcomes must be between 0 and K-1.")

    # Count every possible outcome, including outcomes never observed.
    counts = np.bincount(y, minlength=K)

    # Calculate maximum likelihood probabilities.
    theta = counts.astype(float) / y.size

    return counts, theta

# Step 5 - gaussian_mle
def gaussian_mle(x):
    """
    Fit a Gaussian distribution to repeated measurements using MLE.

    Parameters:
        x: Array or list of coin-weight measurements.

    Returns:
        Tuple (mu, var_mle, var_unbiased), containing Python floats.

        mu: MLE of the mean.
        var_mle: MLE of the variance (division by N).
        var_unbiased: Unbiased variance estimate (division by N-1).

    Raises:
        ValueError: If fewer than two measurements are provided.
    """

    # Convert measurements into a NumPy array of floats.
    x = np.asarray(x, dtype=float)

    # Check that at least two measurements are available.
    N = x.size
    if N < 2:
        raise ValueError("At least two measurements are required.")

    # Calculate the sufficient statistics.
    mean_x = np.mean(x)
    mean_x_squared = np.mean(x ** 2)

    # MLE of the Gaussian mean.
    mu = float(mean_x)

    # MLE of the variance using E[X^2] - E[X]^2.
    # Clip tiny negative values caused by floating-point roundoff.
    var_mle = float(max(0.0, mean_x_squared - mean_x ** 2))

    # Unbiased variance estimate (Bessel's correction).
    var_unbiased = float(var_mle * N / (N - 1))

    return (mu, var_mle, var_unbiased)

# Step 6 - map_bernoulli
def map_bernoulli(N1, N0, a, b):
    """
    Calculate the MAP estimate of the Bernoulli heads probability
    using a Beta(a, b) prior.

    Parameters:
        N1: Number of observed heads.
        N0: Number of observed tails.
        a: Positive first Beta prior parameter.
        b: Positive second Beta prior parameter.

    Returns:
        Python float representing the posterior mode.

    Raises:
        ValueError: If prior parameters are not positive or if
                    the posterior does not have a well-defined mode.
    """

    # Validate that the prior parameters are positive and finite.
    if not all(math.isfinite(v) for v in (a, b)) or a <= 0 or b <= 0:
        raise ValueError("Beta prior parameters a and b must be positive and finite.")

    # Validate the observed counts.
    if not all(math.isfinite(v) for v in (N1, N0)) or N1 < 0 or N0 < 0:
        raise ValueError("N1 and N0 must be non-negative and finite.")

    # Calculate the posterior Beta distribution parameters.
    alpha = N1 + a
    beta = N0 + b

    # Check whether the posterior has a well-defined mode in [0, 1].
    if alpha < 1 or beta < 1 or alpha + beta <= 2:
        raise ValueError("The posterior Beta distribution has no well-defined mode.")

    # Calculate the MAP estimate (posterior mode).
    theta_map = (alpha - 1) / (alpha + beta - 2)

    return float(theta_map)

# Step 7 - log_loss
def log_loss(p_heads, y):
    """
    Calculate the average negative log-likelihood (log loss)
    of observed coin flips under a predicted heads probability.

    Parameters:
        p_heads: Predicted probability of heads.
        y: One-dimensional array or list of future coin flips
           (1 represents heads and 0 represents tails).

    Returns:
        Python float representing average log loss in nats per flip.

    Raises:
        ValueError: If y contains invalid flips or is empty.
    """

    # Validate the coin-flip observations and obtain sufficient statistics.
    N1, N0 = coin_counts(y)

    # Calculate the total number of observed flips.
    N = N1 + N0

    # Log loss is undefined when there are no observations.
    if N == 0:
        raise ValueError("Cannot calculate log loss for an empty flip sequence.")

    # Calculate the total negative log-likelihood using the earlier function.
    total_nll = bernoulli_nll(p_heads, N1, N0)

    # Divide by the number of flips to obtain average log loss.
    return float(total_nll / N)

# Step 8 - black_swan_risk
def black_swan_risk(theta, n, n_future, n_rep, rng):
    """
    Measure the risk of MLE and add-one smoothing through simulation.

    Parameters:
        theta: True probability of heads.
        n: Number of observed coin flips.
        n_future: Number of future coin flips.
        n_rep: Number of simulation repetitions.
        rng: NumPy random number generator.

    Returns:
        Dictionary containing:
            mle_infinite: Fraction of repetitions with infinite MLE loss.
            mle_loss_when_finite: Mean MLE loss over finite-loss repetitions.
            add_one_loss: Mean loss using add-one smoothing.
            oracle_loss: Mean loss using the true probability.
    """

    # Store finite MLE losses separately.
    mle_losses_finite = []

    # Store losses from add-one smoothing and the oracle.
    add_one_losses = []
    oracle_losses = []

    # Count repetitions in which MLE produces infinite loss.
    mle_infinite_count = 0

    # Run the simulation.
    for _ in range(n_rep):

        # Step 1: Generate the observed coin flips.
        y = (rng.random(n) < theta).astype(int)

        # Step 2: Generate future coin flips.
        future = (rng.random(n_future) < theta).astype(int)

        # Step 3: Obtain the sufficient statistics.
        N1, N0 = coin_counts(y)

        # Calculate the MLE estimate of the heads probability.
        p_mle = N1 / n

        # Evaluate the MLE prediction on future observations.
        mle_loss = log_loss(p_mle, future)

        # Record whether the MLE prediction has infinite loss.
        if math.isinf(mle_loss):
            mle_infinite_count += 1
        else:
            mle_losses_finite.append(mle_loss)

        # Calculate the add-one smoothed estimate using Beta(2, 2).
        p_add_one = map_bernoulli(N1, N0, 2, 2)

        # Evaluate the add-one smoothed prediction.
        add_one_losses.append(log_loss(p_add_one, future))

        # Evaluate the oracle prediction using the true theta.
        oracle_losses.append(log_loss(theta, future))

    # Calculate the fraction of repetitions with infinite MLE loss.
    mle_infinite = float(mle_infinite_count / n_rep)

    # Calculate the mean MLE loss over finite-loss repetitions.
    # Return NaN if every MLE loss was infinite.
    if mle_losses_finite:
        mle_loss_when_finite = float(np.mean(mle_losses_finite))
    else:
        mle_loss_when_finite = float("nan")

    # Calculate the average add-one smoothing loss.
    add_one_loss = float(np.mean(add_one_losses))

    # Calculate the average oracle loss.
    oracle_loss = float(np.mean(oracle_losses))

    # Return the simulation results.
    return {
        "mle_infinite": mle_infinite,
        "mle_loss_when_finite": mle_loss_when_finite,
        "add_one_loss": add_one_loss,
        "oracle_loss": oracle_loss
    }

# Step 9 - beta_logpdf
def log_beta(a, b):
    """
    Calculate the logarithm of the Beta function.

    Parameters:
        a: First positive Beta distribution parameter.
        b: Second positive Beta distribution parameter.

    Returns:
        Python float representing log(B(a, b)).
    """

    # Calculate log Beta using the log Gamma function
    # to avoid numerical overflow.
    return float(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))


def beta_logpdf(theta, a, b):
    """
    Calculate the log probability density of a Beta distribution.

    Parameters:
        theta: Probability value (scalar or NumPy array).
        a: First positive Beta distribution parameter.
        b: Second positive Beta distribution parameter.

    Returns:
        Python float for scalar theta.
        NumPy array for array-valued theta.

    Notes:
        - Uses the convention 0 * log(0) = 0.
        - Returns -inf outside the interval [0, 1].
        - Suppresses divide-by-zero and invalid-operation warnings.
    """

    # Convert theta into a NumPy array of floats.
    theta = np.asarray(theta, dtype=float)

    # Identify values inside the valid probability interval.
    inside = (theta >= 0.0) & (theta <= 1.0)

    # Replace invalid values temporarily to avoid unnecessary
    # logarithm warnings. These values will ultimately become -inf.
    safe_theta = np.where(inside, theta, 0.5)

    # Initialize the log-density with zeros.
    log_density = np.zeros_like(theta, dtype=float)

    with np.errstate(divide='ignore', invalid='ignore'):

        # First term: (a - 1) * log(theta).
        # Skip it when a == 1 to correctly implement 0 * log(0) = 0.
        if a != 1:
            log_density += (a - 1) * np.log(safe_theta)

        # Second term: (b - 1) * log(1 - theta).
        # Skip it when b == 1 for the same boundary convention.
        if b != 1:
            log_density += (b - 1) * np.log1p(-safe_theta)

    # Subtract the log Beta normalization constant.
    log_density -= log_beta(a, b)

    # Assign negative infinity to values outside [0, 1].
    log_density = np.where(inside, log_density, -np.inf)

    # Return a Python float for scalar input.
    if theta.ndim == 0:
        return float(log_density)

    return log_density

# Step 10 - beta_posterior
def beta_posterior(N1, N0, a, b):
    """
    Update a Beta(a, b) prior using Bernoulli observations
    and summarize the resulting posterior distribution.

    Parameters:
        N1: Number of observed heads.
        N0: Number of observed tails.
        a: First parameter of the Beta prior.
        b: Second parameter of the Beta prior.

    Returns:
        Dictionary containing:
            a: Updated first posterior parameter.
            b: Updated second posterior parameter.
            mean: Posterior mean.
            mode: Posterior mode (MAP estimate), or None if undefined.
            var: Posterior variance.
            prior_weight: Weight assigned to the prior mean.
    """

    # Update the Beta parameters using the observed counts.
    alpha = a + N1
    beta = b + N0

    # Total posterior concentration.
    total = alpha + beta

    # Calculate the posterior mean.
    posterior_mean = float(alpha / total)

    # Calculate the posterior mode using the previously
    # implemented MAP estimator.
    try:
        posterior_mode = map_bernoulli(N1, N0, a, b)
    except ValueError:
        posterior_mode = None

    # Calculate the posterior variance.
    posterior_var = float(
        (alpha * beta) / (total ** 2 * (total + 1))
    )

    # Calculate the weight assigned to the prior mean
    # in the posterior mean.
    prior_weight = float(
        (a + b) / (a + b + N1 + N0)
    )

    # Return the posterior summary.
    return {
        "a": alpha,
        "b": beta,
        "mean": posterior_mean,
        "mode": posterior_mode,
        "var": posterior_var,
        "prior_weight": prior_weight
    }

# Step 11 - log_evidence
def log_evidence(N1, N0, a, b, binomial=False):
    """
    Calculate the log marginal likelihood of coin flips
    under a Beta(a, b) prior.

    Parameters:
        N1: Number of observed heads.
        N0: Number of observed tails.
        a: First Beta prior parameter.
        b: Second Beta prior parameter.
        binomial: If True, include the binomial coefficient.

    Returns:
        Python float representing the log marginal likelihood.

    Notes:
        - Uses the previously implemented log_beta() function.
        - Uses math.lgamma() for numerical stability.
        - binomial=False evaluates a specific sequence.
        - binomial=True evaluates the probability of the counts.
    """

    # Total number of observed flips.
    N = N1 + N0

    # Calculate the log marginal likelihood for a specific sequence.
    evidence = log_beta(a + N1, b + N0) - log_beta(a, b)

    # Include the binomial coefficient when requested.
    if binomial:
        log_combination = (
            math.lgamma(N + 1)
            - math.lgamma(N1 + 1)
            - math.lgamma(N0 + 1)
        )

        evidence += log_combination

    return float(evidence)

# Step 12 - sequential_predictive
def sequential_predictive(y, a, b):
    """
    Calculate sequential one-step-ahead predictive probabilities
    for a sequence of Bernoulli coin flips.

    Parameters:
        y: One-dimensional array or list of coin flips (0 or 1).
        a: First parameter of the Beta prior.
        b: Second parameter of the Beta prior.

    Returns:
        NumPy float array where p[t] is the predictive probability
        of heads before observing the flip at position t.

    Notes:
        - Uses coin_counts() to validate the observations.
        - p[0] is the prior predictive probability.
        - Each prediction uses only previously observed flips.
        - An empty input returns an empty float array.
    """

    # Validate the coin-flip sequence using the earlier function.
    coin_counts(y)

    # Convert the validated observations into a float array.
    y = np.asarray(y, dtype=float)

    # Calculate the number of heads observed before each position.
    # Subtracting the current observation excludes it from its own prediction.
    heads_before = np.cumsum(y) - y

    # Calculate the number of observations before each position.
    t = np.arange(len(y), dtype=float)

    # Calculate the sequential posterior predictive probabilities.
    p = (a + heads_before) / (a + b + t)

    return np.asarray(p, dtype=float)

# Step 13 - predictive_counts
def predictive_counts(M, N1, N0, a, b):
    """
    Predict the number of heads in M future coin flips
    using Bayesian Beta-Binomial and plug-in Binomial models.

    Parameters:
        M: Number of future coin flips.
        N1: Number of observed heads.
        N0: Number of observed tails.
        a: First parameter of the Beta prior.
        b: Second parameter of the Beta prior.

    Returns:
        Tuple (bayes, plugin), containing two NumPy float arrays
        of length M + 1.

        bayes[x]: Beta-Binomial posterior predictive probability.
        plugin[x]: Binomial probability using the MAP estimate.
    """

    # Update the Beta prior parameters using observed data.
    alpha = a + N1
    beta = b + N0

    # Calculate the normalization constant for the
    # Beta-Binomial predictive distribution.
    log_normalizer = log_beta(alpha, beta)

    # Initialize the Bayesian predictive distribution.
    bayes = np.zeros(M + 1, dtype=float)

    # Calculate the Beta-Binomial probabilities.
    for x in range(M + 1):

        # Logarithm of the binomial coefficient C(M, x).
        log_combination = (
            math.lgamma(M + 1)
            - math.lgamma(x + 1)
            - math.lgamma(M - x + 1)
        )

        # Calculate the log Beta-Binomial probability.
        log_probability = (
            log_combination
            + log_beta(x + alpha, M - x + beta)
            - log_normalizer
        )

        # Convert log probability into probability.
        bayes[x] = math.exp(log_probability)

    # Normalize to ensure that the probabilities sum to one.
    bayes /= np.sum(bayes)

    # Calculate the MAP estimate using the previously
    # implemented MAP estimator.
    theta_map = map_bernoulli(N1, N0, a, b)

    # Initialize the plug-in Binomial distribution.
    plugin = np.zeros(M + 1, dtype=float)

    # Handle boundary probabilities separately to avoid log(0).
    if theta_map == 0.0:
        plugin[0] = 1.0

    elif theta_map == 1.0:
        plugin[M] = 1.0

    else:
        # Calculate the plug-in Binomial probabilities.
        for x in range(M + 1):

            # Logarithm of the binomial coefficient.
            log_combination = (
                math.lgamma(M + 1)
                - math.lgamma(x + 1)
                - math.lgamma(M - x + 1)
            )

            # Logarithm of the Binomial probability.
            log_probability = (
                log_combination
                + x * math.log(theta_map)
                + (M - x) * math.log1p(-theta_map)
            )

            # Convert log probability into probability.
            plugin[x] = math.exp(log_probability)

        # Normalize the plug-in probabilities.
        plugin /= np.sum(plugin)

    return bayes, plugin

# Step 14 - beta_cdf
def beta_cdf(x, a, b):
    """
    Calculate the regularized incomplete beta function I_x(a, b),
    which is the CDF of a Beta(a, b) distribution.

    Parameters:
        x: Scalar evaluation point.
        a: First positive Beta distribution parameter.
        b: Second positive Beta distribution parameter.

    Returns:
        Python float representing the Beta CDF.

    Notes:
        - Uses the continued fraction with the modified Lentz algorithm.
        - Uses symmetry to improve convergence.
        - Uses log_beta() for numerical stability.
        - Does not require SciPy.
    """

    # Convert inputs to Python floats.
    x = float(x)
    a = float(a)
    b = float(b)

    # Validate the Beta distribution parameters.
    if not (math.isfinite(a) and math.isfinite(b)) or a <= 0 or b <= 0:
        raise ValueError("Beta parameters a and b must be positive and finite.")

    if math.isnan(x):
        raise ValueError("x must not be NaN.")

    # Handle boundary and out-of-range values.
    if x <= 0.0:
        return 0.0

    if x >= 1.0:
        return 1.0

    # Constants for the modified Lentz algorithm.
    MAX_ITER = 20000
    EPS = 3e-14
    FPMIN = 1e-300

    def beta_continued_fraction(aa, bb, xx):
        """
        Evaluate the continued fraction for the incomplete
        beta function using the modified Lentz algorithm.
        """

        qab = aa + bb
        qap = aa + 1.0
        qam = aa - 1.0

        # Initialize the continued fraction.
        c = 1.0
        d = 1.0 - qab * xx / qap

        if abs(d) < FPMIN:
            d = FPMIN

        d = 1.0 / d
        h = d

        # Iteratively evaluate the continued fraction.
        for m in range(1, MAX_ITER + 1):

            m2 = 2 * m

            # First continued-fraction coefficient.
            aa_m = (
                m * (bb - m) * xx
                / ((qam + m2) * (aa + m2))
            )

            # Modified Lentz update.
            d = 1.0 + aa_m * d
            if abs(d) < FPMIN:
                d = FPMIN

            c = 1.0 + aa_m / c
            if abs(c) < FPMIN:
                c = FPMIN

            d = 1.0 / d
            h *= d * c

            # Second continued-fraction coefficient.
            aa_m = (
                -(aa + m) * (qab + m) * xx
                / ((aa + m2) * (qap + m2))
            )

            # Modified Lentz update.
            d = 1.0 + aa_m * d
            if abs(d) < FPMIN:
                d = FPMIN

            c = 1.0 + aa_m / c
            if abs(c) < FPMIN:
                c = FPMIN

            d = 1.0 / d

            # Update the continued fraction.
            delta = d * c
            h *= delta

            # Check convergence.
            if abs(delta - 1.0) < EPS:
                return h

        # Raise an error if convergence was not achieved.
        raise ArithmeticError(
            "Incomplete beta continued fraction did not converge."
        )

    # Calculate the logarithm of the beta prefactor.
    log_bt = (
        a * math.log(x)
        + b * math.log1p(-x)
        - log_beta(a, b)
    )

    bt = math.exp(log_bt)

    # Use symmetry to evaluate the continued fraction
    # in its faster-converging region.
    if x < (a + 1.0) / (a + b + 2.0):

        result = (
            bt * beta_continued_fraction(a, b, x) / a
        )

    else:

        result = (
            1.0
            - bt * beta_continued_fraction(b, a, 1.0 - x) / b
        )

    # Protect against tiny floating-point errors outside [0, 1].
    result = min(1.0, max(0.0, result))

    return float(result)

# Step 15 - mixture_posterior
def mixture_posterior(weights, components, N1, N0):
    """
    Update a mixture-of-Betas prior using observed Bernoulli data.

    Parameters:
        weights: Sequence of prior mixing weights summing to one.
        components: List of (a_k, b_k) Beta prior parameter tuples.
        N1: Number of observed heads.
        N0: Number of observed tails.

    Returns:
        Tuple (post_weights, post_components), where:

        post_weights:
            NumPy float array of posterior mixing weights.

        post_components:
            List of updated (a_k + N1, b_k + N0) tuples.
    """

    # Convert prior weights into a NumPy float array.
    weights = np.asarray(weights, dtype=float)

    # Validate the prior weights.
    if weights.ndim != 1 or len(weights) == 0:
        raise ValueError("Weights must be a non-empty one-dimensional sequence.")

    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Weights must be finite and non-negative.")

    if not np.isclose(np.sum(weights), 1.0):
        raise ValueError("Prior mixing weights must sum to one.")

    # Convert components into a list of parameter tuples.
    components = list(components)

    # Ensure that the number of weights matches the number of components.
    if len(weights) != len(components):
        raise ValueError("Weights and components must have the same length.")

    if not all(len(component) == 2 for component in components):
        raise ValueError("Each component must contain exactly two parameters.")

    # Validate the observed counts.
    if N1 < 0 or N0 < 0:
        raise ValueError("Observed counts must be non-negative.")

    # Calculate the posterior parameters and log unnormalized weights.
    post_components = []
    log_weights = []

    for weight, (a_k, b_k) in zip(weights, components):

        # Validate the Beta component parameters.
        if not all(math.isfinite(v) and v > 0 for v in (a_k, b_k)):
            raise ValueError("Beta component parameters must be positive and finite.")

        # Update the component using the observed heads and tails.
        a_post = a_k + N1
        b_post = b_k + N0

        post_components.append((a_post, b_post))

        # Calculate the log prior weight.
        # A zero prior weight corresponds to negative infinity.
        if weight == 0.0:
            log_weight = -np.inf
        else:
            log_weight = math.log(weight)

        # Calculate the log marginal likelihood for this component.
        evidence = log_evidence(N1, N0, a_k, b_k, binomial=False)

        # Store the unnormalized posterior weight in log space.
        log_weights.append(log_weight + evidence)

    # Convert log weights into a NumPy array.
    log_weights = np.asarray(log_weights, dtype=float)

    # Log-sum-exp normalization for numerical stability.
    max_log_weight = np.max(log_weights)

    if not np.isfinite(max_log_weight):
        raise ValueError("The posterior mixing weights cannot be normalized.")

    # Exponentiate after subtracting the maximum log weight.
    post_weights = np.exp(log_weights - max_log_weight)

    # Normalize posterior mixing weights so that they sum to one.
    post_weights /= np.sum(post_weights)

    return post_weights, post_components

