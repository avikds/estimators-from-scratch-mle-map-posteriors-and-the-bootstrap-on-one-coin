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

