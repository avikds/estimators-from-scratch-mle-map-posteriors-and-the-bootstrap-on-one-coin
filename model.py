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

