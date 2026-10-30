"""
Estimators from Scratch: MLE, MAP, Posteriors and the Bootstrap on One Coin

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - coin_counts
import numpy as np

def coin_counts(y):
    """
    Convert a sequence of coin flips into sufficient statistics.

    Parameters
    ----------
    y : array-like
        One-dimensional sequence containing only 1 (heads) and 0 (tails).

    Returns
    -------
    tuple
        (N1, N0), where N1 is the number of heads and N0 is the number of tails.
        Both values are Python integers.
    """

    # Convert the input into a NumPy array.
    try:
        y = np.asarray(y)
    except (TypeError, ValueError) as exc:
        raise ValueError("y must be a one-dimensional array or list.") from exc

    # Validate that the input is one-dimensional.
    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional array or list.")

    # Validate that every entry is either 0 (tails) or 1 (heads).
    try:
        valid = np.all((y == 0) | (y == 1))
    except (TypeError, ValueError):
        raise ValueError("Every entry in y must be either 0 or 1.")

    if not valid:
        raise ValueError("Every entry in y must be either 0 or 1.")

    # Count heads (1) and tails (0).
    N1 = int(np.count_nonzero(y == 1))
    N0 = int(np.count_nonzero(y == 0))

    return (N1, N0)

# Step 2 - bernoulli_nll
def bernoulli_nll(theta, N1, N0):
    """
    Compute the negative log-likelihood of Bernoulli coin flips.

    Parameters
    ----------
    theta : float or numpy.ndarray
        Probability of heads, in [0, 1].
    N1 : int
        Number of heads.
    N0 : int
        Number of tails.

    Returns
    -------
    float or numpy.ndarray
        Negative log-likelihood. Returns a Python float for scalar
        input and a NumPy array for array input.
    """

    # Convert theta to a NumPy array for element-wise computation.
    theta_array = np.asarray(theta, dtype=float)

    # Initialize the negative log-likelihood with zeros.
    nll = np.zeros_like(theta_array, dtype=float)

    # Suppress warnings when evaluating log(0) at boundary values.
    with np.errstate(divide="ignore", invalid="ignore"):

        # Contribution from heads.
        # Skip the term when N1 = 0 (convention: 0 * log(0) = 0).
        if N1 != 0:
            nll -= N1 * np.log(theta_array)

        # Contribution from tails.
        # Skip the term when N0 = 0.
        if N0 != 0:
            nll -= N0 * np.log1p(-theta_array)

    # Return a Python float for scalar input.
    if np.isscalar(theta):
        return float(nll)

    # Return a NumPy array for array input.
    return nll

# Step 3 - golden_section_min
import math

def golden_section_min(f, lo, hi, tol=1e-10):
    """
    Minimize a one-dimensional unimodal function using golden-section search.

    Parameters
    ----------
    f : callable
        Objective function to minimize.
    lo : float
        Lower bound of the search interval.
    hi : float
        Upper bound of the search interval.
    tol : float, optional
        Search tolerance. Default is 1e-10.

    Returns
    -------
    float
        Midpoint of the final search interval.
    """

    # Convert interval bounds and tolerance to Python floats.
    a = float(lo)
    b = float(hi)
    tol = float(tol)

    # Validate the search interval and tolerance.
    if a > b:
        raise ValueError("lo must be less than or equal to hi.")

    if not math.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be a positive finite number.")

    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("Search bounds must be finite.")

    # If both endpoints are identical, return that point.
    if a == b:
        return float(a)

    # Golden ratio constant.
    r = (math.sqrt(5) - 1) / 2

    # Initialize the two interior points.
    c = b - r * (b - a)
    d = a + r * (b - a)

    # Evaluate the function at both interior points.
    fc = f(c)
    fd = f(d)

    # Continue until the interval is sufficiently small.
    while (b - a) > tol:

        if fc < fd:
            # The minimum lies in [a, d].
            b = d
            d = c
            fd = fc

            # Calculate and evaluate only the new interior point.
            c = b - r * (b - a)
            fc = f(c)

        else:
            # The minimum lies in [c, b].
            a = c
            c = d
            fc = fd

            # Calculate and evaluate only the new interior point.
            d = a + r * (b - a)
            fd = f(d)

    # Return the midpoint of the final interval.
    return float((a + b) / 2)

