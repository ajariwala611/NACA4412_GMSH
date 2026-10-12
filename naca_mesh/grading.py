"""Spacing utilities shared by planar and spanwise block meshes."""

import math
import numpy as np

def cosine_spacing(N, start=0.0, end=1.0):
    beta = np.linspace(0, math.pi, N)
    x = 0.5 * (1 - np.cos(beta))   # cosine spacing
    return start + (end - start) * x

def create_geometric_grading(n_elements, progression, first_layer_height=None, first_layer_count=1, normalize=True):
    """
    Creates geometric grading with fixed first layer height.

    Parameters:
    -----------
    n_elements : int
        Number of elements
    progression : float
        Progression ratio
    first_layer_height : float, optional
        First layer height (kept fixed). If None, computed from progression.
    first_layer_count : int
        Number of elements in first layer (default: 1)
    normalize : bool
        Normalize remaining heights so final value = 1.0 (default: True)

    Returns:
    --------
    tuple : (element_counts, heights)
    """

    if not isinstance(n_elements, int) or n_elements < 1:
        raise ValueError("n_elements must be a positive integer")
    if not math.isfinite(progression) or progression <= 0:
        raise ValueError("progression must be finite and positive")
    if not isinstance(first_layer_count, int) or first_layer_count < 1:
        raise ValueError("first_layer_count must be a positive integer")
    if first_layer_height is not None and not 0 < first_layer_height <= 1:
        raise ValueError("first_layer_height must be within (0, 1]")
    if n_elements == 1:
        return [first_layer_count], [1.0 if normalize or first_layer_height is None else first_layer_height]
    n = n_elements
    r = progression

    # Use provided first layer height or compute it
    if first_layer_height is None:
        a = 1.0 / n if abs(r - 1.0) < 1e-14 else (r - 1) / (r**n - 1)
        first_height = a
    else:
        first_height = first_layer_height
        a = first_layer_height

    # Initialize as lists
    element_counts = [first_layer_count]
    heights = [first_height]

    # Build remaining layers
    for i in range(1, n - 1):
        element_counts.append(1)
        heights.append(heights[i - 1] + a * r**i)

    # Final layer
    element_counts.append(1)
    heights.append(heights[-1] + a * r**(n - 1))

    # Normalize if requested (keep first height fixed, scale the rest)
    if normalize and first_layer_height is not None:
        total_height = heights[-1]
        # Keep first height fixed, scale the rest proportionally
        heights = [heights[0]] + [first_height + (h - first_height) * (1.0 - first_height) / (total_height - first_height) for h in heights[1:]]
    elif normalize:
        # Normalize all heights
        total_height = heights[-1]
        heights = [h / total_height for h in heights]

    return element_counts, heights

def first_interval(L: float, r: float, n_intervals: int) -> float:
    """$$a = \\frac{L(r-1)}{r^{n}\\!-\\!1}$$ if $$r\\ne1$$, else $$a = L/n$$."""
    if n_intervals <= 0:
        raise ValueError("n_intervals must be >= 1")
    if abs(r - 1.0) < 1e-14:
        return L / n_intervals
    if n_intervals * math.log(r) > 700:
        return 0.0
    return L * (r - 1.0) / (r**n_intervals - 1.0)

def progression_for_same_first_interval(
    L_target: float,
    n_intervals: int,
    a_target: float,
    r_lo: float = 1.0,
    r_hi: float = 1e6,
    max_iter: int = 80,
    tol: float = 1e-14,
) -> float:
    """
    Solve for $$r\\ge1$$ such that $$\\text{first\\_interval}(L_{\\text{target}}, r, n) = a_{\\text{target}}$$.
    """
    if a_target <= 0:
        raise ValueError("a_target must be > 0")
    a_uniform = L_target / n_intervals
    if abs(a_uniform - a_target) / a_target < 1e-12:
        return 1.0
    if a_target > a_uniform:
        # would require r < 1; clamp for typical wall clustering use
        return 1.0

    def f(r):
        return first_interval(L_target, r, n_intervals) - a_target

    lo = max(r_lo, 1.0)
    hi = max(r_hi, lo * 10.0)
    while f(hi) > 0:
        hi *= 10.0
        if hi > 1e12:
            raise RuntimeError("Failed to bracket progression ratio.")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) < tol:
            return mid
        if fmid > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
