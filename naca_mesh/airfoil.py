"""NACA four-digit coordinates and analytic surface normals.

Coordinates are normalized by chord. Sampling stations are the chord parameter,
not the final surface X coordinate after the camber-normal thickness offset.
"""

import numpy as np


def decode_airfoil(code):
    """Return maximum camber, camber position, and thickness from four digits."""
    if not isinstance(code, str) or len(code) != 4 or not code.isascii() or not code.isdigit():
        raise ValueError("airfoil must be a four-digit string, e.g. '4412' or '0012'")
    m, p, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
    if t <= 0 or (m > 0 and p == 0):
        raise ValueError("airfoil requires positive thickness and a valid camber position")
    return m, p, t


def camber(x, m, p):
    """Return camber, slope, and second derivative, including symmetric profiles."""
    x = np.asarray(x, dtype=float)
    if m == 0:
        zeros = np.zeros_like(x)
        return zeros, zeros, zeros
    if not 0 < p < 1:
        raise ValueError("cambered airfoils require 0 < camber position < 1")
    yc = np.where(x < p, m / p**2 * (2*p*x - x**2),
                  m / (1-p)**2 * ((1-2*p) + 2*p*x - x**2))
    dyc = np.where(x < p, 2*m / p**2 * (p-x), 2*m / (1-p)**2 * (p-x))
    d2yc = np.where(x < p, -2*m / p**2, -2*m / (1-p)**2)
    return yc, dyc, d2yc


def _stations(x):
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)) or np.any((x < 0) | (x > 1)):
        raise ValueError("chord sampling stations must be finite and within [0, 1]")
    return x


def thickness_camber_angle(x, m=0.04, p=0.4, t=0.12, closed_TE=True):
    x = _stations(x)
    a4 = 0.1036 if closed_TE else 0.1015
    yt = 5*t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - a4*x**4)
    yc, dyc, _ = camber(x, m, p)
    return yt, yc, np.arctan(dyc)


def coordinates(x, m=0.04, p=0.4, t=0.12, closed_TE=True):
    """Return upper X/Y and lower X/Y, for either arrays or scalar stations."""
    scalar = np.ndim(x) == 0
    x = _stations(x)
    yt, yc, theta = thickness_camber_angle(x, m, p, t, closed_TE)
    values = (x - yt*np.sin(theta), yc + yt*np.cos(theta),
              x + yt*np.sin(theta), yc - yt*np.cos(theta))
    return tuple(float(v) for v in values) if scalar else values


def outward_normal_on_outer_surface(xparam, m=0.04, p=0.4, t=0.12,
                                    closed_TE=True, eps=1e-12):
    """Return (X, Y, normal X, normal Y) for upper and lower surfaces.

    The leading-edge derivative is evaluated at eps, matching the existing
    generator's regularization. Normals point out of the airfoil.
    """
    x = _stations(xparam)
    xe = np.maximum(x, eps)
    a4 = 0.1036 if closed_TE else 0.1015
    yt = 5*t*(0.2969*np.sqrt(xe) - 0.1260*xe - 0.3516*xe**2 + 0.2843*xe**3 - a4*xe**4)
    yt_p = 5*t*(0.2969*(1.0/(2.0*np.sqrt(xe))) - 0.1260 - 0.7032*xe + 0.8529*xe**2 - 4.0*a4*xe**3)
    yc, dyc, d2yc = camber(x, m, p)
    theta = np.arctan(dyc)
    theta_p = d2yc / (1 + dyc**2)
    s, c = np.sin(theta), np.cos(theta)
    xu, yu = x - yt*s, yc + yt*c
    xl, yl = x + yt*s, yc - yt*c
    xu_p = 1 - yt_p*s - yt*c*theta_p
    yu_p = dyc + yt_p*c - yt*s*theta_p
    xl_p = 1 + yt_p*s + yt*c*theta_p
    yl_p = dyc - yt_p*c + yt*s*theta_p
    su, sl = np.hypot(xu_p, yu_p), np.hypot(xl_p, yl_p)
    su, sl = np.where(su == 0, 1, su), np.where(sl == 0, 1, sl)
    return (xu, yu, -yu_p/su, xu_p/su), (xl, yl, yl_p/sl, -xl_p/sl)
