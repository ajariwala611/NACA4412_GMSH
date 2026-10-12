"""Shared airfoil splines and normal-offset nose geometry for all mesh modes."""

import gmsh
import numpy as np
from .config import Params
from .airfoil import outward_normal_on_outer_surface

def nearest_station_from_upper(xu, yu, x_target):
    """Return (idx, x_near_u, y_near_u) where x_near_u = xu[idx] is closest to x_target."""
    i = min(range(len(xu)), key=lambda k: abs(xu[k] - x_target))
    return i, xu[i], yu[i]

def nearest_station_from_lower(xl, yl, x_target):
    """Return (idx, x_near_l, y_near_l) where x_near_l = xl[idx] is closest to x_target."""
    i = min(range(len(xl)), key=lambda k: abs(xl[k] - x_target))
    return i, xl[i], yl[i]

def _pt_xyz(pt_tag):
    coords = gmsh.model.getValue(0,pt_tag,[])
    return coords[0], coords[1], coords[2]

def _unit_surface_normal_at_index(x_arr, y_arr):
    x = np.asarray(x_arr, dtype=float)
    y = np.asarray(y_arr, dtype=float)

    # Numerical tangent via gradients (robust on non-uniform x if sorted)
    dx = np.gradient(x)
    dy = np.gradient(y)
    tlen = np.hypot(dx, dy)
    # Avoid division by zero
    tlen[tlen == 0.0] = 1.0

    # Unit tangent (tx, ty) and unit normal (nx, ny)
    tx = dx / tlen
    ty = dy / tlen
    nx = ty
    ny =  tx

    return nx, ny

def build_three_airfoil_splines(xu, yu, xl, yl, p: Params, make_wire=True):
    """
    Build three splines by filtering points around a threshold x_th (no interpolation).

    Inputs:
      xu, yu : upper surface arrays (typically TE -> LE, decreasing x)
      xl, yl : lower surface arrays (typically LE -> TE, increasing x)
      c      : chord scale (multiplies x and y)
      x_th   : threshold x where to split (default 0.1)
      make_wire : retained for compatibility; this helper returns separate splines

    Returns:
      Upper/nose/lower curve tags, TE/upper-cut/LE/lower-cut point tags,
      and the three spline point lists.
    """

    # Create TE and LE points (reused across splines)
    pt_TE = gmsh.model.geo.addPoint(1.0 * p.c, 0.0, 0.0)  # $$x=1$$
    pt_LE = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)      # $$x=0$$

    nU = len(xu)
    nL = len(xl)
    if nU < 2 or nL < 2:
        raise ValueError("Upper/lower arrays must contain TE and LE at least.")

    # Create gmsh points for INTERNAL samples (exclude TE at idx 0 and LE at idx -1)
    upper_pts_tags = {}  # index -> point tag (for i in 1..nU-2)
    for i in range(1, nU - 1):
        upper_pts_tags[i] = gmsh.model.geo.addPoint(xu[i] * p.c, yu[i] * p.c, 0.0)
    lower_pts_tags = {}  # index -> point tag (for i in 1..nL-2)
    for i in range(1, nL - 1):
        lower_pts_tags[i] = gmsh.model.geo.addPoint(xl[i] * p.c, yl[i] * p.c, 0.0)

    # Choose a single cut X from the upper samples (nearest to x_th)
    iU_cut, x_cut, yU_cut = nearest_station_from_upper(xu, yu, p.x_th)
    # Clamp to internal range for upper
    pt_Ucut_index = max(1, min(nU - 2, iU_cut))

    # -----------------------------
    # Upper far: TE -> (upper cut point)
    # -----------------------------
    # Use the same x_cut for filtering; take all internal points strictly beyond the cut toward TE
    upper_far_internal = [i for i in range(1, nU - 1) if xu[i] > x_cut]
    upper_far_pts = [pt_TE]
    if len(upper_far_internal) >= 1:
        # Append those before the cut index (to avoid duplicating the cut)
        upper_far_pts += [upper_pts_tags[i] for i in upper_far_internal if i < pt_Ucut_index]
    # Always append the explicit cut point
    upper_far_pts.append(upper_pts_tags[pt_Ucut_index])

    # -----------------------------
    # Nose: (upper cut) -> LE -> (lower up to cut) -> (lower cut)
    # -----------------------------
    # Upper-side nose segment: indices after the cut toward LE
    nose_pts = [upper_pts_tags[pt_Ucut_index]]
    for i in range(pt_Ucut_index + 1, nU - 1):
        nose_pts.append(upper_pts_tags[i])
    nose_pts.append(pt_LE)

    # Lower-side: find the nearest index to the SAME x_cut
    iL_cut, _, yL_cut = nearest_station_from_lower(xl, yl, x_cut)
    # Clamp to internal range for lower
    pt_Lcut_index = max(1, min(nL - 2, iL_cut))

    # Lower nose internal: all internal points up to and including x <= x_cut
    lower_nose_internal = [i for i in range(1, nL - 1) if xl[i] <= x_cut]
    if len(lower_nose_internal) >= 1:
        # Append those strictly before the cut index (avoid duplicating the cut)
        nose_pts += [lower_pts_tags[i] for i in lower_nose_internal if i < pt_Lcut_index]
    # Append the explicit lower cut point
    pt_Lcut = lower_pts_tags[pt_Lcut_index]
    nose_pts.append(pt_Lcut)

    # -----------------------------
    # Lower far: (lower cut) ->... -> TE
    # -----------------------------
    lower_far_pts = [pt_Lcut]
    for i in range(pt_Lcut_index + 1, nL - 1):
        lower_far_pts.append(lower_pts_tags[i])
    lower_far_pts.append(pt_TE)

    # Create splines
    curve_upper_far = gmsh.model.geo.addSpline(upper_far_pts)
    curve_nose      = gmsh.model.geo.addSpline(nose_pts)
    curve_lower_far = gmsh.model.geo.addSpline(lower_far_pts)
    gmsh.model.geo.synchronize()
    gmsh.model.geo.mesh.setTransfiniteCurve(curve_upper_far,p.upper_surface)
    gmsh.model.geo.mesh.setTransfiniteCurve(curve_nose,p.nose_surface,"Bump",p.nose_bump_coeff)
    gmsh.model.geo.mesh.setTransfiniteCurve(curve_lower_far,p.lower_surface)

    return curve_upper_far, curve_nose, curve_lower_far, [pt_TE, upper_far_pts[-1], pt_LE, nose_pts[-1]], upper_far_pts, nose_pts, lower_far_pts

def offset_nose_segments_analytic(p, iU, iL, d, outward_sign=+1):
    """
    Offsets the upper + lower nose segments using analytic NACA normals,
    using p.xU_cos for upper parameterization and p.xL_cos for lower.

    Assumes:
      p.xu,p.yu are built from p.xU_cos
      p.xl,p.yl are built from p.xL_cos
      p has p.m, p.p, p.t, p.closed_TE (or defaults)
    """

    # ---- Upper segment (indices iU+1 .. end in p.xu/p.yu) ----
    idxU = np.arange(iU+1, len(p.xu))

    xU = np.asarray(p.xu[idxU], dtype=float)
    yU = np.asarray(p.yu[idxU], dtype=float)
    xUpar = np.asarray(p.xU_cos[idxU], dtype=float)

    # keep your original “reverse if increasing”
    if xU[0] < xU[-1]:
        xU = xU[::-1]
        yU = yU[::-1]
        xUpar = xUpar[::-1]

    (xu_chk, yu_chk, nux, nuy), _ = outward_normal_on_outer_surface(
        xUpar, m=p.m, p=p.p, t=p.t, closed_TE=getattr(p, "closed_TE", True)
    )

    xU_off = xU + outward_sign * d * nux
    yU_off = yU + outward_sign * d * nuy

    # ---- Lower segment (indices 1 .. iL-1 in p.xl/p.yl) ----
    idxL = np.arange(1, iL)

    xL = np.asarray(p.xl[idxL], dtype=float)
    yL = np.asarray(p.yl[idxL], dtype=float)
    xLpar = np.asarray(p.xL_cos[idxL], dtype=float)

    if xL[0] > xL[-1]:
        xL = xL[::-1]
        yL = yL[::-1]
        xLpar = xLpar[::-1]

    _, (xl_chk, yl_chk, nlx, nly) = outward_normal_on_outer_surface(
        xLpar, m=p.m, p=p.p, t=p.t, closed_TE=getattr(p, "closed_TE", True)
    )

    xL_off = xL + outward_sign * d * nlx
    yL_off = yL + outward_sign * d * nly

    return np.concatenate([xU_off, xL_off]), np.concatenate([yU_off, yL_off])

def build_scaled_nose_spline(p_upper_tag, p_lower_tag, d, z, p):

    x_upper, y_upper, _ = _pt_xyz(p_upper_tag)

    iU_cut, x_cut, _ = nearest_station_from_upper(p.xu, p.yu, p.x_th)
    iL_cut, _,    _ = nearest_station_from_lower(p.xl, p.yl, x_cut)

    # Clamp indices
    iU = max(1, min(len(p.xu) - 2, iU_cut))
    iL = max(1, min(len(p.xl) - 2, iL_cut))

    # distance to use (your original definition)
    d = np.sqrt((x_upper - p.xu[iU])**2 + (y_upper - p.yu[iU])**2)


    # analytic offset (outward_sign=+1 -> outward)
    x_off, y_off = offset_nose_segments_analytic(p, iU, iL, d, outward_sign=+1)

    tags = [gmsh.model.geo.addPoint(float(x), float(y), float(z))
            for x, y in zip(x_off, y_off)]

    gmsh.model.geo.synchronize()
    new_spline_tag = gmsh.model.geo.addSpline([p_upper_tag] + tags + [p_lower_tag])
    gmsh.model.geo.synchronize()
    return new_spline_tag
