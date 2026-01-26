import gmsh
import sys
from dataclasses import dataclass
import numpy as np
from typing import Dict, List, Any

@dataclass
class Mesh2DContext:
    surfaces_2d: List[int]                 # [s_inner, s_left, s_right, s_top]
    curve_specs: Dict[int, Dict[str, Any]] # curveTag -> {"count": int, "progression": float|None}
    points: Dict[str, int]                 # {"A": A, "B": B, ...}
    curves: Dict[str, int]                 # {"AE": l_AE, "EF": l_EF, ...}
    surface_loops: Dict[str, List[str]]    # {"inner": ["EF","FG","GH","HE"], ...}

@dataclass
class Params:
    W: float = 0.1     # total width in Z (mapped to gmsh x)
    H: float = 0.05      # total height in Y (mapped to gmsh y)
    h_in: float = 0.04   # inner rectangle height (top of inner block)

    ny: int = 20       # nodes on verticals AND diagonals
    nz_mid: int = 5    # nodes on inner bottom/top AND outer top
    nz_left_right: int = 20

    # grading near y=0 (set 1.0 for uniform)
    y_progression: float = 1.1


def first_interval(L: float, r: float, n_intervals: int) -> float:
    """First interval size for geometric grading with ratio r over n_intervals."""
    if n_intervals <= 0:
        raise ValueError("n_intervals must be >= 1")
    if abs(r - 1.0) < 1e-14:
        return L / n_intervals
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
    Find r such that first_interval(L_target, r, n_intervals) == a_target.
    Uses bisection (monotone in r for r>=1).
    """
    if a_target <= 0:
        raise ValueError("a_target must be > 0")

    # If uniform already matches
    a_uniform = L_target / n_intervals
    if abs(a_uniform - a_target) / a_target < 1e-12:
        return 1.0

    # For r>=1, first_interval decreases as r increases.
    # Need a_target <= a_uniform to have a solution with r>=1.
    if a_target > a_uniform:
        # This would require r < 1 (clustering near the opposite end).
        # In most CFD wall-clustering cases, you want r>=1, so we clamp.
        return 1.0

    def f(r):
        return first_interval(L_target, r, n_intervals) - a_target

    # Ensure bracket
    lo = max(r_lo, 1.0)
    hi = max(r_hi, lo * 10.0)
    flo = f(lo)
    fhi = f(hi)
    while fhi > 0:  # need hi where f(hi) <= 0
        hi *= 10.0
        fhi = f(hi)
        if hi > 1e12:
            raise RuntimeError("Failed to bracket progression ratio.")

    # Bisection
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

def build_curve_specs(p, r_out, r_in,
                      l_AE, l_EF, l_FB, l_CD, l_GH,
                      l_BC, l_FG, l_HE, l_DA,
                      l_DH, l_CG):
    """
    Returns a dict: curveTag -> {"count": int, "progression": float|None}
    """
    specs = {
        l_AE: {"count": p.nz_left_right, "progression": None},
        l_EF: {"count": p.nz_mid,        "progression": None},
        l_FB: {"count": p.nz_left_right, "progression": None},
        l_CD: {"count": p.nz_mid,        "progression": None},
        l_GH: {"count": p.nz_mid,        "progression": None},

        l_DH: {"count": p.nz_left_right, "progression": None},
        l_CG: {"count": p.nz_left_right, "progression": None},
    }

    if abs(p.y_progression - 1.0) < 1e-14:
        specs.update({
            l_BC: {"count": p.ny, "progression": None},
            l_FG: {"count": p.ny, "progression": None},
            l_HE: {"count": p.ny, "progression": None},
            l_DA: {"count": p.ny, "progression": None},
        })
    else:
        specs.update({
            l_BC: {"count": p.ny, "progression": float(r_out)},
            l_FG: {"count": p.ny, "progression": float(r_in)},
            l_HE: {"count": p.ny, "progression": float(1.0 / r_in)},
            l_DA: {"count": p.ny, "progression": float(1.0 / r_out)},
        })

    return specs

def compute_uniform_inner_width(W: float, nz_left: int, nz_mid: int):
    """
    Choose inner width w_in so that spacing along Z is uniform across:
      A-E (nz_left nodes), E-F (nz_mid nodes), F-B (nz_left nodes)
    with symmetry (left==right).
    """
    if nz_left < 2 or nz_mid < 2:
        raise ValueError("Need at least 2 nodes per curve.")
    denom = 2 * (nz_left - 1) + (nz_mid - 1)
    dz = W / denom
    z1 = dz * (nz_left - 1)          # left (and right) segment length
    w_in = dz * (nz_mid - 1)         # inner width
    return w_in, z1, dz

def nested_mesh_yz_transfinite(p: Params) -> Mesh2DContext:
    if p.ny < 2 or p.nz_mid < 2:
        raise ValueError("Transfinite divisions are number of NODES; must be >= 2.")
    if not (0.0 < p.h_in < p.H):
        raise ValueError("Require 0 < h_in < H.")

    # --- sizing logic ---
    w_in, z1, dz = compute_uniform_inner_width(p.W, p.nz_left_right, p.nz_mid)
    z2 = p.W - z1
    r_out = p.y_progression
    a_out = first_interval(p.H, p.y_progression, p.ny - 1)
    r_in  = progression_for_same_first_interval(p.h_in, p.ny - 1, a_out)

    geo = gmsh.model.geo

    # Points in gmsh XY plane: gmsh x := z, gmsh y := y
    A  = geo.addPoint(0.0, 0.0, 0.0)
    B  = geo.addPoint(0.0, 0.0, p.W)
    C  = geo.addPoint(0.0, p.H, p.W)
    D  = geo.addPoint(0.0, p.H, 0.0)

    E  = geo.addPoint(0.0, 0.0, z1)
    F  = geo.addPoint(0.0, 0.0, z2)
    G  = geo.addPoint(0.0, p.h_in, z2)
    Hh = geo.addPoint(0.0, p.h_in, z1)

    # Curves
    l_AE = geo.addLine(A, E)
    l_EF = geo.addLine(E, F)
    l_FB = geo.addLine(F, B)

    l_BC = geo.addLine(B, C)
    l_CD = geo.addLine(C, D)
    l_DA = geo.addLine(D, A)

    l_FG = geo.addLine(F, G)
    l_GH = geo.addLine(G, Hh)
    l_HE = geo.addLine(Hh, E)

    l_DH = geo.addLine(D, Hh)
    l_CG = geo.addLine(C, G)

    # Surface loop definitions by CURVE NAME (deterministic rebuild)
    surface_loops = {
    "inner": ["EF", "FG", "GH", "HE"],
    "left":  ["AE", "-HE", "-DH", "DA"],
    "right": ["FB", "BC", "CG", "-FG"],
    "top":   ["-CD", "CG", "GH", "-DH"],
    }


    # Keep named points/curves maps for later copy/rebuild
    points = {"A": A, "B": B, "C": C, "D": D, "E": E, "F": F, "G": G, "H": Hh}
    curves  = {
        "AE": l_AE, "EF": l_EF, "FB": l_FB,
        "BC": l_BC, "CD": l_CD, "DA": l_DA,
        "FG": l_FG, "GH": l_GH, "HE": l_HE,
        "DH": l_DH, "CG": l_CG
    }

    # Build surfaces (you can keep your signed loops here exactly as-is)
    loop_inner = geo.addCurveLoop([l_EF, l_FG, l_GH, l_HE])
    s_inner = geo.addPlaneSurface([loop_inner])

    loop_left  = geo.addCurveLoop([l_AE, -l_HE, -l_DH, l_DA])
    s_left  = geo.addPlaneSurface([loop_left])

    loop_right = geo.addCurveLoop([l_FB, l_BC, l_CG, -l_FG])
    s_right = geo.addPlaneSurface([loop_right])

    loop_top   = geo.addCurveLoop([-l_CD, l_CG, l_GH, -l_DH])
    s_top   = geo.addPlaneSurface([loop_top])

    geo.synchronize()

    # --- Transfinite curves (node counts) ---
    gmsh.model.mesh.setTransfiniteCurve(l_AE, p.nz_left_right)
    gmsh.model.mesh.setTransfiniteCurve(l_EF, p.nz_mid)
    gmsh.model.mesh.setTransfiniteCurve(l_FB, p.nz_left_right)

    gmsh.model.mesh.setTransfiniteCurve(l_CD, p.nz_mid)
    gmsh.model.mesh.setTransfiniteCurve(l_GH, p.nz_mid)

    if abs(p.y_progression - 1.0) < 1e-14:
        gmsh.model.mesh.setTransfiniteCurve(l_BC, p.ny)
        gmsh.model.mesh.setTransfiniteCurve(l_FG, p.ny)
        gmsh.model.mesh.setTransfiniteCurve(l_HE, p.ny)
        gmsh.model.mesh.setTransfiniteCurve(l_DA, p.ny)
    else:
        gmsh.model.mesh.setTransfiniteCurve(l_BC, p.ny, "Progression", r_out)
        gmsh.model.mesh.setTransfiniteCurve(l_FG, p.ny, "Progression", r_in)
        gmsh.model.mesh.setTransfiniteCurve(l_HE, p.ny, "Progression", 1.0 / r_in)
        gmsh.model.mesh.setTransfiniteCurve(l_DA, p.ny, "Progression", 1.0 / r_out)

    gmsh.model.mesh.setTransfiniteCurve(l_DH, p.nz_left_right)
    gmsh.model.mesh.setTransfiniteCurve(l_CG, p.nz_left_right)

    # --- Transfinite surfaces + recombine ---
    for s in (s_inner, s_left, s_right, s_top):
        gmsh.model.mesh.setTransfiniteSurface(s)
        gmsh.model.mesh.setRecombine(2, s, 90)

    # Specs for later reuse (your existing helper)
    curve_specs = build_curve_specs(
        p, r_out, r_in,
        l_AE, l_EF, l_FB, l_CD, l_GH,
        l_BC, l_FG, l_HE, l_DA,
        l_DH, l_CG
    )

    surfaces_2d = [s_inner, s_left, s_right, s_top]

    return Mesh2DContext(
        surfaces_2d=surfaces_2d,
        curve_specs=curve_specs,
        points=points,
        curves=curves,
        surface_loops=surface_loops
    )

def _signed_curve_tag(curves_2d, curve_copy_map, name_with_sign):
    sign = 1
    key = name_with_sign
    if isinstance(name_with_sign, str) and name_with_sign.startswith("-"):
        sign = -1
        key = name_with_sign[1:]
    c_old = curves_2d[key]
    c_new = curve_copy_map[c_old]
    return sign * c_new


def copy_wireframe_rebuild_surfaces_transfinite(
    points_2d,          # dict: name -> pointTag
    curves_2d,          # dict: name -> curveTag
    surface_loops,      # dict: surfName -> [curveNames CCW]
    curve_specs,        # dict: curveTag -> {"count": N, "progression": r|None}
    dx=0.0, dy=0.0, dz=0.0,
    recombine_angle=90,
):
    """
    Copy a 2D transfinite block wireframe, translate it, rebuild surfaces,
    and reapply all transfinite constraints correctly.

    RETURNS:
        copied_surfaces: dict {surfaceName -> surfaceTag}
        curve_copy_map : dict {origCurveTag -> copiedCurveTag}
        point_copy_map : dict {origPointTag -> copiedPointTag}
    """

    geo = gmsh.model.geo

    # ------------------------------------------------------------
    # 1) COPY POINTS
    # ------------------------------------------------------------
    point_copy_map = {}
    for name, ptag in points_2d.items():
        out = geo.copy([(0, ptag)])
        geo.translate(out, dx, dy, dz)
        point_copy_map[ptag] = out[0][1]

    # ------------------------------------------------------------
    # 2) COPY CURVES USING COPIED POINTS
    # ------------------------------------------------------------
    curve_copy_map = {}
    for cname, ctag in curves_2d.items():
        # get endpoints
        bnd = gmsh.model.getBoundary([(1, ctag)], oriented=False)
        p0, p1 = [t[1] for t in bnd if t[0] == 0]

        p0c = point_copy_map[p0]
        p1c = point_copy_map[p1]

        c_new = geo.addLine(p0c, p1c)
        curve_copy_map[ctag] = c_new

    geo.synchronize()

    # ------------------------------------------------------------
    # 3) REBUILD SURFACES FROM COPIED CURVES
    # ------------------------------------------------------------
    copied_surfaces = {}

    for sname, loop in surface_loops.items():
        loop_tags = []
        for cname in loop:
            ctag = curves_2d[cname]
            loop_tags.append(curve_copy_map[ctag])

        cl = geo.addCurveLoop(loop_tags)
        s_new = geo.addPlaneSurface([cl])
        copied_surfaces[sname] = s_new

    geo.synchronize()

    # ------------------------------------------------------------
    # 4) REAPPLY TRANSFINITE CURVES
    # ------------------------------------------------------------
    for c_old, spec in curve_specs.items():
        c_new = curve_copy_map[c_old]

        if spec["progression"] is None:
            gmsh.model.mesh.setTransfiniteCurve(c_new, spec["count"])
        else:
            gmsh.model.mesh.setTransfiniteCurve(
                c_new,
                spec["count"],
                "Progression",
                spec["progression"],
            )

    # ------------------------------------------------------------
    # 5) TRANSFINITE + RECOMBINE SURFACES
    # ------------------------------------------------------------
    for s_new in copied_surfaces.values():
        gmsh.model.mesh.setTransfiniteSurface(s_new)
        gmsh.model.mesh.setRecombine(2, s_new, recombine_angle)

    return copied_surfaces, curve_copy_map, point_copy_map



if __name__ == "__main__":
    gmsh.initialize()
    gmsh.model.add("YZ_nested")

    ctx = nested_mesh_yz_transfinite(Params())

    # later: copy/rebuild using ctx.points, ctx.curves, ctx.surface_loops, ctx.curve_specs
    copied_surfaces, curve_map, point_map = \
    copy_wireframe_rebuild_surfaces_transfinite(
        points_2d=ctx.points,
        curves_2d=ctx.curves,
        surface_loops=ctx.surface_loops,
        curve_specs=ctx.curve_specs,
        dx=0.3
    )

    gmsh.model.mesh.generate(2)

    if "-nopopup" not in sys.argv:
        gmsh.fltk.run()
    gmsh.finalize()

