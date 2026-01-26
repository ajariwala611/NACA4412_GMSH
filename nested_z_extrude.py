import gmsh
import sys
from dataclasses import dataclass
import numpy as np

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

def nested_mesh_yz_transfinite(p: Params):
    if p.ny < 2 or p.nz_mid < 2:
        raise ValueError("Transfinite divisions are number of NODES; must be >= 2.")
    if not (0.0 < p.h_in < p.H):
        raise ValueError("Require 0 < h_in < H.")


    # Adaptively choose inner width to make Z-spacing uniform
    w_in, z1, dz = compute_uniform_inner_width(p.W, p.nz_left_right, p.nz_mid)
    z2 = p.W - z1
    r_out = p.y_progression
    a_out = first_interval(p.H, p.y_progression, p.ny - 1)
    r_in = progression_for_same_first_interval(p.h_in, p.ny - 1, a_out)
    # Sanity
    if not (0.0 < w_in < p.W):
        raise RuntimeError(f"Computed w_in={w_in} is invalid for W={p.W}.")
    if abs((2 * z1 + w_in) - p.W) > 1e-10:
        raise RuntimeError("Uniform-Z computation inconsistent.")

    print(f"[uniform-Z] dz = {dz:.6g}, z1(left/right) = {z1:.6g}, w_in = {w_in:.6g}")

    gmsh.initialize()
    gmsh.model.add("YZ_transfinite_blocks_uniformZ")
    geo = gmsh.model.geo

    # Points in gmsh XY plane: gmsh x := z, gmsh y := y
    A = geo.addPoint(0.0, 0.0, 0.0)
    B = geo.addPoint(0.0, 0.0, p.W)
    C = geo.addPoint(0.0, p.H, p.W)
    D = geo.addPoint(0.0, p.H, 0.0)

    E = geo.addPoint(0.0, 0.0, z1)
    F = geo.addPoint(0.0, 0.0, z2)
    G = geo.addPoint(0.0, p.h_in, z2)
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

    # 4 block surfaces
    loop_inner = geo.addCurveLoop([l_EF, l_FG, l_GH, l_HE])
    s_inner = geo.addPlaneSurface([loop_inner])
    gmsh.model.geo.mesh.setTransfiniteSurface(s_inner)
    gmsh.model.geo.mesh.setRecombine(2, s_inner, 90)

    loop_left = geo.addCurveLoop([l_AE, -l_HE, -l_DH, l_DA])
    s_left = geo.addPlaneSurface([loop_left])
    gmsh.model.geo.mesh.setTransfiniteSurface(s_left)
    gmsh.model.geo.mesh.setRecombine(2, s_left, 90)

    loop_right = geo.addCurveLoop([l_FB, l_BC, l_CG, -l_FG])
    s_right = geo.addPlaneSurface([loop_right])
    gmsh.model.geo.mesh.setTransfiniteSurface(s_right)
    gmsh.model.geo.mesh.setRecombine(2, s_right, 90)

    loop_top = geo.addCurveLoop([-l_CD, l_CG, l_GH, -l_DH])
    s_top = geo.addPlaneSurface([loop_top])
    gmsh.model.geo.mesh.setTransfiniteSurface(s_top)
    gmsh.model.geo.mesh.setRecombine(2, s_top, 90)

    geo.synchronize()

    # --- Transfinite curves (node counts) ---
    gmsh.model.geo.mesh.setTransfiniteCurve(l_AE, p.nz_left_right)
    gmsh.model.geo.mesh.setTransfiniteCurve(l_EF, p.nz_mid)
    gmsh.model.geo.mesh.setTransfiniteCurve(l_FB, p.nz_left_right)

    gmsh.model.geo.mesh.setTransfiniteCurve(l_CD, p.nz_mid)
    gmsh.model.geo.mesh.setTransfiniteCurve(l_GH, p.nz_mid)

    # Verticals with optional grading in y
    if abs(p.y_progression - 1.0) < 1e-14:
        gmsh.model.geo.mesh.setTransfiniteCurve(l_BC, p.ny)
        gmsh.model.geo.mesh.setTransfiniteCurve(l_FG, p.ny)
        gmsh.model.geo.mesh.setTransfiniteCurve(l_HE, p.ny)
        gmsh.model.geo.mesh.setTransfiniteCurve(l_DA, p.ny)
    else:
        gmsh.model.geo.mesh.setTransfiniteCurve(l_BC, p.ny, "Progression", r_out)         # B->C
        gmsh.model.geo.mesh.setTransfiniteCurve(l_FG, p.ny, "Progression", r_in)         # F->G
        gmsh.model.geo.mesh.setTransfiniteCurve(l_HE, p.ny, "Progression", 1.0 / r_in)   # H->E
        gmsh.model.geo.mesh.setTransfiniteCurve(l_DA, p.ny, "Progression", 1.0 / r_out)   # D->A

    # Diagonals (must match nz_left/nz_right == ny)
    gmsh.model.geo.mesh.setTransfiniteCurve(l_DH, p.nz_left_right)
    gmsh.model.geo.mesh.setTransfiniteCurve(l_CG, p.nz_left_right)

    gmsh.model.geo.synchronize()

    surfaces_2d =  [s_inner, s_left, s_right, s_top]
    return surfaces_2d

def extrude_blocks_along_x(
    surfaces_2d,
    Lx=0.2,   # extrusion length along gmsh Z
    Nx=17     # number of nodes along Z (=> Nz-1 hex layers)
):
    """
    Extrude GEO surfaces (created with gmsh.model.geo) along +Z and
    enforce transfinite constraints so the result stays structured all-hex.

    Usage requirements:
      - Call AFTER you define 2D geometry and transfinite curves/surfaces,
        but BEFORE gmsh.model.geo.mesh.generate(...)
      - Do NOT call gmsh.model.geo.mesh.generate(2) before extrusion.
    """
    geo = gmsh.model.geo
    vol_tags = []

    # Extrude one surface at a time to avoid entity duplication issues
    for s in surfaces_2d:
        out = geo.extrude(
            [(2, s)],
            Lx, 0.0, 0.0,            
            numElements=[Nx - 1],
            recombine=True
        )
        for dim, tag in out:
            if dim == 3:
                vol_tags.append(tag)

    # Synchronize after geometry creation
    geo.synchronize()

    # Set transfinite volumes (structured hex intent)
    for v in vol_tags:
        gmsh.model.geo.mesh.setTransfiniteVolume(v)

    # Enforce Nz nodes on all curves that run purely along Z
    # (These are the "extrusion direction" curves created by the sweep.)
    for v in vol_tags:
        bnd = gmsh.model.getBoundary([(3, v)], oriented=False, recursive=True)
        curve_dimtags = [dt for dt in bnd if dt[0] == 1]

        for _, ctag in curve_dimtags:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(1, ctag)
            dx = abs(xmax - xmin)
            dy = abs(ymax - ymin)
            dz = abs(zmax - zmin)

            # Curve is (approximately) parallel to Z if dx,dy ~ 0 and dz > 0
            if dx < 1e-10 and dy < 1e-10 and dz > 1e-10:
                gmsh.model.geo.mesh.setTransfiniteCurve(ctag, Nx)

if __name__ == "__main__":
    surfaces_2d = nested_mesh_yz_transfinite(Params())
    extrude_blocks_along_x(surfaces_2d)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)

    if "-nopopup" not in sys.argv:
        gmsh.fltk.run()

    gmsh.finalize()
