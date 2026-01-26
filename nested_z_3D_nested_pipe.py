import gmsh
import sys
import numpy as np
import math
from dataclasses import dataclass

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
    """First interval size for  gmsh.model.occmetric grading with ratio r over n_intervals."""
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

    # Points in gmsh XY plane: gmsh x := z, gmsh y := y
    A = gmsh.model.occ.addPoint(1.0, 0.0, 0.0)
    B = gmsh.model.occ.addPoint(1.0, 0.0, p.W)
    C = gmsh.model.occ.addPoint(1.0, p.H, p.W)
    D = gmsh.model.occ.addPoint(1.0, p.H, 0.0)

    E = gmsh.model.occ.addPoint(1.0, 0.0, z1)
    F = gmsh.model.occ.addPoint(1.0, 0.0, z2)
    G = gmsh.model.occ.addPoint(1.0, p.h_in, z2)
    Hh =    gmsh.model.occ.addPoint(1.0, p.h_in, z1)

    # Curves
    l_AE =  gmsh.model.occ.addLine(A, E)
    l_EF =  gmsh.model.occ.addLine(E, F)
    l_FB =  gmsh.model.occ.addLine(F, B)

    l_BC =  gmsh.model.occ.addLine(B, C)
    l_CD =  gmsh.model.occ.addLine(C, D)
    l_DA =  gmsh.model.occ.addLine(D, A)

    l_FG =  gmsh.model.occ.addLine(F, G)
    l_GH =  gmsh.model.occ.addLine(G, Hh)
    l_HE =  gmsh.model.occ.addLine(Hh, E)

    l_DH =  gmsh.model.occ.addLine(D, Hh)
    l_CG =  gmsh.model.occ.addLine(C, G)
    
    # 4 block surfaces
    loop_inner =    gmsh.model.occ.addCurveLoop([l_EF, l_FG, l_GH, l_HE])
    s_inner =   gmsh.model.occ.addPlaneSurface([loop_inner])
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setTransfiniteSurface(s_inner)
    gmsh.model.mesh.setRecombine(2, s_inner, 90)

    loop_left = gmsh.model.occ.addCurveLoop([l_AE, -l_HE, -l_DH, l_DA])
    s_left =    gmsh.model.occ.addPlaneSurface([loop_left])
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setTransfiniteSurface(s_left)
    gmsh.model.mesh.setRecombine(2, s_left, 90)

    loop_right =    gmsh.model.occ.addCurveLoop([l_FB, l_BC, l_CG, -l_FG])
    s_right =   gmsh.model.occ.addPlaneSurface([loop_right])
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setTransfiniteSurface(s_right)
    gmsh.model.mesh.setRecombine(2, s_right, 90)

    loop_top =  gmsh.model.occ.addCurveLoop([-l_CD, l_CG, l_GH, -l_DH])
    s_top = gmsh.model.occ.addPlaneSurface([loop_top])
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setTransfiniteSurface(s_top)
    gmsh.model.mesh.setRecombine(2, s_top, 90)
    gmsh.model.occ.synchronize()

    # --- Transfinite curves (node counts) ---
    gmsh.model.mesh.setTransfiniteCurve(l_AE, p.nz_left_right)
    gmsh.model.mesh.setTransfiniteCurve(l_EF, p.nz_mid)
    gmsh.model.mesh.setTransfiniteCurve(l_FB, p.nz_left_right)

    gmsh.model.mesh.setTransfiniteCurve(l_CD, p.nz_mid)
    gmsh.model.mesh.setTransfiniteCurve(l_GH, p.nz_mid)

    # Verticals with optional grading in y
    if abs(p.y_progression - 1.0) < 1e-14:
        gmsh.model.mesh.setTransfiniteCurve(l_BC, p.ny)
        gmsh.model.mesh.setTransfiniteCurve(l_FG, p.ny)
        gmsh.model.mesh.setTransfiniteCurve(l_HE, p.ny)
        gmsh.model.mesh.setTransfiniteCurve(l_DA, p.ny)
    else:
        gmsh.model.mesh.setTransfiniteCurve(l_BC, p.ny, "Progression", r_out)         # B->C
        gmsh.model.mesh.setTransfiniteCurve(l_FG, p.ny, "Progression", r_in)         # F->G
        gmsh.model.mesh.setTransfiniteCurve(l_HE, p.ny, "Progression", 1.0 / r_in)   # H->E
        gmsh.model.mesh.setTransfiniteCurve(l_DA, p.ny, "Progression", 1.0 / r_out)   # D->A

    # Diagonals (must match nz_left/nz_right == ny)
    gmsh.model.mesh.setTransfiniteCurve(l_DH, p.nz_left_right)
    gmsh.model.mesh.setTransfiniteCurve(l_CG, p.nz_left_right)

    surfaces_2d = [s_inner, s_left, s_right, s_top]
    gmsh.model.occ.synchronize()
    return surfaces_2d

# Initialize gmsh
gmsh.initialize()
gmsh.model.add("naca4412_pipe_extrusion")

# NACA 4412 airfoil parameters
m = 0.04  # maximum camber (4%)
p = 0.4   # position of maximum camber (40% chord)
t = 0.12  # maximum thickness (12%)
c = 1.0
Nu = 100
Nl = 100
# --- helper: analytical NACA-4412 with closed TE -------------
def naca4412(x, closed_TE=True):
    m, p, t = 0.04, 0.4, 0.12
    yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - (0.1036 if closed_TE else 0.1015)*x**4)
    yc  = np.where(x < p, m/p**2*(2*p*x - x**2),
                   m/(1-p)**2*((1-2*p)+2*p*x - x**2))
    dyc = np.where(x < p, 2*m/p**2*(p-x),
                   2*m/(1-p)**2*(p-x))
    th  = np.arctan(dyc)
    xu, yu = x - yt*np.sin(th), yc + yt*np.cos(th)
    xl, yl = x + yt*np.sin(th), yc - yt*np.cos(th)
    return xu, yu, xl, yl

# --- helper: cosine-distributed spacing ---------------------------
def cosine_spacing(N, start=0.0, end=1.0):
    beta = np.linspace(0, math.pi, N)
    x = 0.5 * (1 - np.cos(beta))   # cosine spacing
    return start + (end - start) * x

# --- Trailing and Leading Edge Points ---
xU_cos = cosine_spacing(Nu + 1, 1.0, 0.0)
xL_cos = cosine_spacing(Nl + 1, 0.0, 1.0)
xu, yu, _, _ = naca4412(xU_cos)
_, _, xl, yl = naca4412(xL_cos)

def build_three_airfoil_splines(xu, yu, xl, yl, c, x_th=0.1, make_wire=True):
    """
    Build three splines by filtering points around a threshold x_th (no interpolation).

    Inputs:
      xu, yu : upper surface arrays (typically TE -> LE, decreasing x)
      xl, yl : lower surface arrays (typically LE -> TE, increasing x)
      c      : chord scale (multiplies x and y)
      x_th   : threshold x where to split (default 0.1)
      make_wire : if True, return a closed wire connecting the three splines

    Returns:
      curve_upper_far, curve_nose, curve_lower_far, wire_tag (or None if make_wire=False)
    """

    # Create TE and LE points (reused across splines)
    pt_TE = gmsh.model.occ.addPoint(1.0 * c, 0.0, 0.0)  # $$x=1$$
    pt_LE = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)      # $$x=0$$

    nU = len(xu)
    nL = len(xl)
    if nU < 2 or nL < 2:
        raise ValueError("Upper/lower arrays must contain TE and LE at least.")

    # Create gmsh points for INTERNAL samples (exclude TE at idx 0 and LE at idx -1)
    upper_pts_tags = {}  # index -> point tag (for i in 1..nU-2)
    for i in range(1, nU - 1):
        upper_pts_tags[i] = gmsh.model.occ.addPoint(xu[i] * c, yu[i] * c, 0.0)
    lower_pts_tags = {}  # index -> point tag (for i in 1..nL-2)
    for i in range(1, nL - 1):
        lower_pts_tags[i] = gmsh.model.occ.addPoint(xl[i] * c, yl[i] * c, 0.0)

    # -----------------------------
    # Upper far: TE -> (last upper internal with x > x_th)
    # -----------------------------
    upper_far_internal = [i for i in range(1, nU - 1) if xu[i] > x_th]
    upper_far_pts = [pt_TE]
    pt_Ucut = None
    if len(upper_far_internal) >= 1:
        for i in upper_far_internal:
            upper_far_pts.append(upper_pts_tags[i])
        pt_Ucut_index = upper_far_internal[-1]
        pt_Ucut = upper_pts_tags[pt_Ucut_index]
    else:
        # Fallback: ensure at least two points by adding the first internal (closest to TE)
        pt_Ucut_index = 1
        pt_Ucut = upper_pts_tags[pt_Ucut_index]
        upper_far_pts.append(pt_Ucut)

    # -----------------------------
    # Nose: (pt_Ucut) -> LE -> (lower up to threshold) -> (pt_Lcut)
    # Upper-side nose segment: indices after pt_Ucut_index toward LE
    # -----------------------------
    nose_pts = [pt_Ucut]
    for i in range(pt_Ucut_index + 1, nU - 1):
        nose_pts.append(upper_pts_tags[i])
    nose_pts.append(pt_LE)

    # Lower-side nose segment: from just after LE up to threshold
    lower_nose_internal = [i for i in range(1, nL - 1) if xl[i] <= x_th]
    if len(lower_nose_internal) >= 1:
        for i in lower_nose_internal[:-1]:
            nose_pts.append(lower_pts_tags[i])
        pt_Lcut_index = lower_nose_internal[-1]
    else:
        # Fallback: take the first internal point (closest to LE)
        pt_Lcut_index = 1

    pt_Lcut = lower_pts_tags[pt_Lcut_index]
    nose_pts.append(pt_Lcut)

    # -----------------------------
    # Lower far: (pt_Lcut) ->... -> TE
    # -----------------------------
    lower_far_pts = [pt_Lcut]
    for i in range(pt_Lcut_index + 1, nL - 1):
        lower_far_pts.append(lower_pts_tags[i])
    lower_far_pts.append(pt_TE)

    # Create splines
    curve_upper_far = gmsh.model.occ.addSpline(upper_far_pts)
    curve_nose      = gmsh.model.occ.addSpline(nose_pts)
    curve_lower_far = gmsh.model.occ.addSpline(lower_far_pts)
    gmsh.model.occ.synchronize()

    wire_tag = None
    if make_wire:
        # Curves connect end-to-end: TE -> Ucut ->... -> LE ->... -> Lcut ->... -> TE
        wire_tag = gmsh.model.occ.addWire([curve_upper_far, curve_nose, curve_lower_far], checkClosed=True)
        gmsh.model.occ.synchronize()

    return curve_upper_far, curve_nose, curve_lower_far, wire_tag

curve_upper, curve_nose, curve_lower, airfoil_wire = build_three_airfoil_splines(xu, yu, xl, yl, c, x_th=0.1, make_wire=False)

airfoil_loop = gmsh.model.occ.addWire([curve_upper])
# gmsh.model.mesh.setTransfiniteCurve(airfoil_loop, 501)  
gmsh.model.occ.synchronize()

nested_surf = nested_mesh_yz_transfinite(Params())
gmsh.model.occ.synchronize()
# outtag = [(2, nested_surf[0])]

# for i in range(len(xu) - 1):
#     outtag = gmsh.model.occ.extrude([(2, outtag[0][1])], 
#                            dx=xu[i+1]-xu[i], 
#                            dy=yu[i+1]-yu[i], 
#                            dz=0, 
#                            numElements=[3], 
#                            recombine=True)
for surf in nested_surf:
    print(gmsh.model.getBoundary([(2,surf)], oriented=False, recursive=False))
    pipe_tag = gmsh.model.occ.addPipe([(2, surf)], airfoil_loop,'CorrectedFrenet')
    pipe_surf = gmsh.model.occ.getSurfaceLoops(pipe_tag[0][1])
    surface_tags = pipe_surf[1][0]
    gmsh.model.occ.synchronize()
    # print(surface_tags)
    for surf_tag in surface_tags:
        print(gmsh.model.getBoundary([(2,surf_tag)], oriented=False, recursive=False))
        gmsh.model.mesh.setTransfiniteSurface(surf_tag)
        gmsh.model.mesh.setRecombine(2, surf_tag)
        # gmsh.model.mesh.setTransfiniteAutomatic([(2,surf_tag)],cornerAngle=90)
    # Get curves of this surface
        curve_loops = gmsh.model.occ.getCurveLoops(surf_tag)
        curve_tags = curve_loops[1][0]
        # print(curve_tags)
        for curve_tag in curve_tags:
            gmsh.model.mesh.setTransfiniteCurve(curve_tag, 10)
            # gmsh.model.mesh.setTransfiniteAutomatic([(1,curve_tag)],cornerAngle=90)
    # gmsh.model.occ.remove([(2, surf)])    
gmsh.model.occ.synchronize()
# exit()
# Set transfinite volume for structured mesh
volumes = gmsh.model.getEntities(3)

if volumes:
    for vol in volumes:
        gmsh.model.mesh.setTransfiniteVolume(vol[1])
        gmsh.model.mesh.setRecombine(3, vol[1],90)
gmsh.model.occ.synchronize()
# Generate 3D mesh
gmsh.model.mesh.generate(3)

# Launch GUI (optional - comment out if running in batch mode)
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

# Finalize
gmsh.finalize()