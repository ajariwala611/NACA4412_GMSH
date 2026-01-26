import gmsh
import sys
import numpy as np
import math
from dataclasses import dataclass

@dataclass
class Params:
    # Geometry
    W: float = 0.1          # Z extent
    H: float = 0.3          # nominal top Y (used if you do not provide absolute y_top)
    h_in: float = 0.25      # inner block top height offset from bottom

    # Transfinite counts for the 4-block template
    ny: int = 30            # verticals AND diagonals
    nz_mid: int = 5         # inner bottom/top AND outer top
    nz_left_right: int = 15 # outer left/right horizontals

    # Vertical grading
    y_progression: float = 1.1

    # Airfoil three-spline counts (optional, for your nose/upper/lower curves)
    upper_surface: int = 200
    lower_surface: int = 50
    nose_surface: int = 30
    c: float = 1.0
    Nu: int = 200
    Nl: int = 200
    # ---------- Helpers (embedded) ----------
    def __init__(self):
        """Initialize and compute uniform Z segments"""
        self.w_in, self.z1, self.z2, self.dz = self.uniform_z_segments()

    @staticmethod
    def first_interval(L: float, r: float, n_intervals: int) -> float:
        """$$a = \\frac{L(r-1)}{r^{n}\\!-\\!1}$$ if $$r\\ne1$$, else $$a = L/n$$."""
        if n_intervals <= 0:
            raise ValueError("n_intervals must be >= 1")
        if abs(r - 1.0) < 1e-14:
            return L / n_intervals
        return L * (r - 1.0) / (r**n_intervals - 1.0)

    @staticmethod
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
            return Params.first_interval(L_target, r, n_intervals) - a_target

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

    def uniform_z_segments(self):
        """
        Compute uniform spacing across three horizontal segments:
        A–E (nz_left_right), E–F (nz_mid), F–B (nz_left_right).
        Returns (w_in, z1, z2, dz).
        """
        if self.nz_left_right < 2 or self.nz_mid < 2:
            raise ValueError("Need at least 2 nodes per curve.")
        denom = 2 * (self.nz_left_right - 1) + (self.nz_mid - 1)
        dz = self.W / denom
        z1 = dz * (self.nz_left_right - 1)
        w_in = dz * (self.nz_mid - 1)
        z2 = self.W - z1
        if not (0.0 < w_in < self.W):
            raise RuntimeError(f"Computed w_in={w_in} is invalid for W={self.W}.")
        if abs((2 * z1 + w_in) - self.W) > 1e-10:
            raise RuntimeError("Uniform-Z computation inconsistent.")
        return w_in, z1, z2, dz

    def grading(self, H_eff: float, h_eff: float):
        """
        Compute vertical grading parameters:
        - $$a_{\\text{out}}$$ = first interval on the outer verticals (using $$H_{\\text{eff}}$$ and $$r_{\\text{out}}$$),
        - $$r_{\\text{in}}$$ chosen so the inner verticals have the same first interval size $$a_{\\text{out}}$$ over $$h_{\\text{eff}}$$.
        Returns (r_out, a_out, r_in).
        """
        if H_eff <= 0.0:
            raise ValueError("H_eff must be > 0.")
        if not (0.0 < h_eff < H_eff):
            raise ValueError("Require 0 < h_eff < H_eff.")
        r_out = self.y_progression
        a_out = Params.first_interval(H_eff, r_out, self.ny - 1)
        r_in  = Params.progression_for_same_first_interval(h_eff, self.ny - 1, a_out)
        return r_out, a_out, r_in

def _pt_xyz(pt_tag):
    coords = gmsh.model.getValue(0,pt_tag,[])
    return coords[0], coords[1], coords[2] 

def create_nested_block_surfaces_from_points(A, B, C, D, E, F, G, H, p: Params, orientation_down = 0, recombine_angle=90):
    # Infer vertical levels from point coordinates
    _, yA, _ = _pt_xyz(A)
    _, yC, _ = _pt_xyz(C)
    _, yG, _ = _pt_xyz(G)
    bottom_y = yA
    top_y    = yC
    inner_y  = yG

    H_eff = top_y - bottom_y
    h_eff = inner_y - bottom_y

    downward = False
    if H_eff < 0.0:
        # Normalize spans to positive and flip orientation
        bottom_y, top_y = top_y, bottom_y
        H_eff = -H_eff
        h_eff = -h_eff
        downward = True

    if not (H_eff > 0.0):
        raise ValueError("Invalid vertical span: top_y must be > bottom_y.")
    if not (bottom_y < inner_y < top_y):
        raise ValueError("Require bottom_y < inner_y < top_y.")

    # Grading (outer first-interval and inner matching) with normalized spans
    r_out, a_out, r_in = p.grading(H_eff, h_eff)

    # Curves
    AE = gmsh.model.occ.addLine(A, E)
    EF = gmsh.model.occ.addLine(E, F)
    FB = gmsh.model.occ.addLine(F, B)

    BC = gmsh.model.occ.addLine(B, C)
    CD = gmsh.model.occ.addLine(C, D)
    DA = gmsh.model.occ.addLine(D, A)

    FG = gmsh.model.occ.addLine(F, G)
    GH = gmsh.model.occ.addLine(G, H)
    HE = gmsh.model.occ.addLine(H, E)

    DH = gmsh.model.occ.addLine(D, H)
    CG = gmsh.model.occ.addLine(C, G)

    # Surfaces
    loop_inner = gmsh.model.occ.addCurveLoop([EF, FG, GH, HE])
    s_inner = gmsh.model.occ.addPlaneSurface([loop_inner])

    loop_left = gmsh.model.occ.addCurveLoop([AE, -HE, -DH, DA])
    s_left = gmsh.model.occ.addPlaneSurface([loop_left])

    loop_right = gmsh.model.occ.addCurveLoop([FB, BC, CG, -FG])
    s_right = gmsh.model.occ.addPlaneSurface([loop_right])

    loop_top = gmsh.model.occ.addCurveLoop([-CD, CG, GH, -DH])
    s_top = gmsh.model.occ.addPlaneSurface([loop_top])

    gmsh.model.occ.synchronize()

    # Transfinite surfaces + recombine
    for s in (s_inner, s_left, s_right, s_top):
        gmsh.model.mesh.setTransfiniteSurface(s)
        gmsh.model.mesh.setRecombine(2, s, recombine_angle)

    # Transfinite curves (counts)
    gmsh.model.mesh.setTransfiniteCurve(AE, p.nz_left_right)
    gmsh.model.mesh.setTransfiniteCurve(EF, p.nz_mid)
    gmsh.model.mesh.setTransfiniteCurve(FB, p.nz_left_right)

    gmsh.model.mesh.setTransfiniteCurve(CD, p.nz_mid)
    gmsh.model.mesh.setTransfiniteCurve(GH, p.nz_mid)

    if abs(p.y_progression - 1.0) < 1e-14:
        gmsh.model.mesh.setTransfiniteCurve(BC, p.ny)
        gmsh.model.mesh.setTransfiniteCurve(FG, p.ny)
        gmsh.model.mesh.setTransfiniteCurve(HE, p.ny)
        gmsh.model.mesh.setTransfiniteCurve(DA, p.ny)
    else:
        if not orientation_down:
            gmsh.model.mesh.setTransfiniteCurve(BC, p.ny, "Progression", r_out)        # bottom -> top clusters at bottom
            gmsh.model.mesh.setTransfiniteCurve(FG, p.ny, "Progression", r_in)
            gmsh.model.mesh.setTransfiniteCurve(HE, p.ny, "Progression", 1.0 / r_in)
            gmsh.model.mesh.setTransfiniteCurve(DA, p.ny, "Progression", 1.0 / r_out)
        else:
            gmsh.model.mesh.setTransfiniteCurve(BC, p.ny, "Progression", 1.0 / r_out)  # flip: clusters at top
            gmsh.model.mesh.setTransfiniteCurve(FG, p.ny, "Progression", 1.0 / r_in)
            gmsh.model.mesh.setTransfiniteCurve(HE, p.ny, "Progression", r_in)
            gmsh.model.mesh.setTransfiniteCurve(DA, p.ny, "Progression", r_out)

    # Diagonals (structured requirement: nz_left_right == ny)
    gmsh.model.mesh.setTransfiniteCurve(DH, p.nz_left_right)
    gmsh.model.mesh.setTransfiniteCurve(CG, p.nz_left_right)

    return [s_inner, s_left, s_right, s_top], [AE,EF,FB,BC,CD,DA,FG,GH,HE,DH,CG]

def nested_mesh_yz_transfinite_at_x(
    p: Params,
    pt_tag,             # bottom y (absolute)
    orientation_down = 0,
):
    """
    Build the nested 4-block transfinite surface at a single plane x0.
    Coordinates: (x0, y, z). y spans [y0, y0+H]; z spans [0, W].
    Returns:
      surfaces_2d: [s_inner, s_left, s_right, s_top]
      mesh_dict: dictionary of surfaces/curves settings (if return_settings=True)
    """
    if p.ny < 2 or p.nz_mid < 2:
        raise ValueError("Transfinite divisions are number of NODES; must be >= 2.")
    if not (0.0 < p.h_in < p.H):
        raise ValueError("Require 0 < h_in < H.")
    x0, y0, _ =  _pt_xyz(pt_tag)

    if not orientation_down:
        H_eff = p.H            
        h_eff = p.h_in 
    elif orientation_down:
        H_eff = - p.H            
        h_eff = - p.h_in 

    w_in, z1, z2, dz = p.uniform_z_segments()
    # Points: (x0 fixed, y varies, z varies)
    A  = pt_tag
    B  = gmsh.model.occ.addPoint(x0, y0,    p.W)
    C  = gmsh.model.occ.addPoint(x0, H_eff, p.W)
    D  = gmsh.model.occ.addPoint(x0, H_eff, 0.0)
    E  = gmsh.model.occ.addPoint(x0, y0,    z1)
    F  = gmsh.model.occ.addPoint(x0, y0,    z2)
    G  = gmsh.model.occ.addPoint(x0, h_eff, z2)
    H = gmsh.model.occ.addPoint(x0, h_eff, z1)
    gmsh.model.occ.synchronize()
    surfaces, curves = create_nested_block_surfaces_from_points(A, B, C, D, E, F, G, H, p, recombine_angle=90)
    pts = [A, B, C, D, E, F, G, H]

    return surfaces, curves, pts

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

def nearest_station_from_upper(xu, yu, x_target):
    """Return (idx, x_near_u, y_near_u) where x_near_u = xu[idx] is closest to x_target."""
    i = min(range(len(xu)), key=lambda k: abs(xu[k] - x_target))
    return i, xu[i], yu[i]  # FIX: use yu[i], not yu[I]

def nearest_station_from_lower(xl, yl, x_target):
    """Return (idx, x_near_l, y_near_l) where x_near_l = xl[idx] is closest to x_target."""
    i = min(range(len(xl)), key=lambda k: abs(xl[k] - x_target))
    return i, xl[i], yl[i]

def build_three_airfoil_splines(xu, yu, xl, yl, p: Params, x_th=0.1, make_wire=True):
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
    pt_TE = gmsh.model.occ.addPoint(1.0 * p.c, 0.0, 0.0)  # $$x=1$$
    pt_LE = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)      # $$x=0$$

    nU = len(xu)
    nL = len(xl)
    if nU < 2 or nL < 2:
        raise ValueError("Upper/lower arrays must contain TE and LE at least.")

    # Create gmsh points for INTERNAL samples (exclude TE at idx 0 and LE at idx -1)
    upper_pts_tags = {}  # index -> point tag (for i in 1..nU-2)
    for i in range(1, nU - 1):
        upper_pts_tags[i] = gmsh.model.occ.addPoint(xu[i] * p.c, yu[i] * p.c, 0.0)
    lower_pts_tags = {}  # index -> point tag (for i in 1..nL-2)
    for i in range(1, nL - 1):
        lower_pts_tags[i] = gmsh.model.occ.addPoint(xl[i] * p.c, yl[i] * p.c, 0.0)

    # Choose a single cut X from the upper samples (nearest to x_th)
    iU_cut, x_cut, yU_cut = nearest_station_from_upper(xu, yu, x_th)
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
    curve_upper_far = gmsh.model.occ.addSpline(upper_far_pts)
    curve_nose      = gmsh.model.occ.addSpline(nose_pts)
    curve_lower_far = gmsh.model.occ.addSpline(lower_far_pts)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setTransfiniteCurve(curve_upper_far,p.upper_surface)
    gmsh.model.mesh.setTransfiniteCurve(curve_nose,p.nose_surface)
    gmsh.model.mesh.setTransfiniteCurve(curve_lower_far,p.lower_surface)
    wire_tag = None
    if make_wire:
        # Curves connect end-to-end: TE -> Ucut ->... -> LE ->... -> Lcut ->... -> TE
        wire_tag = gmsh.model.occ.addWire([curve_upper_far, curve_nose, curve_lower_far], checkClosed=True)
        gmsh.model.occ.synchronize()

    return curve_upper_far, curve_nose, curve_lower_far, wire_tag, [pt_TE, upper_far_pts[-1], pt_LE, nose_pts[-1]]

gmsh.initialize()
gmsh.model.add("nested_at_x")

p = Params()
# --- Trailing and Leading Edge Points ---
xU_cos = cosine_spacing(p.Nu + 1, 1.0, 0.0)
xL_cos = cosine_spacing(p.Nl + 1, 0.0, 1.0)
xu, yu, _, _ = naca4412(xU_cos)
_, _, xl, yl = naca4412(xL_cos)

left_airfoil_curve_upper, left_airfoil_curve_nose, left_airfoil_curve_lower, left_airfoil_wire, airfoil_split_tags = build_three_airfoil_splines(xu, yu, xl, yl, p, x_th=0.1, make_wire=False)

# upper_airfoil_surface = gmsh.model.occ.extrude([(1,left_airfoil_curve_upper)],0.0,0.0,p.W,numElements=[2*(p.nz_left_right-1) + p.nz_mid-1],recombine=True)[1][1]
# lower_airfoil_surface = gmsh.model.occ.extrude([(1,left_airfoil_curve_lower)],0.0,0.0,p.W,numElements=[2*(p.nz_left_right-1) + p.nz_mid-1],recombine=True)[1][1]
gmsh.model.occ.synchronize()
# gmsh.model.mesh.setTransfiniteSurface(upper_airfoil_surface)
# gmsh.model.mesh.setRecombine(2, upper_airfoil_surface, 90)
# gmsh.model.mesh.setTransfiniteSurface(nose_airfoil_surface)
# gmsh.model.mesh.setRecombine(2, nose_airfoil_surface, 90)
# gmsh.model.mesh.setTransfiniteSurface(lower_airfoil_surface)
# gmsh.model.mesh.setRecombine(2, lower_airfoil_surface, 90)

surface_tags_10c_upper, curve_tags_10c_upper, pts_tags_10c_upper = nested_mesh_yz_transfinite_at_x(p,airfoil_split_tags[1],orientation_down=0)
surface_tags_10c_lower, curve_tags_10c_lower, pts_tags_10c_lower = nested_mesh_yz_transfinite_at_x(p,airfoil_split_tags[3],orientation_down=1)
# surface_tags_TE_upper,  curve_tags_TE_upper, pts_tags_TE_upper = nested_mesh_yz_transfinite_at_x(p,airfoil_split_tags[0],orientation_down=0)
# surface_tags_TE_lower,  curve_tags_TE_lower, pts_tags_TE_lower = nested_mesh_yz_transfinite_at_x(p,airfoil_split_tags[0],orientation_down=1)
gmsh.model.occ.synchronize()
gmsh.model.occ.removeAllDuplicates()

airfoil_split_upper = _pt_xyz(pts_tags_10c_upper[3])
airfoil_split_lower = _pt_xyz(pts_tags_10c_lower[3])
x_arc_center = (airfoil_split_upper[0] + airfoil_split_lower[0]) / 2
y_arc_center = (airfoil_split_upper[1] + airfoil_split_lower[1]) / 2
z_arc_center = (airfoil_split_upper[2] + airfoil_split_lower[2]) / 2
arc_ceter_pt_tag = gmsh.model.occ.addPoint(x_arc_center,0.0,0.0)
gmsh.model.occ.synchronize()
# arc_tag = gmsh.model.occ.addCircleArc(pts_tags_10c_upper[3], arc_ceter_pt_tag, pts_tags_10c_lower[3])
arc_tag_1 = gmsh.model.occ.addCircleArc(pts_tags_10c_lower[3], arc_ceter_pt_tag, pts_tags_10c_upper[3])
# arc_tag_1_wire = gmsh.model.occ.addWire([arc_tag_1])
# gmsh.model.occ.remove([(0,arc_ceter_pt_tag)])
gmsh.model.occ.synchronize()
gmsh.model.mesh.setTransfiniteCurve(arc_tag_1,p.nose_surface)
circ1_loop = gmsh.model.occ.addCurveLoop([curve_tags_10c_upper[5],arc_tag_1,curve_tags_10c_lower[5],left_airfoil_curve_nose])
circ1_surf = gmsh.model.occ.addPlaneSurface([circ1_loop])
gmsh.model.occ.synchronize()
gmsh.model.mesh.setTransfiniteSurface(circ1_surf)
gmsh.model.mesh.setRecombine(2,circ1_surf,90)


airfoil_split_upper = _pt_xyz(pts_tags_10c_upper[7])
airfoil_split_lower = _pt_xyz(pts_tags_10c_lower[7])
x_arc_center = (airfoil_split_upper[0] + airfoil_split_lower[0]) / 2
y_arc_center = (airfoil_split_upper[1] + airfoil_split_lower[1]) / 2
z_arc_center = (airfoil_split_upper[2] + airfoil_split_lower[2]) / 2
arc_ceter_pt_tag = gmsh.model.occ.addPoint(x_arc_center,0,z_arc_center)
gmsh.model.occ.synchronize()
arc_tag_2 = gmsh.model.occ.addCircleArc(pts_tags_10c_lower[7], arc_ceter_pt_tag, pts_tags_10c_upper[7])
gmsh.model.occ.remove([(0,arc_ceter_pt_tag)])
gmsh.model.occ.synchronize()
nose_airfoil_extrude_1 = gmsh.model.occ.extrude([(1,left_airfoil_curve_nose)],0.0,0.0,z_arc_center,numElements=[p.nz_left_right-1],recombine=True)[0][1]
gmsh.model.occ.synchronize()
gmsh.model.mesh.setTransfiniteCurve(arc_tag_2,p.nose_surface)
circ2_loop = gmsh.model.occ.addCurveLoop([curve_tags_10c_upper[8],arc_tag_2,curve_tags_10c_lower[8],nose_airfoil_extrude_1])
circ2_surf = gmsh.model.occ.addPlaneSurface([circ2_loop])
gmsh.model.occ.synchronize()
gmsh.model.mesh.setTransfiniteSurface(circ2_surf)
gmsh.model.mesh.setRecombine(2,circ2_surf,90)

airfoil_split_upper = _pt_xyz(pts_tags_10c_upper[6])
airfoil_split_lower = _pt_xyz(pts_tags_10c_lower[6])
x_arc_center = (airfoil_split_upper[0] + airfoil_split_lower[0]) / 2
y_arc_center = (airfoil_split_upper[1] + airfoil_split_lower[1]) / 2
z_arc_center = (airfoil_split_upper[2] + airfoil_split_lower[2]) / 2
arc_ceter_pt_tag = gmsh.model.occ.addPoint(x_arc_center,0,z_arc_center)
gmsh.model.occ.synchronize()
arc_tag_3 = gmsh.model.occ.addCircleArc(pts_tags_10c_lower[6], arc_ceter_pt_tag, pts_tags_10c_upper[6])
gmsh.model.occ.remove([(0,arc_ceter_pt_tag)])
gmsh.model.occ.synchronize()
nose_airfoil_extrude_2 = gmsh.model.occ.extrude([(1,nose_airfoil_extrude_1)],0.0,0.0,p.z2-p.z1,numElements=[p.nz_mid-1],recombine=True)[0][1]
gmsh.model.occ.synchronize()
gmsh.model.mesh.setTransfiniteCurve(arc_tag_3,p.nose_surface)
circ3_loop = gmsh.model.occ.addCurveLoop([curve_tags_10c_upper[6],arc_tag_3,curve_tags_10c_lower[6],nose_airfoil_extrude_2])
circ3_surf = gmsh.model.occ.addPlaneSurface([circ3_loop])
gmsh.model.occ.synchronize()
gmsh.model.mesh.setTransfiniteSurface(circ3_surf)
gmsh.model.mesh.setRecombine(2,circ3_surf,90)

airfoil_split_upper = _pt_xyz(pts_tags_10c_upper[2])
airfoil_split_lower = _pt_xyz(pts_tags_10c_lower[2])
x_arc_center = (airfoil_split_upper[0] + airfoil_split_lower[0]) / 2
y_arc_center = (airfoil_split_upper[1] + airfoil_split_lower[1]) / 2
z_arc_center = (airfoil_split_upper[2] + airfoil_split_lower[2]) / 2
arc_ceter_pt_tag = gmsh.model.occ.addPoint(x_arc_center,0,z_arc_center)
gmsh.model.occ.synchronize()
arc_tag_4 = gmsh.model.occ.addCircleArc(pts_tags_10c_lower[2], arc_ceter_pt_tag, pts_tags_10c_upper[2])
# arc_tag_4_wire = gmsh.model.occ.addWire([arc_tag_4])
# gmsh.model.occ.remove([(0,arc_ceter_pt_tag)])
gmsh.model.occ.synchronize()
nose_airfoil_extrude_right = gmsh.model.occ.extrude([(1,nose_airfoil_extrude_2)],0.0,0.0,p.W-p.z2,numElements=[p.nz_left_right-1],recombine=True)[0][1]
gmsh.model.occ.synchronize()
gmsh.model.mesh.setTransfiniteCurve(arc_tag_4,p.nose_surface)
circ4_loop = gmsh.model.occ.addCurveLoop([curve_tags_10c_upper[3],arc_tag_4,curve_tags_10c_lower[3],nose_airfoil_extrude_right])
circ4_surf = gmsh.model.occ.addPlaneSurface([circ4_loop])
gmsh.model.occ.synchronize()
gmsh.model.mesh.setTransfiniteSurface(circ4_surf)
gmsh.model.mesh.setRecombine(2,circ4_surf,90)

# circ5_loop = gmsh.model.occ.addCurveLoop([curve_tags_10c_upper[4],-arc_tag_1,-curve_tags_10c_lower[4],arc_tag_4])
# circ5_surf = gmsh.model.occ.addPlaneSurface([circ5_loop])
# gmsh.model.occ.synchronize()
# gmsh.model.mesh.setTransfiniteSurface(circ5_surf)
# gmsh.model.mesh.setRecombine(2,circ5_surf,90)

# circ5_loop = gmsh.model.occ.addCurveLoop([curve_tags_10c_upper[7],arc_tag_2,curve_tags_10c_lower[7],arc_tag_3])
# circ5_surf = gmsh.model.occ.addPlaneSurface([circ5_loop])
# gmsh.model.occ.synchronize()
# gmsh.model.mesh.setTransfiniteSurface(circ5_surf)
# gmsh.model.mesh.setRecombine(2,circ5_surf,90)

# circ5_loop = gmsh.model.occ.addCurveLoop([curve_tags_10c_upper[9],-arc_tag_2,-curve_tags_10c_lower[9],arc_tag_1])
# circ5_surf = gmsh.model.occ.addPlaneSurface([circ5_loop])
# gmsh.model.occ.synchronize()
# gmsh.model.mesh.setTransfiniteSurface(circ5_surf)
# gmsh.model.mesh.setRecombine(2,circ5_surf,90)

# circ5_loop = gmsh.model.occ.addCurveLoop([curve_tags_10c_upper[10],-arc_tag_3,-curve_tags_10c_lower[10],arc_tag_4])
# circ5_surf = gmsh.model.occ.addPlaneSurface([circ5_loop])
# gmsh.model.occ.synchronize()
# gmsh.model.mesh.setTransfiniteSurface(circ5_surf)
# gmsh.model.mesh.setRecombine(2,circ5_surf,90)

gmsh.model.occ.synchronize()

# Generate 3D mesh
gmsh.model.mesh.generate(1)

# Launch GUI (optional - comment out if running in batch mode)
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

# Finalize
gmsh.finalize()