import gmsh
import sys
import os
import numpy as np
import math
from dataclasses import dataclass

@dataclass
class Params:
    # Geometry
    W: float = 0.1          # Z extent
    H: float = 0.2          
    h_in: float = 0.15      
    Lx: float = 10.0
    Ly: float = 5.0

    # Mesh elements
    ny: int = 25            # verticals AND diagonals
    nz_mid: int = 5         # inner bottom/top AND outer top
    nz_left_right: int = 9 # outer left/right horizontals
    upper_surface: int = 350
    lower_surface: int = 50
    nose_surface: int = 150
    
    # Wake params
    Ny_outer: int = 20
    Nx_far_wake: int = 50
    Nx_near_wake: int = 40
    near_wake_length: float = 0.02
    wake_stretch_in_Y: float = 1.0
    near_wake_offset_in_x_1: float = 1.0
    near_wake_offset_in_x_2: float = 1.0

    # Grading
    nose_bump_coeff: float = 4
    y_progression: float = 1.175
    y_outer_vertical_progression: float = 1.075
    y_outer_horizontal_progression: float = 1.0325

    # Airfoil params to create splines 
    x_th: float = 0.1
    c: float = 1.0
    Nu: int = 500
    Nl: int = 500
    m: float = 0.04 
    p: float = 0.4
    t: float = 0.12
    
    # ---------- Helpers (embedded) ----------
    def __init__(self):
        """Initialize and compute uniform Z segments"""
        self.w_in, self.z1, self.z2, self.dz = self.uniform_z_segments()
        self.xU_cos = self.cosine_spacing(self.Nu + 1, 1.0, 0.0)
        self.xL_cos = self.cosine_spacing(self.Nl + 1, 0.0, 1.0)
        self.xu, self.yu, _, _ = self.naca4412(self.xU_cos)
        _, _, self.xl,self.yl = self.naca4412(self.xL_cos)
        # Wake params
        self.outer_veritcal_ele, self.outer_vertical_heights = self.create_geometric_grading(self.Ny_outer,self.y_outer_vertical_progression,None,1)
        self.outer_horizontal_ele, self.outer_horizontal_heigths = self.create_geometric_grading(self.Nx_far_wake,self.y_outer_horizontal_progression,self.near_wake_length,self.Nx_near_wake)
    @staticmethod
    def cosine_spacing(N, start=0.0, end=1.0):
        beta = np.linspace(0, math.pi, N)
        x = 0.5 * (1 - np.cos(beta))   # cosine spacing
        return start + (end - start) * x
    @staticmethod
    def naca4412(x, closed_TE=True):
        """
        NACA 4412 coordinates at stations x.

        mode:
        - "normal"  : offset along camber-line normal (classic airfoil definition)
        - "vertical": keep x identical; offset only in y (xu == xl == x)

        Returns: xu, yu, xl, yl (arrays or scalars matching x)
        """
        m, p, t = 0.04, 0.4, 0.12
        x = np.asarray(x, dtype=float)

        yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2
                    + 0.2843*x**3 - (0.1036 if closed_TE else 0.1015)*x**4)

        yc  = np.where(x < p, m/p**2*(2*p*x - x**2),
                    m/(1-p)**2*((1-2*p) + 2*p*x - x**2))
        dyc = np.where(x < p, 2*m/p**2*(p - x),
                    2*m/(1-p)**2*(p - x))
        th  = np.arctan(dyc)

        xu = x - yt*np.sin(th)
        yu = yc + yt*np.cos(th)
        xl = x + yt*np.sin(th)
        yl = yc - yt*np.cos(th)

        if np.isscalar(x):
            return float(xu), float(yu), float(xl), float(yl)
        return np.asarray(xu,float), yu, np.asarray(xl,float), yl
    
    @staticmethod
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
        
        n = n_elements
        r = progression
        
        # Use provided first layer height or compute it
        if first_layer_height is None:
            a = (r - 1) / (r**n - 1)
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
    
def naca4412(x, closed_TE=True, mode="normal"):
    """
    NACA 4412 coordinates at stations x.

    mode:
      - "normal"  : offset along camber-line normal (classic airfoil definition)
      - "vertical": keep x identical; offset only in y (xu == xl == x)

    Returns: xu, yu, xl, yl (arrays or scalars matching x)
    """
    m, p, t = 0.04, 0.4, 0.12
    x = np.asarray(x, dtype=float)

    yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2
                  + 0.2843*x**3 - (0.1036 if closed_TE else 0.1015)*x**4)

    yc  = np.where(x < p, m/p**2*(2*p*x - x**2),
                   m/(1-p)**2*((1-2*p) + 2*p*x - x**2))
    dyc = np.where(x < p, 2*m/p**2*(p - x),
                   2*m/(1-p)**2*(p - x))
    th  = np.arctan(dyc)

    xu = x - yt*np.sin(th)
    yu = yc + yt*np.cos(th)
    xl = x + yt*np.sin(th)
    yl = yc - yt*np.cos(th)

    if np.isscalar(x):
        return float(xu), float(yu), float(xl), float(yl)
    return xu, yu, xl, yl

def naca4412_yt_yc_theta(x):
    m, p, t = 0.04, 0.4, 0.12
    x = np.asarray(x, dtype=float)

    yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2
                  + 0.2843*x**3 - (0.1036)*x**4)

    yc  = np.where(x < p, m/p**2*(2*p*x - x**2),
                   m/(1-p)**2*((1-2*p) + 2*p*x - x**2))
    theta = np.arctan(np.where(x < p, 2*m/p**2*(p - x),
                    2*m/(1-p)**2*(p - x)))
    return yt, yc, theta

# --- helper: cosine-distributed spacing ---------------------------
def cosine_spacing(N, start=0.0, end=1.0):
    beta = np.linspace(0, math.pi, N)
    x = 0.5 * (1 - np.cos(beta))   # cosine spacing
    return start + (end - start) * x

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
    AE = gmsh.model.geo.addLine(A, E)
    EF = gmsh.model.geo.addLine(E, F)
    FB = gmsh.model.geo.addLine(F, B)

    BC = gmsh.model.geo.addLine(B, C)
    CD = gmsh.model.geo.addLine(C, D)
    DA = gmsh.model.geo.addLine(D, A)

    FG = gmsh.model.geo.addLine(F, G)
    GH = gmsh.model.geo.addLine(G, H)
    HE = gmsh.model.geo.addLine(H, E)

    DH = gmsh.model.geo.addLine(D, H)
    CG = gmsh.model.geo.addLine(C, G)
    gmsh.model.geo.synchronize()
    # Surfaces
    loop_inner = gmsh.model.geo.addCurveLoop([EF, FG, GH, HE])
    s_inner = gmsh.model.geo.addPlaneSurface([loop_inner])

    loop_left = gmsh.model.geo.addCurveLoop([AE, -HE, -DH, DA])
    s_left = gmsh.model.geo.addPlaneSurface([loop_left])

    loop_right = gmsh.model.geo.addCurveLoop([FB, BC, CG, -FG])
    s_right = gmsh.model.geo.addPlaneSurface([loop_right])

    loop_top = gmsh.model.geo.addCurveLoop([-CD, CG, GH, -DH])
    s_top = gmsh.model.geo.addPlaneSurface([loop_top])

    gmsh.model.geo.synchronize()

    # Transfinite surfaces + recombine
    for s in (s_inner, s_left, s_right, s_top):
        gmsh.model.geo.mesh.setTransfiniteSurface(s)
        gmsh.model.geo.mesh.setRecombine(2, s, recombine_angle)

    # Transfinite curves (counts)
    gmsh.model.geo.mesh.setTransfiniteCurve(AE, p.nz_left_right)
    gmsh.model.geo.mesh.setTransfiniteCurve(EF, p.nz_mid)
    gmsh.model.geo.mesh.setTransfiniteCurve(FB, p.nz_left_right)

    gmsh.model.geo.mesh.setTransfiniteCurve(CD, p.nz_mid)
    gmsh.model.geo.mesh.setTransfiniteCurve(GH, p.nz_mid)

    if abs(p.y_progression - 1.0) < 1e-14:
        gmsh.model.geo.mesh.setTransfiniteCurve(BC, p.ny)
        gmsh.model.geo.mesh.setTransfiniteCurve(FG, p.ny)
        gmsh.model.geo.mesh.setTransfiniteCurve(HE, p.ny)
        gmsh.model.geo.mesh.setTransfiniteCurve(DA, p.ny)
    else:
        if not orientation_down:
            gmsh.model.geo.mesh.setTransfiniteCurve(BC, p.ny, "Progression", r_out)        # bottom -> top clusters at bottom
            gmsh.model.geo.mesh.setTransfiniteCurve(FG, p.ny, "Progression", r_in)
            gmsh.model.geo.mesh.setTransfiniteCurve(HE, p.ny, "Progression", 1.0 / r_in)
            gmsh.model.geo.mesh.setTransfiniteCurve(DA, p.ny, "Progression", 1.0 / r_out)
        else:
            gmsh.model.geo.mesh.setTransfiniteCurve(BC, p.ny, "Progression", 1.0 / r_out)  # flip: clusters at top
            gmsh.model.geo.mesh.setTransfiniteCurve(FG, p.ny, "Progression", 1.0 / r_in)
            gmsh.model.geo.mesh.setTransfiniteCurve(HE, p.ny, "Progression", r_in)
            gmsh.model.geo.mesh.setTransfiniteCurve(DA, p.ny, "Progression", r_out)

    # Diagonals (structured requirement: nz_left_right == ny)
    gmsh.model.geo.mesh.setTransfiniteCurve(DH, p.nz_left_right)
    gmsh.model.geo.mesh.setTransfiniteCurve(CG, p.nz_left_right)
    gmsh.model.geo.synchronize()
    return [s_inner, s_left, s_right, s_top], [AE,EF,FB,BC,CD,DA,FG,GH,HE,DH,CG]

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
    offset = 0
    if x0 < 1:
        offset = 0.05
    if not orientation_down:
        H_eff = p.H - offset
        h_eff = p.h_in - offset
    elif orientation_down:
        H_eff = - p.H + offset
        h_eff = - p.h_in + offset

    # Outer and inner offsets along the local SURFACE normal (upper/lower)
    iU_cut, x_cut, yU_cut = nearest_station_from_upper(p.xu, p.yu, p.x_th)
    iL_cut, _, yL_cut     = nearest_station_from_lower(p.xl, p.yl, x_cut)

    if not orientation_down:
        # Upper side
        nU = len(p.xu)
        pt_Ucut_index = max(1, min(nU - 2, iU_cut))

        # Base surface point
        xu0 = p.xu[pt_Ucut_index]
        yu0 = p.yu[pt_Ucut_index]
        x_arr = [p.xu[pt_Ucut_index-1],p.xu[pt_Ucut_index],p.xu[pt_Ucut_index+1]]
        y_arr = [p.yu[pt_Ucut_index-1],p.yu[pt_Ucut_index],p.yu[pt_Ucut_index+1]]
        nx, ny = _unit_surface_normal_at_index(x_arr,y_arr)
        
        # Outer/inner from surface point along local normal
        x0_H = xu0 + (H_eff) * nx[1]
        x0_h = xu0 + (h_eff) * nx[1]
        y0_H = yu0 - (H_eff) * ny[1]
        y0_h = yu0 - (h_eff) * ny[1]

    elif orientation_down: 
        # Lower side
        nL = len(p.xl)
        pt_Lcut_index = max(1, min(nL - 2, iL_cut))

        # Base surface point
        xl0 = p.xl[pt_Lcut_index]
        yl0 = p.yl[pt_Lcut_index]
        x_arr = [p.xl[pt_Lcut_index-1],p.xl[pt_Lcut_index],p.xl[pt_Lcut_index+1]]
        y_arr = [p.yl[pt_Lcut_index-1],p.yl[pt_Lcut_index],p.yl[pt_Lcut_index+1]]
        nx, ny = _unit_surface_normal_at_index(x_arr,y_arr)

        # Outer/inner from surface point along local normal
        x0_H = xl0 - (H_eff) * nx[1]
        x0_h = xl0 - (h_eff) * nx[1]
        y0_H = yl0 + (H_eff) * ny[1]
        y0_h = yl0 + (h_eff) * ny[1]
   
    w_in, z1, z2, dz = p.uniform_z_segments()
    # Points: (x0 fixed, y varies, z varies)
    A  = pt_tag
    B  = gmsh.model.geo.addPoint(x0, y0,    p.W)
    C  = gmsh.model.geo.addPoint(x0_H if x0 < 1.0 else x0, y0_H if x0 < 1.0 else H_eff, p.W)
    D  = gmsh.model.geo.addPoint(x0_H if x0 < 1.0 else x0, y0_H if x0 < 1.0 else H_eff, 0.0)
    E  = gmsh.model.geo.addPoint(x0, y0,    z1)
    F  = gmsh.model.geo.addPoint(x0, y0,    z2)
    G  = gmsh.model.geo.addPoint(x0_h if x0 < 1.0 else x0, y0_h if x0 < 1.0 else h_eff, z2)
    H = gmsh.model.geo.addPoint(x0_h if x0 < 1.0 else x0, y0_h if x0 < 1.0 else h_eff, z1)
    gmsh.model.geo.synchronize()

    surfaces, curves = create_nested_block_surfaces_from_points(A, B, C, D, E, F, G, H, p, recombine_angle=90)
    pts = [A, B, C, D, E, F, G, H]
    gmsh.model.geo.synchronize()
    return surfaces, curves, pts

# --- helper: analytical NACA-4412 with closed TE -------------

def build_three_airfoil_splines(xu, yu, xl, yl, p: Params, make_wire=True):
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

def get_repeating_surfaces(surf1_tag, surf2_tag):
    """Returns all 6 surface tags [surf1, surf2, side1, side2, side3, side4]"""
    gmsh.model.geo.synchronize()
    surf1_tag, surf2_tag = int(surf1_tag), int(surf2_tag)
    
    surfaces_1 = set()
    surfaces_2 = set()
    
    for dim, tag in gmsh.model.getBoundary([(2, surf1_tag)]):
        surfaces_1.update(int(s) for s in gmsh.model.getAdjacencies(dim, tag)[0])
    
    for dim, tag in gmsh.model.getBoundary([(2, surf2_tag)]):
        surfaces_2.update(int(s) for s in gmsh.model.getAdjacencies(dim, tag)[0])
    
    return [surf1_tag, surf2_tag] + sorted(list(surfaces_1 & surfaces_2))


def create_volumes_from_surface_pairs(surface_tags_upper, surface_tags_lower, transfinite=True, verbose=True):
    """
    Creates volumes by looping over all pairs of upper and lower surfaces.
    """
    
    gmsh.model.geo.synchronize()
    volume_tags = []
    
    for i in range(len(surface_tags_upper)):
        # Get the 6 surfaces
        surface_tags = get_repeating_surfaces(
            surface_tags_upper[i],
            surface_tags_lower[i])
              
        # Create volume
        surface_loop = gmsh.model.geo.addSurfaceLoop(surface_tags)
        volume_tag = gmsh.model.geo.addVolume([surface_loop])
        gmsh.model.geo.synchronize()
        
        # Apply mesh settings
        if transfinite:
            gmsh.model.geo.mesh.setTransfiniteVolume(volume_tag)
        gmsh.model.mesh.setRecombine(3, volume_tag, 90)
        gmsh.model.geo.synchronize()
        
        volume_tags.append(volume_tag)

    return volume_tags

def create_surface_from_points(point_tags, transfinite=True, surface_type="plane"):
    """
    Creates a surface from 4 points with automatic curve finding and meshing.
    
    Parameters:
    -----------
    point_tags : list
        List of 4 point tags in order [pt1, pt2, pt3, pt4]
    transfinite : bool
        Whether to apply transfinite meshing (default: True)
    
    Returns:
    --------
    int : Surface tag
    """
    
    gmsh.model.geo.synchronize()
    
    # Find oriented curves connecting the points
    curves = []
    for i in range(4):
        pt1, pt2 = point_tags[i], point_tags[(i + 1) % 4]

        for curve_tag in gmsh.model.getAdjacencies(0, pt1)[0]:
            curve_pts = gmsh.model.getAdjacencies(1, curve_tag)[1]
            
            if pt2 in curve_pts:
                if curve_pts[0] == pt1:
                    curves.append(curve_tag)
                else:
                    curves.append(-curve_tag)
                break

    # Create loop and surface
    loop = gmsh.model.geo.addCurveLoop(curves)
    # Create surface based on type
    if surface_type == "filling":
        surface = gmsh.model.geo.addSurfaceFilling([loop])
    else:  # default to "plane"
        surface = gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.synchronize()
    
    # Apply mesh settings
    if transfinite:
        gmsh.model.geo.mesh.setTransfiniteSurface(surface)
    gmsh.model.geo.mesh.setRecombine(2, surface, 90)
    gmsh.model.geo.synchronize()
    
    return surface

def assign_surfaces_to_physical_groups(p, outer_C_surf_tag, eps=1e-6):
    """
    Assigns surfaces to physical groups.
    
    Parameters:
    -----------
    p : Params
        Parameters object
    outer_C_surf_tag : int
        The tag of the outer C-shell surface (from your geometry creation)
    eps : float
        Small epsilon for bounding box tolerance
    """
    gmsh.model.geo.synchronize()
    pg_inlet = gmsh.model.addPhysicalGroup(2, outer_C_surf_tag)
    gmsh.model.setPhysicalName(2, pg_inlet, "Inlet")
    groups = {
        "Outlet": (p.Lx-eps, -p.Ly-eps, -eps, p.Lx+eps, p.Ly+eps, p.W+eps),
        "Airfoil": (-0.1*p.c, -0.1*p.c, -eps, 1.1*p.c, 0.1*p.c, p.W+eps),
        "Bottom": (-p.Ly-eps, -p.Ly-eps, -eps, p.Lx+eps, -p.Ly+eps, p.W+eps),
        "Top": (-p.Ly-eps, p.Ly-eps, -eps, p.Lx+eps, p.Ly+eps, p.W+eps),
        "Side1": (-p.Ly-eps, -p.Ly-eps, -eps, p.Lx+eps, p.Ly+eps, eps),
        "Side2": (-p.Ly-eps, -p.Ly-eps, p.W-eps, p.Lx+eps, p.Ly+eps, p.W+eps),
    }
    
    # Assign bounding box groups
    for name, bbox in groups.items():
        surfs = gmsh.model.getEntitiesInBoundingBox(*bbox, 2)
        if surfs:
            pg = gmsh.model.addPhysicalGroup(2, [s[1] for s in surfs])
            gmsh.model.setPhysicalName(2, pg, name)
    
    gmsh.model.geo.synchronize()

def create_volume_from_nested_pairs(left_surf_tags, right_surf_tags, left_pt_tags, rigt_pt_tags,
                                    translate, num_ele, coeff=1,oriented_down=False):
    """
    Build connector lines, patch surfaces and volume between a 'left' set (TE_upper)
    and a 'right' set (wake_upper). If 'translate' != 0, top and inner patches are created
    as 'filling' to better accommodate deformation.

    Args:
      left_surf_tags:    list of left surface tags (TE side)
      right_surf_tags:   list of right surface tags (wake side) or None to auto-build
      left_pt_tags:      list of left boundary point tags
      rigt_pt_tags:      list of right boundary point tags or None to auto-build
      translate:         float, Y-translation applied to the 'top' right surface (index 3)
      num_ele:           int, transfinite subdivision on connector lines
      coeff:             float, scale factor applied to 'translate'

    Returns:
      dict with volume_tag and created patch tags.
    """

    # Use generic names internally
    leftSurfaces = left_surf_tags
    leftPoints   = left_pt_tags

    rightSurfaces = right_surf_tags
    rightPoints   = rigt_pt_tags

    # Optional: translate the "top" right surface (index 3) in Y
    if translate:
        pts_to_translate = gmsh.model.getBoundary([(2, rightSurfaces[3])], recursive=True)
        gmsh.model.geo.translate(pts_to_translate, 0, translate, 0)
        gmsh.model.geo.synchronize()
    
    # Connector lines (left -> right) with transfinite constraints
    idxs = [2, 3, 6, 7] if oriented_down else range(len(rightPoints))
    for i in idxs:
        line = gmsh.model.geo.addLine(leftPoints[i], rightPoints[i])
        # If you want biased spacing, use "Progression"; otherwise omit meshType/coef
        gmsh.model.geo.mesh.setTransfiniteCurve(line, num_ele, meshType="Progression", coef=coeff)

    gmsh.model.geo.synchronize()

    # Patch surfaces (exact index pattern preserved)
    _ = create_surface_from_points([leftPoints[0], rightPoints[0], rightPoints[3], leftPoints[3]])
    _ = create_surface_from_points([leftPoints[4], rightPoints[4], rightPoints[7], leftPoints[7]])
    _ = create_surface_from_points([leftPoints[5], rightPoints[5], rightPoints[6], leftPoints[6]])
    _ = create_surface_from_points([leftPoints[1], rightPoints[1], rightPoints[2], leftPoints[2]])

    if oriented_down == False:
        _ = create_surface_from_points([leftPoints[0], rightPoints[0], rightPoints[4], leftPoints[4]])
        _ = create_surface_from_points([leftPoints[4], rightPoints[4], rightPoints[5], leftPoints[5]])
        _ = create_surface_from_points([leftPoints[5], rightPoints[5], rightPoints[1], leftPoints[1]])

    # Top and inner patches: 'filling' iff a translation is applied
    topSurf = create_surface_from_points([leftPoints[3], rightPoints[3], rightPoints[2], leftPoints[2]],
                                        surface_type=("filling" if translate else "plane"))
    innerSurf = create_surface_from_points([leftPoints[7], rightPoints[7], rightPoints[6], leftPoints[6]],
                                        surface_type=("filling" if translate else "plane"))

    # Corner/angle patches (always 'filling')
    leftCornerSurf  = create_surface_from_points([leftPoints[3], rightPoints[3], rightPoints[7], leftPoints[7]],
                                                 surface_type="filling")
    rightCornerSurf = create_surface_from_points([leftPoints[6], rightPoints[6], rightPoints[2], leftPoints[2]],
                                                 surface_type="filling")

    gmsh.model.geo.synchronize()

    # Create volumes between the left and right surface sets
    volume_tag = create_volumes_from_surface_pairs(leftSurfaces, rightSurfaces)
    return volume_tag, topSurf

def copy_upper_to_lower(upper, lower,idxs=(0, 1, 4, 5)):
    for i in idxs:
        lower[i] = upper[i]
    return lower, upper


# def offset_curve(x, y, d, outward_sign=+1):
#     """
#     Compute a parallel curve offset by distance d along the local surface normals.
#     outward_sign: +1 or -1 to choose outward/inward direction.
#     """
#     x = np.asarray(x, dtype=float)
#     y = np.asarray(y, dtype=float)

#     # Numerical tangent via gradients (robust on non-uniform x if sorted)
#     dx = np.gradient(x,edge_order=2)
#     dy = np.gradient(y,edge_order=2)
#     tlen = np.hypot(dx, dy)
#     # Avoid division by zero
#     tlen[tlen == 0.0] = 1.0

#     # Unit tangent (tx, ty) and unit normal (nx, ny)
#     tx = dx / tlen
#     ty = dy / tlen
#     nx = ty
#     ny =  -tx
#     # angles_rad = np.arctan2(ty, tx)
#     # angles_deg = np.degrees(angles_rad)

#     # for i, x_c in enumerate(x):
#     #     print(f"{i}: x= {x_c} y= {y[i]} angle(deg)= {np.pi + angles_rad[i]:.3f}")
#     # exit()
#     # Offset
#     x_off = x + outward_sign * d * nx
#     y_off = y + outward_sign * d * ny
#     return x_off, y_off

# def build_scaled_nose_spline(p_upper_tag, p_lower_tag, d, z, p):
    
#     x_upper,y_upper,_ = _pt_xyz(p_upper_tag)

#     iU_cut, x_cut, _ = nearest_station_from_upper(p.xu, p.yu, p.x_th)
#     iL_cut, _,    _ = nearest_station_from_lower(p.xl, p.yl, x_cut)

#     # Clamp indices
#     iU = max(1, min(len(p.xu) - 2, iU_cut))
#     iL = max(1, min(len(p.xl) - 2, iL_cut))

#     xU_seg = p.xu[iU+1:]    
#     yU_seg = p.yu[iU+1:]
    
#     xL_seg = p.xl[1:iL]
#     yL_seg = p.yl[1:iL]
#     d = np.sqrt((x_upper -p.xu[iU] )**2 + (y_upper - p.yu[iU])**2)
#     if xU_seg[0] < xU_seg[-1]:
#         xU_seg = xU_seg[::-1]
#         yU_seg = yU_seg[::-1]

#     if xL_seg[0] > xL_seg[-1]:
#         xL_seg = xL_seg[::-1]
#         yL_seg = yL_seg[::-1]

#     x_all = np.concatenate([xU_seg, xL_seg])
#     y_all = np.concatenate([yU_seg, yL_seg])

#     x_off, y_off = offset_curve(x_all, y_all, d)

#     tags = [gmsh.model.geo.addPoint(float(x), float(y), float(z))
#             for x, y in zip(x_off, y_off)]

#     gmsh.model.geo.synchronize()
#     new_spline_tag = gmsh.model.geo.addSpline([p_upper_tag] + tags + [p_lower_tag])
#     gmsh.model.geo.synchronize()
#     return new_spline_tag

def outward_normal_on_outer_surface(xparam, m=0.04, p=0.4, t=0.12, closed_TE=True, eps=1e-12):
    """
    Returns outward normals evaluated at chord-parameter xparam:
      nux,nuy for upper, nlx,nly for lower
    plus the surfaces xu,yu,xl,yl (optional but useful for sanity checks).
    """
    x = np.asarray(xparam, dtype=float)
    xe = np.maximum(x, eps)

    a4 = 0.1036 if closed_TE else 0.1015

    yt = 5*t*(0.2969*np.sqrt(xe) - 0.1260*xe - 0.3516*xe**2 + 0.2843*xe**3 - a4*xe**4)
    yt_p = 5*t*(0.2969*(1.0/(2.0*np.sqrt(xe))) - 0.1260 - 0.7032*xe + 0.8529*xe**2 - 4.0*a4*xe**3)

    yc  = np.where(x < p,
                   m/p**2*(2*p*x - x**2),
                   m/(1-p)**2*((1-2*p) + 2*p*x - x**2))
    dyc = np.where(x < p,
                   2*m/p**2*(p - x),
                   2*m/(1-p)**2*(p - x))
    d2yc = np.where(x < p,
                    -2*m/p**2,
                    -2*m/(1-p)**2)

    th   = np.arctan(dyc)
    th_p = d2yc / (1.0 + dyc**2)

    s = np.sin(th)
    c = np.cos(th)

    # surfaces
    xu = x - yt*s
    yu = yc + yt*c
    xl = x + yt*s
    yl = yc - yt*c

    # parametric derivatives wrt xparam
    xu_p = 1.0 - yt_p*s - yt*c*th_p
    yu_p = dyc + yt_p*c - yt*s*th_p

    xl_p = 1.0 + yt_p*s + yt*c*th_p
    yl_p = dyc - yt_p*c + yt*s*th_p

    su = np.hypot(xu_p, yu_p)
    sl = np.hypot(xl_p, yl_p)
    su = np.where(su == 0.0, 1.0, su)
    sl = np.where(sl == 0.0, 1.0, sl)

    # OUTWARD normals (consistent with increasing parameter direction)
    nux = -yu_p / su
    nuy =  xu_p / su

    nlx =  yl_p / sl
    nly = -xl_p / sl

    return (xu, yu, nux, nuy), (xl, yl, nlx, nly)


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


def build_spline_from_pt_tags(pt_tags, tag_begin, tag_end, z_shift):

    if len(pt_tags) < 2:
        raise ValueError("pt_tags must contain at least 2 points (start and end).")
    spline_tags = [tag_begin]
    interior = pt_tags[:-1]
    for i, t in enumerate(interior):
        x,y,z = _pt_xyz(t)
        new_x = x 
        new_y = y 
        new_z = z + z_shift
        tag = gmsh.model.geo.addPoint(new_x, new_y, new_z)
        spline_tags.append(tag)
    spline_tags.append(tag_end)
    new_spline_tag = gmsh.model.geo.addSpline(spline_tags)
    gmsh.model.geo.synchronize()
    return new_spline_tag

gmsh.initialize()
gmsh.model.add("nested_at_x")
script_dir = os.path.dirname(os.path.abspath(__file__))
p = Params()
# --- Trailing and Leading Edge Points ---
xU_cos = cosine_spacing(p.Nu + 1, 1.0, 0.0)
xL_cos = cosine_spacing(p.Nl + 1, 0.0, 1.0)
xu, yu, _, _ = naca4412(xU_cos)
_, _, xl, yl = naca4412(xL_cos)

left_airfoil_curve_upper, left_airfoil_curve_nose, left_airfoil_curve_lower, airfoil_split_tags, upper_pt_tags, nose_pt_tags, lower_pt_tags = build_three_airfoil_splines(xu, yu, xl, yl, p, make_wire=False)
gmsh.model.geo.synchronize()

# -----------------------------------------------------------------------
# C nested
# -----------------------------------------------------------------------
surface_tags_10c_upper, curve_tags_10c_upper, pts_tags_10c_upper = nested_mesh_yz_transfinite_at_x(p,airfoil_split_tags[1],orientation_down=0)
surface_tags_10c_lower, curve_tags_10c_lower, pts_tags_10c_lower = nested_mesh_yz_transfinite_at_x(p,airfoil_split_tags[3],orientation_down=1)


airfoil_split_upper = _pt_xyz(pts_tags_10c_upper[3])
airfoil_split_lower = _pt_xyz(pts_tags_10c_lower[3])
x_arc_center = (airfoil_split_upper[0] + airfoil_split_lower[0]) / 2
y_arc_center = (airfoil_split_upper[1] + airfoil_split_lower[1]) / 2
z_arc_center = 0.0
arc_ceter_pt_tag_1 = gmsh.model.geo.addPoint(x_arc_center,y_arc_center,z_arc_center)
gmsh.model.geo.synchronize()

# arc_tag_1 = gmsh.model.geo.addCircleArc(pts_tags_10c_upper[3], arc_ceter_pt_tag_1, pts_tags_10c_lower[3])
upper_tag = pts_tags_10c_upper[3]
lower_tag = pts_tags_10c_lower[3]
arc_tag_1 = build_scaled_nose_spline(upper_tag,lower_tag,p.H,0.0,p)
gmsh.model.geo.remove([(0,arc_ceter_pt_tag_1)])
gmsh.model.geo.synchronize()
gmsh.model.geo.mesh.setTransfiniteCurve(arc_tag_1,p.nose_surface,"Bump",p.nose_bump_coeff)
circ1_surf = create_surface_from_points([pts_tags_10c_upper[3],pts_tags_10c_lower[3],pts_tags_10c_lower[0],pts_tags_10c_upper[0]])


airfoil_split_upper = _pt_xyz(pts_tags_10c_upper[7])
airfoil_split_lower = _pt_xyz(pts_tags_10c_lower[7])
x_arc_center = (airfoil_split_upper[0] + airfoil_split_lower[0]) / 2
y_arc_center = (airfoil_split_upper[1] + airfoil_split_lower[1]) / 2
z_arc_center = p.z1
arc_ceter_pt_tag = gmsh.model.geo.addPoint(x_arc_center,y_arc_center,z_arc_center)
gmsh.model.geo.synchronize()

# arc_tag_2 = gmsh.model.geo.addCircleArc(pts_tags_10c_upper[7], arc_ceter_pt_tag, pts_tags_10c_lower[7])
upper_tag = pts_tags_10c_upper[7]
lower_tag = pts_tags_10c_lower[7]
arc_tag_2 = build_scaled_nose_spline(upper_tag,lower_tag,p.h_in,p.z1,p)
gmsh.model.geo.mesh.setTransfiniteCurve(arc_tag_2,p.nose_surface,"Bump",p.nose_bump_coeff)
gmsh.model.geo.remove([(0,arc_ceter_pt_tag)])
gmsh.model.geo.synchronize()
# nose_airfoil_extrude_1 = gmsh.model.geo.extrude([(1,left_airfoil_curve_nose)],0.0,0.0,p.z1,numElements=[p.nz_left_right-1],recombine=True)
# nose_airfoil_extrude_1_spline = nose_airfoil_extrude_1[0][1]
# gmsh.model.geo.mesh.setTransfiniteCurve(nose_airfoil_extrude_1_spline,p.nose_surface,"Bump",p.nose_bump_coeff)
# nose_airfoil_extrude_left_surface = nose_airfoil_extrude_1[1][1]
# gmsh.model.geo.mesh.setTransfiniteSurface(nose_airfoil_extrude_left_surface)
# gmsh.model.geo.mesh.setRecombine(2,nose_airfoil_extrude_left_surface,90)
# nose_airfoil_extrude_1_spline = build_spline_from_pt_tags(nose_pt_tags,pts_tags_10c_upper[4], pts_tags_10c_lower[4],p.z1)
nose_airfoil_extrude_1_spline = gmsh.model.geo.copy([(1,left_airfoil_curve_nose)])
gmsh.model.geo.translate(nose_airfoil_extrude_1_spline,0,0,p.z1)
gmsh.model.geo.mesh.setTransfiniteCurve(nose_airfoil_extrude_1_spline[0][1],p.nose_surface,"Bump",p.nose_bump_coeff)
nose_airfoil_extrude_left_surface = create_surface_from_points([pts_tags_10c_upper[0],pts_tags_10c_lower[0],pts_tags_10c_lower[4],pts_tags_10c_upper[4]],surface_type="filling")
gmsh.model.geo.synchronize()
circ2_surf = create_surface_from_points([pts_tags_10c_upper[4],pts_tags_10c_lower[4],pts_tags_10c_lower[7],pts_tags_10c_upper[7]])

airfoil_split_upper = _pt_xyz(pts_tags_10c_upper[6])
airfoil_split_lower = _pt_xyz(pts_tags_10c_lower[6])
x_arc_center = (airfoil_split_upper[0] + airfoil_split_lower[0]) / 2
y_arc_center = (airfoil_split_upper[1] + airfoil_split_lower[1]) / 2
z_arc_center = p.z2
arc_ceter_pt_tag = gmsh.model.geo.addPoint(x_arc_center,y_arc_center,z_arc_center)
gmsh.model.geo.synchronize()

# arc_tag_3 = gmsh.model.geo.addCircleArc(pts_tags_10c_upper[6], arc_ceter_pt_tag, pts_tags_10c_lower[6])
upper_tag = pts_tags_10c_upper[6]
lower_tag = pts_tags_10c_lower[6]
arc_tag_3 = build_scaled_nose_spline(upper_tag,lower_tag,p.h_in,p.z2,p)
gmsh.model.geo.remove([(0,arc_ceter_pt_tag)])
gmsh.model.geo.synchronize()
gmsh.model.geo.mesh.setTransfiniteCurve(arc_tag_3,p.nose_surface,"Bump",p.nose_bump_coeff)
# nose_airfoil_extrude_2 = gmsh.model.geo.extrude([(1,nose_airfoil_extrude_1_spline)],0.0,0.0,p.z2-p.z1,numElements=[p.nz_mid-1],recombine=True)
# nose_airfoil_extrude_2_spline = nose_airfoil_extrude_2[0][1]
# gmsh.model.geo.mesh.setTransfiniteCurve(nose_airfoil_extrude_2_spline,p.nose_surface,"Bump",p.nose_bump_coeff)
# nose_airfoil_extrude_mid_surface = nose_airfoil_extrude_2[1][1]
# gmsh.model.geo.mesh.setTransfiniteSurface(nose_airfoil_extrude_mid_surface)
# gmsh.model.geo.mesh.setRecombine(2,nose_airfoil_extrude_mid_surface,90)
# gmsh.model.geo.synchronize()
# nose_airfoil_extrude_2_spline = build_spline_from_pt_tags(nose_pt_tags,pts_tags_10c_upper[5], pts_tags_10c_lower[5],p.z2)
nose_airfoil_extrude_2_spline = gmsh.model.geo.copy([(1,left_airfoil_curve_nose)])
gmsh.model.geo.translate(nose_airfoil_extrude_2_spline,0,0,p.z2)
gmsh.model.geo.mesh.setTransfiniteCurve(nose_airfoil_extrude_2_spline[0][1],p.nose_surface,"Bump",p.nose_bump_coeff)
nose_airfoil_extrude_mid_surface = create_surface_from_points([pts_tags_10c_upper[4],pts_tags_10c_lower[4],pts_tags_10c_lower[5],pts_tags_10c_upper[5]],surface_type="filling")
gmsh.model.geo.synchronize()
circ3_surf = create_surface_from_points([pts_tags_10c_upper[5],pts_tags_10c_lower[5],pts_tags_10c_lower[6],pts_tags_10c_upper[6]])

airfoil_split_upper = _pt_xyz(pts_tags_10c_upper[2])
airfoil_split_lower = _pt_xyz(pts_tags_10c_lower[2])
x_arc_center = (airfoil_split_upper[0] + airfoil_split_lower[0]) / 2
y_arc_center = (airfoil_split_upper[1] + airfoil_split_lower[1]) / 2
z_arc_center = p.W
arc_ceter_pt_tag_4 = gmsh.model.geo.addPoint(x_arc_center,y_arc_center,z_arc_center)
gmsh.model.geo.synchronize()

# arc_tag_4 = gmsh.model.geo.addCircleArc(pts_tags_10c_upper[2], arc_ceter_pt_tag_4, pts_tags_10c_lower[2])
upper_tag = pts_tags_10c_upper[2]
lower_tag = pts_tags_10c_lower[2]
arc_tag_4 = build_scaled_nose_spline(upper_tag,lower_tag,p.H,p.W,p)
gmsh.model.geo.remove([(0,arc_ceter_pt_tag_4)])
gmsh.model.geo.synchronize()
gmsh.model.geo.mesh.setTransfiniteCurve(arc_tag_4,p.nose_surface,"Bump",p.nose_bump_coeff)
# nose_airfoil_extrude_right = gmsh.model.geo.extrude([(1,nose_airfoil_extrude_2_spline)],0.0,0.0,p.W-p.z2,numElements=[p.nz_left_right-1],recombine=True)
# nose_airfoil_extrude_right_spline = nose_airfoil_extrude_right[0][1]
# gmsh.model.geo.mesh.setTransfiniteCurve(nose_airfoil_extrude_right_spline,p.nose_surface,"Bump",p.nose_bump_coeff)
# nose_airfoil_extrude_right_surface = nose_airfoil_extrude_right[1][1]
# gmsh.model.geo.mesh.setTransfiniteSurface(nose_airfoil_extrude_right_surface)
# gmsh.model.geo.mesh.setRecombine(2,nose_airfoil_extrude_right_surface,90)
# gmsh.model.geo.synchronize()
# nose_airfoil_extrude_right_spline = build_spline_from_pt_tags(nose_pt_tags,pts_tags_10c_upper[1], pts_tags_10c_lower[1],p.W)
nose_airfoil_extrude_right_spline = gmsh.model.geo.copy([(1,left_airfoil_curve_nose)])
gmsh.model.geo.translate(nose_airfoil_extrude_right_spline,0,0,p.W)
gmsh.model.geo.mesh.setTransfiniteCurve(nose_airfoil_extrude_right_spline[0][1],p.nose_surface,"Bump",p.nose_bump_coeff)
nose_airfoil_extrude_right_surface = create_surface_from_points([pts_tags_10c_upper[5],pts_tags_10c_lower[5],pts_tags_10c_lower[1],pts_tags_10c_upper[1]],surface_type="filling")
gmsh.model.geo.synchronize()
circ4_surf = create_surface_from_points([pts_tags_10c_upper[1],pts_tags_10c_lower[1],pts_tags_10c_lower[2],pts_tags_10c_upper[2]])

outer_C_surf = create_surface_from_points([pts_tags_10c_upper[3],pts_tags_10c_lower[3],pts_tags_10c_lower[2],pts_tags_10c_upper[2]],surface_type="filling")
inner_C_surf = create_surface_from_points([pts_tags_10c_upper[7],pts_tags_10c_lower[7],pts_tags_10c_lower[6],pts_tags_10c_upper[6]],surface_type="filling")     
left_angle_C_surf = create_surface_from_points([pts_tags_10c_upper[3],pts_tags_10c_lower[3],pts_tags_10c_lower[7],pts_tags_10c_upper[7]],surface_type="filling")
right_angle_C_surf = create_surface_from_points([pts_tags_10c_upper[2],pts_tags_10c_lower[2],pts_tags_10c_lower[6],pts_tags_10c_upper[6]],surface_type="filling")

volume_tags_C_nested = create_volumes_from_surface_pairs(surface_tags_10c_upper, surface_tags_10c_lower)

# -----------------------------------------------------------------------
# Top nested
# -----------------------------------------------------------------------

surface_tags_TE_upper,  curve_tags_TE_upper, pts_tags_TE_upper = nested_mesh_yz_transfinite_at_x(p,airfoil_split_tags[0],orientation_down=0)

upper_line_1 = gmsh.model.geo.addLine(pts_tags_10c_upper[3],pts_tags_TE_upper[3])
gmsh.model.geo.synchronize()
gmsh.model.geo.mesh.setTransfiniteCurve(upper_line_1,p.upper_surface)
upper_surf_left = create_surface_from_points([pts_tags_10c_upper[0],pts_tags_TE_upper[0],pts_tags_TE_upper[3],pts_tags_10c_upper[3]])

upper_line_2 = gmsh.model.geo.addLine(pts_tags_10c_upper[7], pts_tags_TE_upper[7])
gmsh.model.geo.synchronize()
gmsh.model.geo.mesh.setTransfiniteCurve(upper_line_2,p.upper_surface)
upper_airfoil_extrude_1 = gmsh.model.geo.extrude([(1,left_airfoil_curve_upper)],0.0,0.0,p.z1,numElements=[p.nz_left_right-1],recombine=True)
upepr_airfoil_extrude_1_spline = upper_airfoil_extrude_1[0][1]
upper_airfoil_extrude_left_surface = upper_airfoil_extrude_1[1][1]
gmsh.model.geo.mesh.setTransfiniteSurface(upper_airfoil_extrude_left_surface)
gmsh.model.geo.mesh.setRecombine(2,upper_airfoil_extrude_left_surface,90)
gmsh.model.geo.synchronize()

# upepr_airfoil_extrude_1_spline = build_spline_from_pt_tags(upper_pt_tags, pts_tags_TE_upper[4],pts_tags_10c_upper[4],p.z1)
# upepr_airfoil_extrude_1_spline = gmsh.model.geo.copy([(1,left_airfoil_curve_upper)])
# gmsh.model.geo.translate(upepr_airfoil_extrude_1_spline,0,0,p.z1)
# gmsh.model.geo.mesh.setTransfiniteCurve(upepr_airfoil_extrude_1_spline[0][1],p.upper_surface)
# nose_airfoil_extrude_left_surface = create_surface_from_points([pts_tags_10c_upper[0],pts_tags_TE_upper[0],pts_tags_TE_upper[4],pts_tags_10c_upper[4]],surface_type="filling")
# gmsh.model.geo.synchronize()
upper_surf_1 = create_surface_from_points([pts_tags_10c_upper[7],pts_tags_TE_upper[7],pts_tags_TE_upper[4],pts_tags_10c_upper[4]])

upper_line_3 = gmsh.model.geo.addLine(pts_tags_10c_upper[6], pts_tags_TE_upper[6])
gmsh.model.geo.synchronize()
gmsh.model.geo.mesh.setTransfiniteCurve(upper_line_3,p.upper_surface)
upper_airfoil_extrude_2 = gmsh.model.geo.extrude([(1,upepr_airfoil_extrude_1_spline)],0.0,0.0,p.z2-p.z1,numElements=[p.nz_mid-1],recombine=True)
upper_airfoil_extrude_2_spline = upper_airfoil_extrude_2[0][1]
upper_airfoil_extrude_mid_surface = upper_airfoil_extrude_2[1][1]
gmsh.model.geo.mesh.setTransfiniteSurface(upper_airfoil_extrude_mid_surface)
gmsh.model.geo.mesh.setRecombine(2,upper_airfoil_extrude_mid_surface,90)
gmsh.model.geo.synchronize()
# upper_airfoil_extrude_2_spline = build_spline_from_pt_tags(upper_pt_tags,pts_tags_TE_upper[5], pts_tags_10c_upper[5],p.z2)
# upper_airfoil_extrude_2_spline = gmsh.model.geo.copy([(1,left_airfoil_curve_upper)])
# gmsh.model.geo.translate(upper_airfoil_extrude_2_spline,0,0,p.z2)
# gmsh.model.geo.mesh.setTransfiniteCurve(upper_airfoil_extrude_2_spline[0][1],p.upper_surface)
# upper_airfoil_extrude_mid_surface = create_surface_from_points([pts_tags_10c_upper[4],pts_tags_TE_upper[4],pts_tags_TE_upper[5],pts_tags_10c_upper[5]],surface_type="filling")
# gmsh.model.geo.synchronize()
upper_surf_2 = create_surface_from_points([pts_tags_10c_upper[5],pts_tags_TE_upper[5],pts_tags_TE_upper[6],pts_tags_10c_upper[6]])

upper_line4 = gmsh.model.geo.addLine(pts_tags_10c_upper[2], pts_tags_TE_upper[2])
gmsh.model.geo.synchronize()
gmsh.model.geo.mesh.setTransfiniteCurve(upper_line4,p.upper_surface)
upper_airfoil_extrude_right = gmsh.model.geo.extrude([(1,upper_airfoil_extrude_2_spline)],0.0,0.0,p.W-p.z2,numElements=[p.nz_left_right-1],recombine=True)
upper_airfoil_extrude_right_spline = upper_airfoil_extrude_right[0][1]
upper_airfoil_extrude_right_surface = upper_airfoil_extrude_right[1][1]
gmsh.model.geo.mesh.setTransfiniteSurface(upper_airfoil_extrude_right_surface)
gmsh.model.geo.mesh.setRecombine(2,upper_airfoil_extrude_right_surface,90)
gmsh.model.geo.synchronize()
# upper_airfoil_extrude_right_spline = build_spline_from_pt_tags(upper_pt_tags,pts_tags_TE_upper[1], pts_tags_10c_upper[1],p.W)
# upper_airfoil_extrude_right_spline = gmsh.model.geo.copy([(1,left_airfoil_curve_upper)])
# gmsh.model.geo.translate(upper_airfoil_extrude_right_spline,0,0,p.W)
# gmsh.model.geo.mesh.setTransfiniteCurve(upper_airfoil_extrude_right_spline[0][1],p.upper_surface)
# upper_airfoil_extrude_mid_surface = create_surface_from_points([pts_tags_10c_upper[5],pts_tags_TE_upper[5],pts_tags_TE_upper[1],pts_tags_10c_upper[1]],surface_type="filling")
# gmsh.model.geo.synchronize()
upper_surf_right = create_surface_from_points([pts_tags_10c_upper[2],pts_tags_TE_upper[2],pts_tags_TE_upper[1],pts_tags_10c_upper[1]])

upper_top_surf = create_surface_from_points([pts_tags_10c_upper[3],pts_tags_TE_upper[3],pts_tags_TE_upper[2],pts_tags_10c_upper[2]])
upper_inner_surf = create_surface_from_points([pts_tags_10c_upper[7],pts_tags_TE_upper[7],pts_tags_TE_upper[6],pts_tags_10c_upper[6]])
upper_left_angle_surf = create_surface_from_points([pts_tags_10c_upper[3],pts_tags_TE_upper[3],pts_tags_TE_upper[7],pts_tags_10c_upper[7]],surface_type="filling")
upper_right_angle_surf = create_surface_from_points([pts_tags_10c_upper[6],pts_tags_TE_upper[6],pts_tags_TE_upper[2],pts_tags_10c_upper[2]],surface_type="filling")

volume_tags_upper_nested = create_volumes_from_surface_pairs(surface_tags_10c_upper, surface_tags_TE_upper)

# -----------------------------------------------------------------------
# Bottom nested
# -----------------------------------------------------------------------

surface_tags_TE_lower,  curve_tags_TE_lower, pts_tags_TE_lower = nested_mesh_yz_transfinite_at_x(p,airfoil_split_tags[0],orientation_down=1)

lower_line_1 = gmsh.model.geo.addLine(pts_tags_10c_lower[3],pts_tags_TE_lower[3])
gmsh.model.geo.synchronize()
gmsh.model.geo.mesh.setTransfiniteCurve(lower_line_1,p.lower_surface)
lower_surf_left = create_surface_from_points([pts_tags_10c_lower[0],pts_tags_TE_lower[0],pts_tags_TE_lower[3],pts_tags_10c_lower[3]])

lower_line_2 = gmsh.model.geo.addLine(pts_tags_10c_lower[7], pts_tags_TE_lower[7])
gmsh.model.geo.synchronize()
gmsh.model.geo.mesh.setTransfiniteCurve(lower_line_2,p.lower_surface)
lower_airfoil_extrude_1 = gmsh.model.geo.extrude([(1,left_airfoil_curve_lower)],0.0,0.0,p.z1,numElements=[p.nz_left_right-1],recombine=True)
lower_airfoil_extrude_1_spline = lower_airfoil_extrude_1[0][1]
lower_airfoil_extrude_left_surface = lower_airfoil_extrude_1[1][1]
gmsh.model.geo.mesh.setTransfiniteSurface(lower_airfoil_extrude_left_surface)
gmsh.model.geo.mesh.setRecombine(2,lower_airfoil_extrude_left_surface,90)
gmsh.model.geo.synchronize()
lower_surf_1 = create_surface_from_points([pts_tags_10c_lower[7],pts_tags_TE_lower[7],pts_tags_TE_upper[4],pts_tags_10c_lower[4]])

lower_line_3 = gmsh.model.geo.addLine(pts_tags_10c_lower[6], pts_tags_TE_lower[6])
gmsh.model.geo.synchronize()
gmsh.model.geo.mesh.setTransfiniteCurve(lower_line_3,p.lower_surface)
lower_airfoil_extrude_2 = gmsh.model.geo.extrude([(1,lower_airfoil_extrude_1_spline)],0.0,0.0,p.z2-p.z1,numElements=[p.nz_mid-1],recombine=True)
lower_airfoil_extrude_2_spline = lower_airfoil_extrude_2[0][1]
lower_airfoil_extrude_mid_surface = lower_airfoil_extrude_2[1][1]
gmsh.model.geo.mesh.setTransfiniteSurface(lower_airfoil_extrude_mid_surface)
gmsh.model.geo.mesh.setRecombine(2,lower_airfoil_extrude_mid_surface,90)
gmsh.model.geo.synchronize()
lower_surf_2 = create_surface_from_points([pts_tags_10c_lower[5],pts_tags_TE_upper[5],pts_tags_TE_lower[6],pts_tags_10c_lower[6]])

lower_line4 = gmsh.model.geo.addLine(pts_tags_10c_lower[2], pts_tags_TE_lower[2])
gmsh.model.geo.synchronize()
gmsh.model.geo.mesh.setTransfiniteCurve(lower_line4,p.lower_surface)
lower_airfoil_extrude_right = gmsh.model.geo.extrude([(1,lower_airfoil_extrude_2_spline)],0.0,0.0,p.W-p.z2,numElements=[p.nz_left_right-1],recombine=True)
lower_airfoil_extrude_right_spline = lower_airfoil_extrude_right[0][1]
lower_airfoil_extrude_right_surface = lower_airfoil_extrude_right[1][1]
gmsh.model.geo.mesh.setTransfiniteSurface(lower_airfoil_extrude_right_surface)
gmsh.model.geo.mesh.setRecombine(2,lower_airfoil_extrude_right_surface,90)
gmsh.model.geo.synchronize()
lower_surf_right = create_surface_from_points([pts_tags_10c_lower[2],pts_tags_TE_lower[2],pts_tags_TE_upper[1],pts_tags_10c_lower[1]])

lower_top_surf = create_surface_from_points([pts_tags_10c_lower[3],pts_tags_TE_lower[3],pts_tags_TE_lower[2],pts_tags_10c_lower[2]])
lower_inner_surf = create_surface_from_points([pts_tags_10c_lower[7],pts_tags_TE_lower[7],pts_tags_TE_lower[6],pts_tags_10c_lower[6]])
lower_left_angle_surf = create_surface_from_points([pts_tags_10c_lower[3],pts_tags_TE_lower[3],pts_tags_TE_lower[7],pts_tags_10c_lower[7]],surface_type="filling")
lower_right_angle_surf = create_surface_from_points([pts_tags_10c_lower[6],pts_tags_TE_lower[6],pts_tags_TE_lower[2],pts_tags_10c_lower[2]],surface_type="filling")

volume_tags_lower_nested = create_volumes_from_surface_pairs(surface_tags_10c_lower, surface_tags_TE_lower)

# -----------------------------------------------------------------------
# Outer domain via extrusion
# -----------------------------------------------------------------------

# outer_top = gmsh.model.geo.extrude([(2,upper_top_surf)],0.0,p.Ly-p.H,0.0,numElements=p.outer_veritcal_ele,heights=p.outer_vertical_heights,recombine=True)
# outer_bottom = gmsh.model.geo.extrude([(2,lower_top_surf)],0.0,-p.Ly+p.H,0.0,numElements=p.outer_veritcal_ele,heights=p.outer_vertical_heights,recombine=True)
# gmsh.model.geo.synchronize()
# gmsh.model.geo.extrude([outer_top[3]],p.Lx-p.c,0.0,0.0,numElements=p.outer_horizontal_ele,heights=p.outer_horizontal_heigths,recombine=True)
# gmsh.model.geo.extrude([outer_bottom[3]],p.Lx-p.c,0.0,0.0,numElements=p.outer_horizontal_ele,heights=p.outer_horizontal_heigths,recombine=True)
# gmsh.model.geo.extrude([(2, surf) for surf in surface_tags_TE_upper],p.Lx-p.c,0.0,0.0,numElements=p.outer_horizontal_ele,heights=p.outer_horizontal_heigths,recombine=True)
# gmsh.model.geo.extrude([(2, surf) for surf in surface_tags_TE_lower],p.Lx-p.c,0.0,0.0,numElements=p.outer_horizontal_ele,heights=p.outer_horizontal_heigths,recombine=True)

# all_surfaces = gmsh.model.getEntities(2)
# for dim,tag in all_surfaces:
#     gmsh.model.geo.mesh.setTransfiniteSurface(tag)
#     gmsh.model.geo.mesh.setRecombine(2,tag,90)

# # corners = get_surface_corners(outer_top[-1][1])
# inlet_C_upper_tags = gmsh.model.getBoundary([outer_top[-1]],recursive=True)[-2:]
# inlet_C_lower_tags = gmsh.model.getBoundary([outer_bottom[-1]],recursive=True)[-2:]

# inlet_c_left = gmsh.model.geo.addCircleArc(inlet_C_upper_tags[0][1],arc_ceter_pt_tag_1,inlet_C_lower_tags[0][1])
# inlet_c_right = gmsh.model.geo.addCircleArc(inlet_C_upper_tags[1][1],arc_ceter_pt_tag_4,inlet_C_lower_tags[1][1])

# gmsh.model.geo.synchronize()
# gmsh.model.geo.mesh.setTransfiniteCurve(inlet_c_left,p.nose_surface)
# gmsh.model.geo.mesh.setTransfiniteCurve(inlet_c_right,p.nose_surface)
# inlet_c_surf = create_surface_from_points([inlet_C_upper_tags[0][1],inlet_C_lower_tags[0][1],inlet_C_lower_tags[1][1],inlet_C_upper_tags[1][1]],surface_type="filling")
# inlet_c_left_surf = create_surface_from_points([inlet_C_upper_tags[0][1],inlet_C_lower_tags[0][1],pts_tags_10c_lower[3],pts_tags_10c_upper[3]])
# inlet_c_right_surf = create_surface_from_points([inlet_C_upper_tags[1][1],inlet_C_lower_tags[1][1],pts_tags_10c_lower[2],pts_tags_10c_upper[2]])

# volume_tags_C_inlet = create_volumes_from_surface_pairs([outer_top[-1][1]],[outer_bottom[-1][1]])
# gmsh.model.geo.synchronize()
# gmsh.model.geo.removeAllDuplicates()

# -----------------------------------------------------------------------
# Outer domain near and far Wake
# -----------------------------------------------------------------------

wake_pt_tag = gmsh.model.geo.addPoint(p.c + (p.Lx-p.c)*p.near_wake_length,0,0)
gmsh.model.geo.synchronize()
surface_tags_wake_upper,  curve_tags_wake_upper, pts_tags_wake_upper = nested_mesh_yz_transfinite_at_x(p,wake_pt_tag,orientation_down=0)
_, near_wake_top_surf = create_volume_from_nested_pairs(surface_tags_TE_upper,surface_tags_wake_upper,pts_tags_TE_upper,pts_tags_wake_upper,0,p.Nx_near_wake)

surface_tags_wake_lower,  curve_tags_wake_lower, pts_tags_wake_lower = nested_mesh_yz_transfinite_at_x(p,wake_pt_tag,orientation_down=1)
gmsh.model.geo.removeAllDuplicates()
gmsh.model.geo.synchronize()

copy_upper_to_lower(pts_tags_wake_upper, pts_tags_wake_lower)
copy_upper_to_lower( pts_tags_TE_upper,   pts_tags_TE_lower)
_ ,near_wake_bottom_surf = create_volume_from_nested_pairs(surface_tags_TE_lower,surface_tags_wake_lower,pts_tags_TE_lower,pts_tags_wake_lower,0,p.Nx_near_wake,oriented_down=True)

far_wake_pt_tag = gmsh.model.geo.addPoint(p.Lx,0,0)
gmsh.model.geo.synchronize()

surface_tags_far_wake_upper,  curve_tags_far_wake_upper, pts_tags_far_wake_upper = nested_mesh_yz_transfinite_at_x(p,far_wake_pt_tag,orientation_down=0)
_, far_wake_top_surf = create_volume_from_nested_pairs(surface_tags_wake_upper,surface_tags_far_wake_upper,pts_tags_wake_upper,pts_tags_far_wake_upper,p.wake_stretch_in_Y,p.Nx_far_wake,p.y_outer_horizontal_progression)
if p.wake_stretch_in_Y > 0.0:
    tags = [3,5,6,8] 
    for tag in tags:
        gmsh.model.geo.mesh.setTransfiniteCurve(curve_tags_far_wake_upper[tag],p.ny)

surface_tags_far_wake_lower,  curve_tags_far_wake_lower, pts_tags_far_wake_lower = nested_mesh_yz_transfinite_at_x(p,far_wake_pt_tag,orientation_down=1)
gmsh.model.geo.removeAllDuplicates()
gmsh.model.geo.synchronize()
copy_upper_to_lower(pts_tags_far_wake_upper, pts_tags_far_wake_lower)
if p.wake_stretch_in_Y > 0.0:
    tags = [3,5,6,8] 
    for tag in tags:
        gmsh.model.geo.mesh.setTransfiniteCurve(curve_tags_far_wake_lower[tag],p.ny)

_, far_wake_bottom_surf = create_volume_from_nested_pairs(surface_tags_wake_lower,surface_tags_far_wake_lower,pts_tags_wake_lower,pts_tags_far_wake_lower,-p.wake_stretch_in_Y,p.Nx_far_wake,p.y_outer_horizontal_progression,oriented_down=True)

# -----------------------------------------------------------------------
# Outer domain top
# -----------------------------------------------------------------------

outer_top_0 = gmsh.model.geo.addPoint(0,p.Ly,0)
outer_top_1 = gmsh.model.geo.addPoint(p.c+p.near_wake_offset_in_x_1,p.Ly,0)
outer_top_2 = gmsh.model.geo.addPoint(p.c+p.near_wake_offset_in_x_1,p.Ly,p.W)
outer_top_3 = gmsh.model.geo.addPoint(0,p.Ly,p.W)
L1 = gmsh.model.geo.addLine(outer_top_0, outer_top_1)
gmsh.model.geo.mesh.setTransfiniteCurve(L1,nPoints=p.upper_surface)
L2 = gmsh.model.geo.addLine(outer_top_1, outer_top_2)
gmsh.model.geo.mesh.setTransfiniteCurve(L2,nPoints=p.nz_mid)
L3 = gmsh.model.geo.addLine(outer_top_2, outer_top_3)
gmsh.model.geo.mesh.setTransfiniteCurve(L3,nPoints=p.upper_surface)
L4 = gmsh.model.geo.addLine(outer_top_3, outer_top_0)
gmsh.model.geo.mesh.setTransfiniteCurve(L4,nPoints=p.nz_mid)
gmsh.model.geo.synchronize()

upper_top_surf_pts = gmsh.model.getBoundary([(2, upper_top_surf)], oriented=False, recursive=True)
upper_top_surf_pts = [tag for dim, tag in upper_top_surf_pts]


line = gmsh.model.geo.addLine(upper_top_surf_pts[0],outer_top_3 )
gmsh.model.geo.mesh.setTransfiniteCurve(line, p.Ny_outer,
                                    meshType="Progression",
                                    coef=p.y_outer_vertical_progression)
line = gmsh.model.geo.addLine(upper_top_surf_pts[1],outer_top_0)
gmsh.model.geo.mesh.setTransfiniteCurve(line, p.Ny_outer,
                                    meshType="Progression",
                                    coef=p.y_outer_vertical_progression)
line = gmsh.model.geo.addLine(upper_top_surf_pts[2],outer_top_2)
gmsh.model.geo.mesh.setTransfiniteCurve(line, p.Ny_outer,
                                    meshType="Progression",
                                    coef=p.y_outer_vertical_progression)
line = gmsh.model.geo.addLine(upper_top_surf_pts[3],outer_top_1)
gmsh.model.geo.mesh.setTransfiniteCurve(line, p.Ny_outer,
                                    meshType="Progression",
                                    coef=p.y_outer_vertical_progression)
outer_top_surf = create_surface_from_points([outer_top_0,outer_top_1,outer_top_2,outer_top_3])
inlet_circ_top_face = create_surface_from_points([outer_top_0,outer_top_3,upper_top_surf_pts[0],upper_top_surf_pts[1]],surface_type="filling")
_ = create_surface_from_points([outer_top_3,outer_top_2,upper_top_surf_pts[2],upper_top_surf_pts[0]])
_ = create_surface_from_points([outer_top_2,outer_top_1,upper_top_surf_pts[3],upper_top_surf_pts[2]],surface_type="filling")
_ = create_surface_from_points([outer_top_1,outer_top_0,upper_top_surf_pts[1],upper_top_surf_pts[3]])

_ = create_volumes_from_surface_pairs([outer_top_surf],[upper_top_surf])
gmsh.model.geo.synchronize()

# -----------------------------------------------------------------------
# Outer domain bottom
# -----------------------------------------------------------------------

outer_bot_0  = gmsh.model.geo.addPoint(0,   -p.Ly, 0)
outer_bot_1  = gmsh.model.geo.addPoint(p.c+p.near_wake_offset_in_x_1, -p.Ly, 0)
outer_bot_2 = gmsh.model.geo.addPoint(p.c+p.near_wake_offset_in_x_1, -p.Ly, p.W)
outer_bot_3 = gmsh.model.geo.addPoint(0,   -p.Ly, p.W)

B1 = gmsh.model.geo.addLine(outer_bot_0,  outer_bot_1)
gmsh.model.geo.mesh.setTransfiniteCurve(B1, nPoints=p.lower_surface)
B2 = gmsh.model.geo.addLine(outer_bot_1,  outer_bot_2)
gmsh.model.geo.mesh.setTransfiniteCurve(B2, nPoints=p.nz_mid)
B3 = gmsh.model.geo.addLine(outer_bot_2, outer_bot_3)
gmsh.model.geo.mesh.setTransfiniteCurve(B3, nPoints=p.lower_surface)
B4 = gmsh.model.geo.addLine(outer_bot_3, outer_bot_0)
gmsh.model.geo.mesh.setTransfiniteCurve(B4, nPoints=p.nz_mid)

lower_top_surf_pts = gmsh.model.getBoundary([(2, lower_top_surf)], oriented=False, recursive=True)
lower_top_surf_pts = [tag for dim, tag in lower_top_surf_pts]

line = gmsh.model.geo.addLine(lower_top_surf_pts[0],outer_bot_3 )
gmsh.model.geo.mesh.setTransfiniteCurve(line, p.Ny_outer,
                                    meshType="Progression",
                                    coef=p.y_outer_vertical_progression)
line = gmsh.model.geo.addLine(lower_top_surf_pts[1],outer_bot_0)
gmsh.model.geo.mesh.setTransfiniteCurve(line, p.Ny_outer,
                                    meshType="Progression",
                                    coef=p.y_outer_vertical_progression)
line = gmsh.model.geo.addLine(lower_top_surf_pts[2],outer_bot_2)
gmsh.model.geo.mesh.setTransfiniteCurve(line, p.Ny_outer,
                                    meshType="Progression",
                                    coef=p.y_outer_vertical_progression)
line = gmsh.model.geo.addLine(lower_top_surf_pts[3],outer_bot_1)
gmsh.model.geo.mesh.setTransfiniteCurve(line, p.Ny_outer,
                                    meshType="Progression",
                                    coef=p.y_outer_vertical_progression)
bottom_top_surf = create_surface_from_points([outer_bot_0,outer_bot_1,outer_bot_2,outer_bot_3])
inlet_circ_bot_face = create_surface_from_points([outer_bot_0,outer_bot_3,lower_top_surf_pts[0],lower_top_surf_pts[1]],surface_type="filling")
_ = create_surface_from_points([outer_bot_3,outer_bot_2,lower_top_surf_pts[2],lower_top_surf_pts[0]])
_ = create_surface_from_points([outer_bot_2,outer_bot_1,lower_top_surf_pts[3],lower_top_surf_pts[2]],surface_type="filling")
_ = create_surface_from_points([outer_bot_1,outer_bot_0,lower_top_surf_pts[1],lower_top_surf_pts[3]])

_ = create_volumes_from_surface_pairs([bottom_top_surf],[lower_top_surf])
gmsh.model.geo.synchronize()

# -----------------------------------------------------------------------
# Outer domain C Inlet
# -----------------------------------------------------------------------

inlet_C_upper_tags = [outer_top_0,outer_top_3]
inlet_C_lower_tags = [outer_bot_0,outer_bot_3]
LE_tag_left = airfoil_split_tags[2]
LE_tag_right = gmsh.model.geo.addPoint(0,0,p.W)
inlet_c_left = gmsh.model.geo.addCircleArc(inlet_C_upper_tags[0],LE_tag_left,inlet_C_lower_tags[0])
inlet_c_right = gmsh.model.geo.addCircleArc(inlet_C_upper_tags[1],LE_tag_right,inlet_C_lower_tags[1])

gmsh.model.geo.synchronize()
gmsh.model.geo.mesh.setTransfiniteCurve(inlet_c_left,p.nose_surface,"Bump",p.nose_bump_coeff)
gmsh.model.geo.mesh.setTransfiniteCurve(inlet_c_right,p.nose_surface,"Bump",p.nose_bump_coeff)
inlet_c_surf = create_surface_from_points([inlet_C_upper_tags[0],inlet_C_lower_tags[0],inlet_C_lower_tags[1],inlet_C_upper_tags[1]],surface_type="filling")
inlet_c_left_surf = create_surface_from_points([inlet_C_upper_tags[0],inlet_C_lower_tags[0],pts_tags_10c_lower[3],pts_tags_10c_upper[3]])
inlet_c_right_surf = create_surface_from_points([inlet_C_upper_tags[1],inlet_C_lower_tags[1],pts_tags_10c_lower[2],pts_tags_10c_upper[2]])

far_inlet_left = gmsh.model.geo.addPoint(-p.Ly,0,0)
far_inlet_right = gmsh.model.geo.addPoint(-p.Ly,0,p.W)
gmsh.model.geo.synchronize()
gmsh.model.mesh.embed(0,[far_inlet_left],1,inlet_c_left) 
gmsh.model.mesh.embed(0,[far_inlet_right],1,inlet_c_right) 
gmsh.model.geo.synchronize()
volume_tags_C_inlet = create_volumes_from_surface_pairs([inlet_circ_top_face],[inlet_circ_bot_face])
gmsh.model.geo.synchronize()
gmsh.model.geo.removeAllDuplicates()

# -----------------------------------------------------------------------
# Top near wake
# -----------------------------------------------------------------------

outer_top_near_wake_0 = outer_top_1
outer_top_near_wake_1 = gmsh.model.geo.addPoint(p.c + (p.Lx-p.c)*p.near_wake_length + p.near_wake_offset_in_x_1 + p.near_wake_offset_in_x_2 ,p.Ly,0)
outer_top_near_wake_2 = gmsh.model.geo.addPoint(p.c + (p.Lx-p.c)*p.near_wake_length + p.near_wake_offset_in_x_1 + p.near_wake_offset_in_x_2,p.Ly,p.W)
outer_top_near_wake_3 = outer_top_2
L1 = gmsh.model.geo.addLine(outer_top_near_wake_0, outer_top_near_wake_1)
gmsh.model.geo.mesh.setTransfiniteCurve(L1,nPoints=p.Nx_near_wake)
L2 = gmsh.model.geo.addLine(outer_top_near_wake_1, outer_top_near_wake_2)
gmsh.model.geo.mesh.setTransfiniteCurve(L2,nPoints=p.nz_mid)
L3 = gmsh.model.geo.addLine(outer_top_near_wake_2, outer_top_near_wake_3)
gmsh.model.geo.mesh.setTransfiniteCurve(L3,nPoints=p.Nx_near_wake)
L4 = gmsh.model.geo.addLine(outer_top_near_wake_3, outer_top_near_wake_0)
gmsh.model.geo.mesh.setTransfiniteCurve(L4,nPoints=p.nz_mid)
gmsh.model.geo.synchronize()

upper_top_surf_pts = gmsh.model.getBoundary([(2, near_wake_top_surf)], oriented=False, recursive=True)
upper_top_surf_pts = [tag for dim, tag in upper_top_surf_pts]


line = gmsh.model.geo.addLine(upper_top_surf_pts[0],outer_top_near_wake_3 )
gmsh.model.geo.mesh.setTransfiniteCurve(line, p.Ny_outer,
                                    meshType="Progression",
                                    coef=p.y_outer_vertical_progression)
line = gmsh.model.geo.addLine(upper_top_surf_pts[1],outer_top_near_wake_0)
gmsh.model.geo.mesh.setTransfiniteCurve(line, p.Ny_outer,
                                    meshType="Progression",
                                    coef=p.y_outer_vertical_progression)
line = gmsh.model.geo.addLine(upper_top_surf_pts[2],outer_top_near_wake_2)
gmsh.model.geo.mesh.setTransfiniteCurve(line, p.Ny_outer,
                                    meshType="Progression",
                                    coef=p.y_outer_vertical_progression)
line = gmsh.model.geo.addLine(upper_top_surf_pts[3],outer_top_near_wake_1)
gmsh.model.geo.mesh.setTransfiniteCurve(line, p.Ny_outer,
                                    meshType="Progression",
                                    coef=p.y_outer_vertical_progression)
outer_top_surf = create_surface_from_points([outer_top_near_wake_0,outer_top_near_wake_1,outer_top_near_wake_2,outer_top_near_wake_3])
# _ = create_surface_from_points([outer_top_near_wake_0,outer_top_near_wake_3,upper_top_surf_pts[0],upper_top_surf_pts[1]])
_ = create_surface_from_points([outer_top_near_wake_3,outer_top_near_wake_2,upper_top_surf_pts[2],upper_top_surf_pts[0]])
_ = create_surface_from_points([outer_top_near_wake_2,outer_top_near_wake_1,upper_top_surf_pts[3],upper_top_surf_pts[2]], surface_type="filling")
_ = create_surface_from_points([outer_top_near_wake_1,outer_top_near_wake_0,upper_top_surf_pts[1],upper_top_surf_pts[3]])

_ = create_volumes_from_surface_pairs([outer_top_surf],[near_wake_top_surf])
gmsh.model.geo.synchronize()

# -----------------------------------------------------------------------
# Bottom near wake
# -----------------------------------------------------------------------

outer_bot_near_wake_0 = outer_bot_1
outer_bot_near_wake_1 = gmsh.model.geo.addPoint(p.c + (p.Lx-p.c)*p.near_wake_length + p.near_wake_offset_in_x_1 + p.near_wake_offset_in_x_2,-p.Ly,0)
outer_bot_near_wake_2 = gmsh.model.geo.addPoint(p.c + (p.Lx-p.c)*p.near_wake_length + p.near_wake_offset_in_x_1 + p.near_wake_offset_in_x_2,-p.Ly,p.W)
outer_bot_near_wake_3 = outer_bot_2
L1 = gmsh.model.geo.addLine(outer_bot_near_wake_0, outer_bot_near_wake_1)
gmsh.model.geo.mesh.setTransfiniteCurve(L1,nPoints=p.Nx_near_wake)
L2 = gmsh.model.geo.addLine(outer_bot_near_wake_1, outer_bot_near_wake_2)
gmsh.model.geo.mesh.setTransfiniteCurve(L2,nPoints=p.nz_mid)
L3 = gmsh.model.geo.addLine(outer_bot_near_wake_2, outer_bot_near_wake_3)
gmsh.model.geo.mesh.setTransfiniteCurve(L3,nPoints=p.Nx_near_wake)
L4 = gmsh.model.geo.addLine(outer_bot_near_wake_3, outer_bot_near_wake_0)
gmsh.model.geo.mesh.setTransfiniteCurve(L4,nPoints=p.nz_mid)
gmsh.model.geo.synchronize()

upper_top_surf_pts = gmsh.model.getBoundary([(2, near_wake_bottom_surf)], oriented=False, recursive=True)
upper_top_surf_pts = [tag for dim, tag in upper_top_surf_pts]


line = gmsh.model.geo.addLine(upper_top_surf_pts[0],outer_bot_near_wake_3 )
gmsh.model.geo.mesh.setTransfiniteCurve(line, p.Ny_outer,
                                    meshType="Progression",
                                    coef=p.y_outer_vertical_progression)
line = gmsh.model.geo.addLine(upper_top_surf_pts[1],outer_bot_near_wake_0)
gmsh.model.geo.mesh.setTransfiniteCurve(line, p.Ny_outer,
                                    meshType="Progression",
                                    coef=p.y_outer_vertical_progression)
line = gmsh.model.geo.addLine(upper_top_surf_pts[2],outer_bot_near_wake_2)
gmsh.model.geo.mesh.setTransfiniteCurve(line, p.Ny_outer,
                                    meshType="Progression",
                                    coef=p.y_outer_vertical_progression)
line = gmsh.model.geo.addLine(upper_top_surf_pts[3],outer_bot_near_wake_1)
gmsh.model.geo.mesh.setTransfiniteCurve(line, p.Ny_outer,
                                    meshType="Progression",
                                    coef=p.y_outer_vertical_progression)
outer_top_surf = create_surface_from_points([outer_bot_near_wake_0,outer_bot_near_wake_1,outer_bot_near_wake_2,outer_bot_near_wake_3])
# _ = create_surface_from_points([outer_bot_near_wake_0,outer_bot_near_wake_3,upper_top_surf_pts[0],upper_top_surf_pts[1]])
_ = create_surface_from_points([outer_bot_near_wake_3,outer_bot_near_wake_2,upper_top_surf_pts[2],upper_top_surf_pts[0]])
_ = create_surface_from_points([outer_bot_near_wake_2,outer_bot_near_wake_1,upper_top_surf_pts[3],upper_top_surf_pts[2]], surface_type="filling")
_ = create_surface_from_points([outer_bot_near_wake_1,outer_bot_near_wake_0,upper_top_surf_pts[1],upper_top_surf_pts[3]])

_ = create_volumes_from_surface_pairs([outer_top_surf],[near_wake_bottom_surf])
gmsh.model.geo.synchronize()

# -----------------------------------------------------------------------
# Top far wake
# -----------------------------------------------------------------------

outer_top_far_wake_0 = outer_top_near_wake_1
outer_top_far_wake_1 = gmsh.model.geo.addPoint(p.Lx,p.Ly,0)
outer_top_far_wake_2 = gmsh.model.geo.addPoint(p.Lx,p.Ly,p.W)
outer_top_far_wake_3 = outer_top_near_wake_2
L1 = gmsh.model.geo.addLine(outer_top_far_wake_0, outer_top_far_wake_1)
gmsh.model.geo.mesh.setTransfiniteCurve(L1,nPoints=p.Nx_far_wake,coef=p.y_outer_horizontal_progression)
L2 = gmsh.model.geo.addLine(outer_top_far_wake_1, outer_top_far_wake_2)
gmsh.model.geo.mesh.setTransfiniteCurve(L2,nPoints=p.nz_mid)
L3 = gmsh.model.geo.addLine(outer_top_far_wake_2, outer_top_far_wake_3)
gmsh.model.geo.mesh.setTransfiniteCurve(L3,nPoints=p.Nx_far_wake, coef=1/p.y_outer_horizontal_progression)
L4 = gmsh.model.geo.addLine(outer_top_far_wake_3, outer_top_far_wake_0)
gmsh.model.geo.mesh.setTransfiniteCurve(L4,nPoints=p.nz_mid)
gmsh.model.geo.synchronize()

upper_top_surf_pts = gmsh.model.getBoundary([(2, far_wake_top_surf)], oriented=False, recursive=True)
upper_top_surf_pts = [tag for dim, tag in upper_top_surf_pts]


line = gmsh.model.geo.addLine(upper_top_surf_pts[0],outer_top_far_wake_3 )
gmsh.model.geo.mesh.setTransfiniteCurve(line, p.Ny_outer,
                                    meshType="Progression",
                                    coef=p.y_outer_vertical_progression)
line = gmsh.model.geo.addLine(upper_top_surf_pts[1],outer_top_far_wake_0)
gmsh.model.geo.mesh.setTransfiniteCurve(line, p.Ny_outer,
                                    meshType="Progression",
                                    coef=p.y_outer_vertical_progression)
line = gmsh.model.geo.addLine(upper_top_surf_pts[2],outer_top_far_wake_2)
gmsh.model.geo.mesh.setTransfiniteCurve(line, p.Ny_outer,
                                    meshType="Progression",
                                    coef=p.y_outer_vertical_progression)
line = gmsh.model.geo.addLine(upper_top_surf_pts[3],outer_top_far_wake_1)
gmsh.model.geo.mesh.setTransfiniteCurve(line, p.Ny_outer,
                                    meshType="Progression",
                                    coef=p.y_outer_vertical_progression)
outer_top_surf = create_surface_from_points([outer_top_far_wake_0,outer_top_far_wake_1,outer_top_far_wake_2,outer_top_far_wake_3])
# _ = create_surface_from_points([outer_top_far_wake_0,outer_top_far_wake_3,upper_top_surf_pts[0],upper_top_surf_pts[1]])
_ = create_surface_from_points([outer_top_far_wake_3,outer_top_far_wake_2,upper_top_surf_pts[2],upper_top_surf_pts[0]])
_ = create_surface_from_points([outer_top_far_wake_2,outer_top_far_wake_1,upper_top_surf_pts[3],upper_top_surf_pts[2]])
_ = create_surface_from_points([outer_top_far_wake_1,outer_top_far_wake_0,upper_top_surf_pts[1],upper_top_surf_pts[3]])

_ = create_volumes_from_surface_pairs([outer_top_surf],[far_wake_top_surf])
gmsh.model.geo.synchronize()

# -----------------------------------------------------------------------
# Top far wake
# -----------------------------------------------------------------------

outer_bot_far_wake_0 = outer_bot_near_wake_1
outer_bot_far_wake_1 = gmsh.model.geo.addPoint(p.Lx,-p.Ly,0)
outer_bot_far_wake_2 = gmsh.model.geo.addPoint(p.Lx,-p.Ly,p.W)
outer_bot_far_wake_3 = outer_bot_near_wake_2
L1 = gmsh.model.geo.addLine(outer_bot_far_wake_0, outer_bot_far_wake_1)
gmsh.model.geo.mesh.setTransfiniteCurve(L1,nPoints=p.Nx_far_wake,coef=p.y_outer_horizontal_progression)
L2 = gmsh.model.geo.addLine(outer_bot_far_wake_1, outer_bot_far_wake_2)
gmsh.model.geo.mesh.setTransfiniteCurve(L2,nPoints=p.nz_mid)
L3 = gmsh.model.geo.addLine(outer_bot_far_wake_2, outer_bot_far_wake_3)
gmsh.model.geo.mesh.setTransfiniteCurve(L3,nPoints=p.Nx_far_wake, coef=1/p.y_outer_horizontal_progression)
L4 = gmsh.model.geo.addLine(outer_bot_far_wake_3, outer_bot_far_wake_0)
gmsh.model.geo.mesh.setTransfiniteCurve(L4,nPoints=p.nz_mid)
gmsh.model.geo.synchronize()

upper_top_surf_pts = gmsh.model.getBoundary([(2, far_wake_bottom_surf)], oriented=False, recursive=True)
upper_top_surf_pts = [tag for dim, tag in upper_top_surf_pts]


line = gmsh.model.geo.addLine(upper_top_surf_pts[0],outer_bot_far_wake_3 )
gmsh.model.geo.mesh.setTransfiniteCurve(line, p.Ny_outer,
                                    meshType="Progression",
                                    coef=p.y_outer_vertical_progression)
line = gmsh.model.geo.addLine(upper_top_surf_pts[1],outer_bot_far_wake_0)
gmsh.model.geo.mesh.setTransfiniteCurve(line, p.Ny_outer,
                                    meshType="Progression",
                                    coef=p.y_outer_vertical_progression)
line = gmsh.model.geo.addLine(upper_top_surf_pts[2],outer_bot_far_wake_2)
gmsh.model.geo.mesh.setTransfiniteCurve(line, p.Ny_outer,
                                    meshType="Progression",
                                    coef=p.y_outer_vertical_progression)
line = gmsh.model.geo.addLine(upper_top_surf_pts[3],outer_bot_far_wake_1)
gmsh.model.geo.mesh.setTransfiniteCurve(line, p.Ny_outer,
                                    meshType="Progression",
                                    coef=p.y_outer_vertical_progression)
outer_top_surf = create_surface_from_points([outer_bot_far_wake_0,outer_bot_far_wake_1,outer_bot_far_wake_2,outer_bot_far_wake_3])
# _ = create_surface_from_points([outer_bot_far_wake_0,outer_bot_far_wake_3,upper_top_surf_pts[0],upper_top_surf_pts[1]])
_ = create_surface_from_points([outer_bot_far_wake_3,outer_bot_far_wake_2,upper_top_surf_pts[2],upper_top_surf_pts[0]])
_ = create_surface_from_points([outer_bot_far_wake_2,outer_bot_far_wake_1,upper_top_surf_pts[3],upper_top_surf_pts[2]])
_ = create_surface_from_points([outer_bot_far_wake_1,outer_bot_far_wake_0,upper_top_surf_pts[1],upper_top_surf_pts[3]])

_ = create_volumes_from_surface_pairs([outer_top_surf],[far_wake_bottom_surf])
gmsh.model.geo.synchronize()

# -----------------------------------------------------------------------
# Add Physical Groups
# -----------------------------------------------------------------------

assign_surfaces_to_physical_groups(p, [inlet_c_surf])

volumes = gmsh.model.getEntities(dim=3)
fluid_tags = [tag for (dim, tag) in volumes]

gmsh.model.addPhysicalGroup(3, fluid_tags, name="fluid")

# -----------------------------------------------------------------------
# Set Periodic surfaces
# -----------------------------------------------------------------------

# side1_dimtags = gmsh.model.getEntitiesForPhysicalName("Side1")
# side2_dimtags = gmsh.model.getEntitiesForPhysicalName("Side2")
# translation = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, p.W, 0, 0, 0, 1]
# gmsh.model.mesh.setPeriodic(2,[tag for dim,tag in side2_dimtags],[tag for dim,tag in side1_dimtags ],translation)

# -----------------------------------------------------------------------
# Extrude airfoil normal
# -----------------------------------------------------------------------

# extrude = gmsh.model.geo.extrudeBoundaryLayer([nose_airfoil_extrude_1[1]],[5],[0.01],recombine=True,second=True)
# gmsh.model.geo.synchronize()
# print(extrude)
# extrude = gmsh.model.geo.extrudeBoundaryLayer([upper_airfoil_extrude_1[1]],[5],[0.01],recombine=True,second=True)
# gmsh.model.geo.synchronize()
# extrude = gmsh.model.geo.extrudeBoundaryLayer([lower_airfoil_extrude_1[1]],[5],[0.01],recombine=True,second=True)
# gmsh.model.geo.synchronize()

# -----------------------------------------------------------------------
# Generate Mesh
# -----------------------------------------------------------------------

gmsh.model.geo.synchronize()
gmsh.model.geo.removeAllDuplicates()
# Generate 3D mesh
gmsh.option.setNumber("Mesh.ElementOrder", 2)      # Quadratic elements
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)  
# gmsh.option.setNumber("Mesh.CgnsExportStructured",1)

gmsh.model.mesh.generate(3)
gmsh.model.mesh.optimize('HighOrderFastCurving')

# Launch GUI (optional - comment out if running in batch mode)
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.write(os.path.join(script_dir, "nested_z_3D_blocks_geo.msh"))
gmsh.model.mesh.setOrder(1)
gmsh.write(os.path.join(script_dir, "nested_z_3D_blocks_geo.cgns"))
gmsh.finalize()