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
    ny: int = 25            # verticals AND diagonals
    nz_mid: int = 5         # inner bottom/top AND outer top
    nz_left_right: int = 9 # outer left/right horizontals
    Ny: int = 10
    Nx: int = 15
    Nwake: int = 20

    # Grading
    y_progression: float = 1.15
    y_outer_vertical_progression: float = 1.5
    y_outer_horizontal_progression: float = 1.5
    y_wake_length: float = 0.015

    upper_surface: int = 250
    lower_surface: int = 50
    nose_surface: int = 40
    c: float = 1.0
    Nu: int = 200
    Nl: int = 200
    # ---------- Helpers (embedded) ----------
    def __init__(self):
        """Initialize and compute uniform Z segments"""
        self.w_in, self.z1, self.z2, self.dz = self.uniform_z_segments()
        self.outer_veritcal_ele, self.outer_vertical_heights = self.create_geometric_grading(self.Ny,self.y_outer_vertical_progression,None,1)
        self.outer_horizontal_ele, self.outer_horizontal_heigths = self.create_geometric_grading(self.Nx,self.y_outer_horizontal_progression,0.015,self.Nwake)
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
    B  = gmsh.model.geo.addPoint(x0, y0,    p.W)
    C  = gmsh.model.geo.addPoint(x0, H_eff, p.W)
    D  = gmsh.model.geo.addPoint(x0, H_eff, 0.0)
    E  = gmsh.model.geo.addPoint(x0, y0,    z1)
    F  = gmsh.model.geo.addPoint(x0, y0,    z2)
    G  = gmsh.model.geo.addPoint(x0, h_eff, z2)
    H = gmsh.model.geo.addPoint(x0, h_eff, z1)
    gmsh.model.geo.synchronize()
    surfaces, curves = create_nested_block_surfaces_from_points(A, B, C, D, E, F, G, H, p, recombine_angle=90)
    pts = [A, B, C, D, E, F, G, H]
    gmsh.model.geo.synchronize()
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
    curve_upper_far = gmsh.model.geo.addSpline(upper_far_pts)
    curve_nose      = gmsh.model.geo.addSpline(nose_pts)
    curve_lower_far = gmsh.model.geo.addSpline(lower_far_pts)
    gmsh.model.geo.synchronize()
    gmsh.model.geo.mesh.setTransfiniteCurve(curve_upper_far,p.upper_surface)
    gmsh.model.geo.mesh.setTransfiniteCurve(curve_nose,p.nose_surface)
    gmsh.model.geo.mesh.setTransfiniteCurve(curve_lower_far,p.lower_surface)

    return curve_upper_far, curve_nose, curve_lower_far, [pt_TE, upper_far_pts[-1], pt_LE, nose_pts[-1]]

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


def create_volumes_from_surface_pairs(surface_tags_upper, surface_tags_lower, verbose=True):
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

gmsh.initialize()
gmsh.model.add("nested_at_x")
script_dir = os.path.dirname(os.path.abspath(__file__))
p = Params()
# --- Trailing and Leading Edge Points ---
xU_cos = cosine_spacing(p.Nu + 1, 1.0, 0.0)
xL_cos = cosine_spacing(p.Nl + 1, 0.0, 1.0)
xu, yu, _, _ = naca4412(xU_cos)
_, _, xl, yl = naca4412(xL_cos)

left_airfoil_curve_upper, left_airfoil_curve_nose, left_airfoil_curve_lower, airfoil_split_tags = build_three_airfoil_splines(xu, yu, xl, yl, p, x_th=0.1, make_wire=False)

gmsh.model.geo.synchronize()

surface_tags_10c_upper, curve_tags_10c_upper, pts_tags_10c_upper = nested_mesh_yz_transfinite_at_x(p,airfoil_split_tags[1],orientation_down=0)
surface_tags_10c_lower, curve_tags_10c_lower, pts_tags_10c_lower = nested_mesh_yz_transfinite_at_x(p,airfoil_split_tags[3],orientation_down=1)
surface_tags_TE_upper,  curve_tags_TE_upper, pts_tags_TE_upper = nested_mesh_yz_transfinite_at_x(p,airfoil_split_tags[0],orientation_down=0)
surface_tags_TE_lower,  curve_tags_TE_lower, pts_tags_TE_lower = nested_mesh_yz_transfinite_at_x(p,airfoil_split_tags[0],orientation_down=1)


# -----------------------------------------------------------------------
# C nested
# -----------------------------------------------------------------------

airfoil_split_upper = _pt_xyz(pts_tags_10c_upper[3])
airfoil_split_lower = _pt_xyz(pts_tags_10c_lower[3])
x_arc_center = (airfoil_split_upper[0] + airfoil_split_lower[0]) / 2
y_arc_center = (airfoil_split_upper[1] + airfoil_split_lower[1]) / 2
z_arc_center = (airfoil_split_upper[2] + airfoil_split_lower[2]) / 2
arc_ceter_pt_tag_1 = gmsh.model.geo.addPoint(x_arc_center,0.0,0.0)
gmsh.model.geo.synchronize()
arc_tag_1 = gmsh.model.geo.addCircleArc(pts_tags_10c_upper[3], arc_ceter_pt_tag_1, pts_tags_10c_lower[3])
# gmsh.model.geo.remove([(0,arc_ceter_pt_tag)])
gmsh.model.geo.synchronize()
gmsh.model.geo.mesh.setTransfiniteCurve(arc_tag_1,p.nose_surface)
circ1_surf = create_surface_from_points([pts_tags_10c_upper[0],pts_tags_10c_lower[0],pts_tags_10c_lower[3],pts_tags_10c_upper[3]])

airfoil_split_upper = _pt_xyz(pts_tags_10c_upper[7])
airfoil_split_lower = _pt_xyz(pts_tags_10c_lower[7])
x_arc_center = (airfoil_split_upper[0] + airfoil_split_lower[0]) / 2
y_arc_center = (airfoil_split_upper[1] + airfoil_split_lower[1]) / 2
z_arc_center = (airfoil_split_upper[2] + airfoil_split_lower[2]) / 2
arc_ceter_pt_tag = gmsh.model.geo.addPoint(x_arc_center,0,p.z1)
gmsh.model.geo.synchronize()
arc_tag_2 = gmsh.model.geo.addCircleArc(pts_tags_10c_upper[7], arc_ceter_pt_tag, pts_tags_10c_lower[7])
gmsh.model.geo.remove([(0,arc_ceter_pt_tag)])
gmsh.model.geo.synchronize()
nose_airfoil_extrude_1 = gmsh.model.geo.extrude([(1,left_airfoil_curve_nose)],0.0,0.0,p.z1,numElements=[p.nz_left_right-1],recombine=True)
nose_airfoil_extrude_1_spline = nose_airfoil_extrude_1[0][1]
nose_airfoil_extrude_left_surface = nose_airfoil_extrude_1[1][1]
gmsh.model.geo.mesh.setTransfiniteSurface(nose_airfoil_extrude_left_surface)
gmsh.model.geo.mesh.setRecombine(2,nose_airfoil_extrude_left_surface,90)
gmsh.model.geo.synchronize()
gmsh.model.geo.mesh.setTransfiniteCurve(arc_tag_2,p.nose_surface)
circ2_surf = create_surface_from_points([pts_tags_10c_upper[7],pts_tags_10c_lower[7],pts_tags_10c_lower[4],pts_tags_10c_upper[4]])

airfoil_split_upper = _pt_xyz(pts_tags_10c_upper[6])
airfoil_split_lower = _pt_xyz(pts_tags_10c_lower[6])
x_arc_center = (airfoil_split_upper[0] + airfoil_split_lower[0]) / 2
y_arc_center = (airfoil_split_upper[1] + airfoil_split_lower[1]) / 2
z_arc_center = (airfoil_split_upper[2] + airfoil_split_lower[2]) / 2
arc_ceter_pt_tag = gmsh.model.geo.addPoint(x_arc_center,0,p.z2)
gmsh.model.geo.synchronize()
arc_tag_3 = gmsh.model.geo.addCircleArc(pts_tags_10c_upper[6], arc_ceter_pt_tag, pts_tags_10c_lower[6])
gmsh.model.geo.remove([(0,arc_ceter_pt_tag)])
gmsh.model.geo.synchronize()
nose_airfoil_extrude_2 = gmsh.model.geo.extrude([(1,nose_airfoil_extrude_1_spline)],0.0,0.0,p.z2-p.z1,numElements=[p.nz_mid-1],recombine=True)
nose_airfoil_extrude_2_spline = nose_airfoil_extrude_2[0][1]
nose_airfoil_extrude_mid_surface = nose_airfoil_extrude_2[1][1]
gmsh.model.geo.mesh.setTransfiniteSurface(nose_airfoil_extrude_mid_surface)
gmsh.model.geo.mesh.setRecombine(2,nose_airfoil_extrude_mid_surface,90)
gmsh.model.geo.synchronize()
gmsh.model.geo.mesh.setTransfiniteCurve(arc_tag_3,p.nose_surface)
circ3_surf = create_surface_from_points([pts_tags_10c_upper[6],pts_tags_10c_lower[6],pts_tags_10c_lower[5],pts_tags_10c_upper[5]])

airfoil_split_upper = _pt_xyz(pts_tags_10c_upper[2])
airfoil_split_lower = _pt_xyz(pts_tags_10c_lower[2])
x_arc_center = (airfoil_split_upper[0] + airfoil_split_lower[0]) / 2
y_arc_center = (airfoil_split_upper[1] + airfoil_split_lower[1]) / 2
z_arc_center = (airfoil_split_upper[2] + airfoil_split_lower[2]) / 2
arc_ceter_pt_tag_4 = gmsh.model.geo.addPoint(x_arc_center,0,p.W)
gmsh.model.geo.synchronize()
arc_tag_4 = gmsh.model.geo.addCircleArc(pts_tags_10c_upper[2], arc_ceter_pt_tag_4, pts_tags_10c_lower[2])
# gmsh.model.geo.remove([(0,arc_ceter_pt_tag_4)])
gmsh.model.geo.synchronize()
nose_airfoil_extrude_right = gmsh.model.geo.extrude([(1,nose_airfoil_extrude_2_spline)],0.0,0.0,p.W-p.z2,numElements=[p.nz_left_right-1],recombine=True)
nose_airfoil_extrude_right_spline = nose_airfoil_extrude_right[0][1]
nose_airfoil_extrude_right_surface = nose_airfoil_extrude_right[1][1]
gmsh.model.geo.mesh.setTransfiniteSurface(nose_airfoil_extrude_right_surface)
gmsh.model.geo.mesh.setRecombine(2,nose_airfoil_extrude_right_surface,90)
gmsh.model.geo.synchronize()
gmsh.model.geo.mesh.setTransfiniteCurve(arc_tag_4,p.nose_surface)
circ4_surf = create_surface_from_points([pts_tags_10c_upper[2],pts_tags_10c_lower[2],pts_tags_10c_lower[1],pts_tags_10c_upper[1]])

outer_C_surf = create_surface_from_points([pts_tags_10c_upper[3],pts_tags_10c_lower[3],pts_tags_10c_lower[2],pts_tags_10c_upper[2]],surface_type="filling")
inner_C_surf = create_surface_from_points([pts_tags_10c_upper[7],pts_tags_10c_lower[7],pts_tags_10c_lower[6],pts_tags_10c_upper[6]],surface_type="filling")     
left_angle_C_surf = create_surface_from_points([pts_tags_10c_upper[3],pts_tags_10c_lower[3],pts_tags_10c_lower[7],pts_tags_10c_upper[7]],surface_type="filling")
right_angle_C_surf = create_surface_from_points([pts_tags_10c_upper[2],pts_tags_10c_lower[2],pts_tags_10c_lower[6],pts_tags_10c_upper[6]],surface_type="filling")

volume_tags_C_nested = create_volumes_from_surface_pairs(surface_tags_10c_upper, surface_tags_10c_lower)

# -----------------------------------------------------------------------
# Top nested
# -----------------------------------------------------------------------

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
upper_surf_right = create_surface_from_points([pts_tags_10c_upper[2],pts_tags_TE_upper[2],pts_tags_TE_upper[1],pts_tags_10c_upper[1]])

upper_top_surf = create_surface_from_points([pts_tags_10c_upper[3],pts_tags_TE_upper[3],pts_tags_TE_upper[2],pts_tags_10c_upper[2]])
upper_inner_surf = create_surface_from_points([pts_tags_10c_upper[7],pts_tags_TE_upper[7],pts_tags_TE_upper[6],pts_tags_10c_upper[6]])
upper_left_angle_surf = create_surface_from_points([pts_tags_10c_upper[3],pts_tags_TE_upper[3],pts_tags_TE_upper[7],pts_tags_10c_upper[7]],surface_type="filling")
upper_right_angle_surf = create_surface_from_points([pts_tags_10c_upper[6],pts_tags_TE_upper[6],pts_tags_TE_upper[2],pts_tags_10c_upper[2]],surface_type="filling")

volume_tags_upper_nested = create_volumes_from_surface_pairs(surface_tags_10c_upper, surface_tags_TE_upper)

# -----------------------------------------------------------------------
# Bottom nested
# -----------------------------------------------------------------------

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
# Outer domain
# -----------------------------------------------------------------------

outer_top = gmsh.model.geo.extrude([(2,upper_top_surf)],0.0,p.Ly-p.H,0.0,numElements=p.outer_veritcal_ele,heights=p.outer_vertical_heights,recombine=True)
outer_bottom = gmsh.model.geo.extrude([(2,lower_top_surf)],0.0,-p.Ly+p.H,0.0,numElements=p.outer_veritcal_ele,heights=p.outer_vertical_heights,recombine=True)
gmsh.model.geo.synchronize()
gmsh.model.geo.extrude([outer_top[3]],p.Lx-p.c,0.0,0.0,numElements=p.outer_horizontal_ele,heights=p.outer_horizontal_heigths,recombine=True)
gmsh.model.geo.extrude([outer_bottom[3]],p.Lx-p.c,0.0,0.0,numElements=p.outer_horizontal_ele,heights=p.outer_horizontal_heigths,recombine=True)
gmsh.model.geo.extrude([(2, surf) for surf in surface_tags_TE_upper],p.Lx-p.c,0.0,0.0,numElements=p.outer_horizontal_ele,heights=p.outer_horizontal_heigths,recombine=True)
gmsh.model.geo.extrude([(2, surf) for surf in surface_tags_TE_lower],p.Lx-p.c,0.0,0.0,numElements=p.outer_horizontal_ele,heights=p.outer_horizontal_heigths,recombine=True)

all_surfaces = gmsh.model.getEntities(2)
for dim,tag in all_surfaces:
    gmsh.model.geo.mesh.setTransfiniteSurface(tag)
    gmsh.model.geo.mesh.setRecombine(2,tag,90)

# corners = get_surface_corners(outer_top[-1][1])
inlet_C_upper_tags = gmsh.model.getBoundary([outer_top[-1]],recursive=True)[-2:]
inlet_C_lower_tags = gmsh.model.getBoundary([outer_bottom[-1]],recursive=True)[-2:]

inlet_c_left = gmsh.model.geo.addCircleArc(inlet_C_upper_tags[0][1],arc_ceter_pt_tag_1,inlet_C_lower_tags[0][1])
inlet_c_right = gmsh.model.geo.addCircleArc(inlet_C_upper_tags[1][1],arc_ceter_pt_tag_4,inlet_C_lower_tags[1][1])
gmsh.model.geo.synchronize()
gmsh.model.geo.mesh.setTransfiniteCurve(inlet_c_left,p.nose_surface)
gmsh.model.geo.mesh.setTransfiniteCurve(inlet_c_right,p.nose_surface)
inlet_c_surf = create_surface_from_points([inlet_C_upper_tags[0][1],inlet_C_lower_tags[0][1],inlet_C_lower_tags[1][1],inlet_C_upper_tags[1][1]],surface_type="filling")
inlet_c_left_surf = create_surface_from_points([inlet_C_upper_tags[0][1],inlet_C_lower_tags[0][1],pts_tags_10c_lower[3],pts_tags_10c_upper[3]])
inlet_c_right_surf = create_surface_from_points([inlet_C_upper_tags[1][1],inlet_C_lower_tags[1][1],pts_tags_10c_lower[2],pts_tags_10c_upper[2]])

volume_tags_C_inlet = create_volumes_from_surface_pairs([outer_top[-1][1]],[outer_bottom[-1][1]])
gmsh.model.geo.synchronize()
gmsh.model.geo.removeAllDuplicates()

# -----------------------------------------------------------------------
#Add Physical Groups
# -----------------------------------------------------------------------

assign_surfaces_to_physical_groups(p, [inlet_c_surf])

volumes = gmsh.model.getEntities(dim=3)
fluid_tags = [tag for (dim, tag) in volumes]

gmsh.model.addPhysicalGroup(3, fluid_tags, name="fluid")

# -----------------------------------------------------------------------
# Generate Mesh
# -----------------------------------------------------------------------

gmsh.model.geo.synchronize()

# Generate 3D mesh
gmsh.option.setNumber("Mesh.ElementOrder", 2)      # Quadratic elements
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2) 

gmsh.model.mesh.generate(3)

gmsh.write(os.path.join(script_dir, "nested_z_3D_blocks_geo.cgns"))
gmsh.write(os.path.join(script_dir, "nested_z_3D_blocks_geo.msh"))

# Launch GUI (optional - comment out if running in batch mode)
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()