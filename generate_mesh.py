"""Structured XY, extruded, and nested meshes around a NACA four-digit airfoil.

Run this file for command-line generation, or import generate_mesh. Shared
configuration and numerical geometry live in the naca_mesh package.
"""

import gmsh
import numpy as np
from pathlib import Path
from dataclasses import replace
from naca_mesh.config import Params
from naca_mesh.geometry import (
    nearest_station_from_upper, nearest_station_from_lower, _pt_xyz,
    _unit_surface_normal_at_index, build_three_airfoil_splines,
    build_scaled_nose_spline,
)


# --- helper: cosine-distributed spacing ---------------------------


def create_nested_block_surfaces_from_points(A, B, C, D, E, F, G, H, p: Params, orientation_down = 0, recombine_angle=90):
    # Infer vertical levels from point coordinates
    xA, yA, _ = _pt_xyz(A)
    xC, yC, _ = _pt_xyz(C)
    xB, yB, _ = _pt_xyz(B)
    xG, yG, _ = _pt_xyz(G)
    xF, yF, _ = _pt_xyz(F)
    bottom_y = np.abs(yA)
    top_y    = np.abs(yC)
    inner_y  = np.abs(yG)

    H_eff = np.sqrt((xB-xC)**2 + (yB-yC)**2)
    h_eff = np.sqrt((xG-xF)**2 + (yG-yF)**2)

    # downward = False
    # if H_eff < 0.0:
    #     # Normalize spans to positive and flip orientation
    #     bottom_y, top_y = top_y, bottom_y
    #     H_eff = -H_eff
    #     h_eff = -h_eff
    #     downward = True

    if not (H_eff > 0.0):
        raise ValueError("Invalid vertical span: top_y must be > bottom_y.")
    if not (bottom_y < inner_y < top_y):
        raise ValueError("Require bottom_y < inner_y < top_y.")

    # Grading (outer first-interval and inner matching) with normalized spans
    r_out, a_out_near_wake, r_in = p.grading(H_eff, h_eff)

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

def assign_surfaces_to_physical_groups(p, outer_C_surf_tag, airfoil_surfaces, eps=1e-6):
    """
    Assigns surfaces to physical groups.

    Parameters:
    -----------
    p : Params
        Parameters object
    outer_C_surf_tag : int
        The tag of the outer C-shell surface (from your geometry creation)
    airfoil_surfaces : list[int]
        The nine wall patches created by the nose/upper/lower construction
    eps : float
        Small epsilon for bounding box tolerance
    """
    gmsh.model.geo.synchronize()
    pg_inlet = gmsh.model.addPhysicalGroup(2, outer_C_surf_tag)
    gmsh.model.setPhysicalName(2, pg_inlet, "Inlet")
    groups = {
        "Outlet": (p.Lx-eps, -p.Ly-eps, -eps, p.Lx+eps, p.Ly+eps, p.W+eps),
        "Airfoil": None,
        "Bottom": (-p.Ly-eps, -p.Ly-eps, -eps, p.Lx+eps, -p.Ly+eps, p.W+eps),
        "Top": (-p.Ly-eps, p.Ly-eps, -eps, p.Lx+eps, p.Ly+eps, p.W+eps),
        "Side1": (-p.Ly-eps, -p.Ly-eps, -eps, p.Lx+eps, p.Ly+eps, eps),
        "Side2": (-p.Ly-eps, -p.Ly-eps, p.W-eps, p.Lx+eps, p.Ly+eps, p.W+eps),
    }

    # Assign bounding box groups
    for name, bbox in groups.items():
        # Use topology for the wall: thicker or more cambered airfoils can lie
        # outside the legacy +/-0.1 bounding box.
        surfs = [(2, tag) for tag in airfoil_surfaces] if name == "Airfoil" else gmsh.model.getEntitiesInBoundingBox(*bbox, 2)
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

def change_TE_near_wake_vert_progression(curve_tags,r_out,r_in,orientation=1, *, p):
    if orientation == 1:
        gmsh.model.geo.mesh.setTransfiniteCurve(curve_tags[3],p.ny,"Progression",r_out)
        gmsh.model.geo.mesh.setTransfiniteCurve(curve_tags[5],p.ny,"Progression",1/r_out)
        gmsh.model.geo.mesh.setTransfiniteCurve(curve_tags[6],p.ny,"Progression",r_in)
        gmsh.model.geo.mesh.setTransfiniteCurve(curve_tags[8],p.ny,"Progression",1/r_in)
    elif orientation == -1:
        gmsh.model.geo.mesh.setTransfiniteCurve(curve_tags[3],p.ny,"Progression",r_out)
        gmsh.model.geo.mesh.setTransfiniteCurve(curve_tags[5],p.ny,"Progression",1/r_out)
        gmsh.model.geo.mesh.setTransfiniteCurve(curve_tags[6],p.ny,"Progression",r_in)
        gmsh.model.geo.mesh.setTransfiniteCurve(curve_tags[8],p.ny,"Progression",1/r_in)

def write_mesh_quality_report(path, dim=3):
    """
    Write Gmsh mesh-quality stats for surface or volume elements.

    Reports:
      - minSJ: sampled minimum scaled Jacobian from Gmsh
      - AR: maxEdge / minEdge per element
    Note: Gmsh exposes "minSJ", not "maxSJ"; the reported maxSJ below is the
    maximum over element-wise minSJ values.
    """
    elem_types, elem_tags_by_type, _ = gmsh.model.mesh.getElements(dim)

    lines = []
    global_sj = []
    global_ar = []

    lines.append("Mesh quality report")
    lines.append("===================")
    lines.append("Scaled Jacobian metric: gmsh qualityName='minSJ'")
    lines.append("Aspect ratio metric: gmsh maxEdge / minEdge")
    lines.append("")

    for elem_type, elem_tags in zip(elem_types, elem_tags_by_type):
        if len(elem_tags) == 0:
            continue

        elem_name, _, order, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(elem_type)
        tags = np.asarray(elem_tags, dtype=np.uint64)

        min_sj = np.asarray(gmsh.model.mesh.getElementQualities(tags, "minSJ"), dtype=float)
        min_edge = np.asarray(gmsh.model.mesh.getElementQualities(tags, "minEdge"), dtype=float)
        max_edge = np.asarray(gmsh.model.mesh.getElementQualities(tags, "maxEdge"), dtype=float)

        ar = np.divide(
            max_edge,
            min_edge,
            out=np.full_like(max_edge, np.inf, dtype=float),
            where=min_edge > 0.0,
        )

        global_sj.append(min_sj)
        global_ar.append(ar)

        worst_sj_i = int(np.argmin(min_sj))
        worst_ar_i = int(np.argmax(ar))

        lines.append(f"Element type: {elem_name}  type={elem_type}  order={order}  nodes={num_nodes}")
        lines.append(f"  count                  : {len(tags)}")
        lines.append(f"  minSJ min/max/mean     : {min_sj.min():.8e}  {min_sj.max():.8e}  {min_sj.mean():.8e}")
        lines.append(f"  minSJ p01/p05/p50      : {np.percentile(min_sj, 1):.8e}  {np.percentile(min_sj, 5):.8e}  {np.percentile(min_sj, 50):.8e}")
        lines.append(f"  minSJ <= 0 count       : {int(np.count_nonzero(min_sj <= 0.0))}")
        lines.append(f"  minSJ < 0.1 count      : {int(np.count_nonzero(min_sj < 0.1))}")
        lines.append(f"  worst minSJ element    : tag={int(tags[worst_sj_i])}  minSJ={min_sj[worst_sj_i]:.8e}")
        lines.append(f"  AR min/max/mean        : {np.nanmin(ar):.8e}  {np.nanmax(ar):.8e}  {np.nanmean(ar):.8e}")
        lines.append(f"  AR p95/p99             : {np.nanpercentile(ar, 95):.8e}  {np.nanpercentile(ar, 99):.8e}")
        lines.append(f"  worst AR element       : tag={int(tags[worst_ar_i])}  AR={ar[worst_ar_i]:.8e}")
        lines.append("")

    if global_sj:
        sj_all = np.concatenate(global_sj)
        ar_all = np.concatenate(global_ar)

        lines.append("Global")
        lines.append("------")
        lines.append(f"  minSJ min/max/mean     : {sj_all.min():.8e}  {sj_all.max():.8e}  {sj_all.mean():.8e}")
        lines.append(f"  minSJ <= 0 count       : {int(np.count_nonzero(sj_all <= 0.0))}")
        lines.append(f"  AR min/max/mean        : {np.nanmin(ar_all):.8e}  {np.nanmax(ar_all):.8e}  {np.nanmean(ar_all):.8e}")
        lines.append(f"  AR p95/p99             : {np.nanpercentile(ar_all, 95):.8e}  {np.nanpercentile(ar_all, 99):.8e}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def build_nested_geometry(p):
    """Assemble the original nested 3D topology without generating a mesh."""
    gmsh.model.add(f"naca{p.airfoil}_nested")
    xu, yu, xl, yl = p.xu, p.yu, p.xl, p.yl

    left_airfoil_curve_upper, left_airfoil_curve_nose, left_airfoil_curve_lower, airfoil_split_tags, upper_pt_tags, nose_pt_tags, lower_pt_tags = build_three_airfoil_splines(xu, yu, xl, yl, p, make_wire=False)
    gmsh.model.geo.synchronize()


    surface_tags_10c_upper, curve_tags_10c_upper, pts_tags_10c_upper = nested_mesh_yz_transfinite_at_x(p,airfoil_split_tags[1],orientation_down=0)
    surface_tags_10c_lower, curve_tags_10c_lower, pts_tags_10c_lower = nested_mesh_yz_transfinite_at_x(p,airfoil_split_tags[3],orientation_down=1)


    airfoil_split_upper = _pt_xyz(pts_tags_10c_upper[3])
    airfoil_split_lower = _pt_xyz(pts_tags_10c_lower[3])
    x_arc_center = (airfoil_split_upper[0] + airfoil_split_lower[0]) / 2
    y_arc_center = (airfoil_split_upper[1] + airfoil_split_lower[1]) / 2
    z_arc_center = 0.0
    arc_ceter_pt_tag_1 = gmsh.model.geo.addPoint(x_arc_center,y_arc_center,z_arc_center)
    gmsh.model.geo.synchronize()

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

    upper_tag = pts_tags_10c_upper[7]
    lower_tag = pts_tags_10c_lower[7]
    arc_tag_2 = build_scaled_nose_spline(upper_tag,lower_tag,p.h_in,p.z1,p)
    gmsh.model.geo.mesh.setTransfiniteCurve(arc_tag_2,p.nose_surface,"Bump",p.nose_bump_coeff)
    gmsh.model.geo.remove([(0,arc_ceter_pt_tag)])
    gmsh.model.geo.synchronize()
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

    upper_tag = pts_tags_10c_upper[6]
    lower_tag = pts_tags_10c_lower[6]
    arc_tag_3 = build_scaled_nose_spline(upper_tag,lower_tag,p.h_in,p.z2,p)
    gmsh.model.geo.remove([(0,arc_ceter_pt_tag)])
    gmsh.model.geo.synchronize()
    gmsh.model.geo.mesh.setTransfiniteCurve(arc_tag_3,p.nose_surface,"Bump",p.nose_bump_coeff)
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

    upper_tag = pts_tags_10c_upper[2]
    lower_tag = pts_tags_10c_lower[2]
    arc_tag_4 = build_scaled_nose_spline(upper_tag,lower_tag,p.H,p.W,p)
    gmsh.model.geo.remove([(0,arc_ceter_pt_tag_4)])
    gmsh.model.geo.synchronize()
    gmsh.model.geo.mesh.setTransfiniteCurve(arc_tag_4,p.nose_surface,"Bump",p.nose_bump_coeff)
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

    change_TE_near_wake_vert_progression(curve_tags_TE_upper,p.y_progression_near_wake,p.r_near_wake_in,1, p=p)

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

    change_TE_near_wake_vert_progression(curve_tags_TE_lower,p.y_progression_near_wake,p.r_near_wake_in,-1, p=p)


    wake_pt_tag = gmsh.model.geo.addPoint(p.c + (p.Lx-p.c)*p.near_wake_length,0,0)
    gmsh.model.geo.synchronize()
    surface_tags_wake_upper,  curve_tags_wake_upper, pts_tags_wake_upper = nested_mesh_yz_transfinite_at_x(p,wake_pt_tag,orientation_down=0)
    _, near_wake_top_surf = create_volume_from_nested_pairs(surface_tags_TE_upper,surface_tags_wake_upper,pts_tags_TE_upper,pts_tags_wake_upper,0,p.Nx_near_wake)

    change_TE_near_wake_vert_progression(curve_tags_wake_upper,p.y_progression_near_wake,p.r_near_wake_in,1, p=p)

    surface_tags_wake_lower,  curve_tags_wake_lower, pts_tags_wake_lower = nested_mesh_yz_transfinite_at_x(p,wake_pt_tag,orientation_down=1)
    gmsh.model.geo.removeAllDuplicates()
    gmsh.model.geo.synchronize()

    copy_upper_to_lower(pts_tags_wake_upper, pts_tags_wake_lower)
    copy_upper_to_lower( pts_tags_TE_upper,   pts_tags_TE_lower)
    _ ,near_wake_bottom_surf = create_volume_from_nested_pairs(surface_tags_TE_lower,surface_tags_wake_lower,pts_tags_TE_lower,pts_tags_wake_lower,0,p.Nx_near_wake,oriented_down=True)

    change_TE_near_wake_vert_progression(curve_tags_wake_lower,p.y_progression_near_wake,p.r_near_wake_in,-1, p=p)

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
    _ = create_surface_from_points([outer_top_near_wake_3,outer_top_near_wake_2,upper_top_surf_pts[2],upper_top_surf_pts[0]])
    _ = create_surface_from_points([outer_top_near_wake_2,outer_top_near_wake_1,upper_top_surf_pts[3],upper_top_surf_pts[2]], surface_type="filling")
    _ = create_surface_from_points([outer_top_near_wake_1,outer_top_near_wake_0,upper_top_surf_pts[1],upper_top_surf_pts[3]])

    _ = create_volumes_from_surface_pairs([outer_top_surf],[near_wake_top_surf])
    gmsh.model.geo.synchronize()


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
    _ = create_surface_from_points([outer_bot_near_wake_3,outer_bot_near_wake_2,upper_top_surf_pts[2],upper_top_surf_pts[0]])
    _ = create_surface_from_points([outer_bot_near_wake_2,outer_bot_near_wake_1,upper_top_surf_pts[3],upper_top_surf_pts[2]], surface_type="filling")
    _ = create_surface_from_points([outer_bot_near_wake_1,outer_bot_near_wake_0,upper_top_surf_pts[1],upper_top_surf_pts[3]])

    _ = create_volumes_from_surface_pairs([outer_top_surf],[near_wake_bottom_surf])
    gmsh.model.geo.synchronize()


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
    _ = create_surface_from_points([outer_top_far_wake_3,outer_top_far_wake_2,upper_top_surf_pts[2],upper_top_surf_pts[0]])
    _ = create_surface_from_points([outer_top_far_wake_2,outer_top_far_wake_1,upper_top_surf_pts[3],upper_top_surf_pts[2]])
    _ = create_surface_from_points([outer_top_far_wake_1,outer_top_far_wake_0,upper_top_surf_pts[1],upper_top_surf_pts[3]])

    _ = create_volumes_from_surface_pairs([outer_top_surf],[far_wake_top_surf])
    gmsh.model.geo.synchronize()


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
    _ = create_surface_from_points([outer_bot_far_wake_3,outer_bot_far_wake_2,upper_top_surf_pts[2],upper_top_surf_pts[0]])
    _ = create_surface_from_points([outer_bot_far_wake_2,outer_bot_far_wake_1,upper_top_surf_pts[3],upper_top_surf_pts[2]])
    _ = create_surface_from_points([outer_bot_far_wake_1,outer_bot_far_wake_0,upper_top_surf_pts[1],upper_top_surf_pts[3]])

    _ = create_volumes_from_surface_pairs([outer_top_surf],[far_wake_bottom_surf])
    gmsh.model.geo.synchronize()


    airfoil_surfaces = [
        nose_airfoil_extrude_left_surface, nose_airfoil_extrude_mid_surface,
        nose_airfoil_extrude_right_surface,
        upper_airfoil_extrude_left_surface, upper_airfoil_extrude_mid_surface,
        upper_airfoil_extrude_right_surface,
        lower_airfoil_extrude_left_surface, lower_airfoil_extrude_mid_surface,
        lower_airfoil_extrude_right_surface,
    ]
    assign_surfaces_to_physical_groups(p, [inlet_c_surf], airfoil_surfaces)

    volumes = gmsh.model.getEntities(dim=3)
    fluid_tags = [tag for (dim, tag) in volumes]

    gmsh.model.addPhysicalGroup(3, fluid_tags, name="fluid")


    gmsh.model.geo.synchronize()
    gmsh.model.geo.removeAllDuplicates()


def generate_mesh(p=None, *, mode="3d-nested", output_dir=None, prefix=None, order=2, gui=False):
    """Generate an XY grid, uniform-Z extrusion, or nested spanwise mesh.

    This entry point owns the Gmsh session and always finalizes it. Geometry
    operations retain the original order. Importing this module creates no mesh.
    """
    # Recompute derived arrays even when callers have edited a Params instance.
    p = Params() if p is None else replace(p)
    p.validate()
    if mode not in ("2d", "3d-extrude", "3d-nested"):
        raise ValueError("mode must be 2d, 3d-extrude, or 3d-nested")
    prefix = prefix or f"naca{p.airfoil}_{mode}"
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    if not prefix or Path(prefix).name != prefix or prefix in (".", ".."):
        raise ValueError("prefix must be a filename without directory components")
    if gmsh.isInitialized():
        raise RuntimeError("generate_mesh requires its own uninitialized Gmsh session")
    output_dir = Path(__file__).resolve().parent if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        if mode == "3d-nested":
            build_nested_geometry(p)
        else:
            from naca_mesh.planar import build_planar_xy
            planar_surfaces, boundary_curves = build_planar_xy(p)
            if mode == "3d-extrude":
                from naca_mesh.extruded import extrude_planar_xy
                extrude_planar_xy(planar_surfaces, boundary_curves, p.W, p.z_elements)
        gmsh.option.setNumber("Mesh.ElementOrder", order)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

        dimension = 2 if mode == "2d" else 3
        gmsh.model.mesh.generate(dimension)

        quality_path = str(output_dir / f"{prefix}_quality_order{order}.txt")
        if mode == "3d-nested":
            write_mesh_quality_report(quality_path)
        else:
            write_mesh_quality_report(quality_path, dim=dimension)

        gmsh.write(str(output_dir / f"{prefix}.msh"))


        gmsh.write(str(output_dir / f"{prefix}.cgns"))


        if gui:
            gmsh.fltk.run()
        return {"msh": output_dir / f"{prefix}.msh",
                "cgns": output_dir / f"{prefix}.cgns",
                "quality": output_dir / f"{prefix}_quality_order{order}.txt"}
    finally:
        gmsh.finalize()


def generate_nested(p=None, *, output_dir=None, prefix="nested_z_3D_blocks_geo", order=2, gui=False):
    """Compatibility entry point preserving the original nested output names."""
    return generate_mesh(p, mode="3d-nested", output_dir=output_dir, prefix=prefix,
                         order=order, gui=gui)


if __name__ == "__main__":
    from naca_mesh.cli import main
    main(generate_mesh)
