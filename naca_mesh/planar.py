"""Direct construction of the full external-flow grid in the XY plane.

Fourteen named quadrilateral blocks reuse the original nested generator's XY
layout. No spanwise geometry or volume is created here.
"""

import gmsh

from .geometry import (build_three_airfoil_splines, build_scaled_nose_spline,
                       nearest_station_from_upper, nearest_station_from_lower,
                       _unit_surface_normal_at_index)


def build_planar_xy(p):
    geo = gmsh.model.geo
    gmsh.model.add(f"naca{p.airfoil}_xy")
    upper, nose, lower, split, *_ = build_three_airfoil_splines(
        p.xu, p.yu, p.xl, p.yl, p, make_wire=False)
    te, cut_u, le, cut_l = split
    points = {"TE": te, "A": cut_u, "B": cut_l, "LE": le}

    def point(name, x, y):
        points[name] = geo.addPoint(float(x), float(y), 0)

    # Use exactly the existing three-sample surface-normal calculation at the
    # split. The normal-offset nose spline is shared with the nested builder.
    iu, x_cut, _ = nearest_station_from_upper(p.xu, p.yu, p.x_th)
    il, _, _ = nearest_station_from_lower(p.xl, p.yl, x_cut)
    iu, il = max(1, min(len(p.xu)-2, iu)), max(1, min(len(p.xl)-2, il))
    nx, ny = _unit_surface_normal_at_index(p.xu[iu-1:iu+2], p.yu[iu-1:iu+2])
    distance = p.H - 0.05  # existing split-offset reduction
    point("U", p.xu[iu] + distance*nx[1], p.yu[iu] - distance*ny[1])
    nx, ny = _unit_surface_normal_at_index(p.xl[il-1:il+2], p.yl[il-1:il+2])
    point("L", p.xl[il] + distance*nx[1], p.yl[il] - distance*ny[1])
    point("TU", p.c, p.H)
    point("TL", p.c, -p.H)

    wake_x = p.c + (p.Lx-p.c)*p.near_wake_length
    outer_te_x = p.c + p.near_wake_offset_in_x_1
    outer_wake_x = wake_x + p.near_wake_offset_in_x_1 + p.near_wake_offset_in_x_2
    for name, x, y in (
        ("NW", wake_x, 0), ("NWT", wake_x, p.H), ("NWB", wake_x, -p.H),
        ("FW", p.Lx, 0), ("FWT", p.Lx, p.H+p.wake_stretch_in_Y),
        ("FWB", p.Lx, -p.H-p.wake_stretch_in_Y),
        ("OU", 0, p.Ly), ("OL", 0, -p.Ly),
        ("OTU", outer_te_x, p.Ly), ("OTL", outer_te_x, -p.Ly),
        ("ONU", outer_wake_x, p.Ly), ("ONL", outer_wake_x, -p.Ly),
        ("OFT", p.Lx, p.Ly), ("OFB", p.Lx, -p.Ly),
    ):
        point(name, x, y)
    geo.synchronize()

    curves = {}

    def remember(a, b, tag):
        curves[a, b], curves[b, a] = tag, -tag
        return tag

    remember("TE", "A", upper)
    remember("A", "B", nose)
    remember("B", "TE", lower)
    outer_nose = remember("U", "L", build_scaled_nose_spline(
        points["U"], points["L"], distance, 0, p))
    geo.mesh.setTransfiniteCurve(outer_nose, p.nose_surface, "Bump", p.nose_bump_coeff)
    inlet = remember("OU", "OL", geo.addCircleArc(points["OU"], points["LE"], points["OL"]))
    geo.mesh.setTransfiniteCurve(inlet, p.nose_surface, "Bump", p.nose_bump_coeff)

    def line(a, b, count, ratio=1):
        tag = remember(a, b, geo.addLine(points[a], points[b]))
        geo.mesh.setTransfiniteCurve(tag, count, "Progression", ratio)
        return tag

    line("A", "U", p.ny, p.y_progression)
    line("B", "L", p.ny, p.y_progression)
    line("TE", "TU", p.ny, p.y_progression_near_wake)
    line("TE", "TL", p.ny, p.y_progression_near_wake)
    line("U", "TU", p.upper_surface)
    line("L", "TL", p.lower_surface)
    for a, b in (("NW", "NWT"), ("NW", "NWB")):
        line(a, b, p.ny, p.y_progression_near_wake)
    for a, b in (("FW", "FWT"), ("FW", "FWB")):
        line(a, b, p.ny)
    for a, b in (("TE", "NW"), ("TU", "NWT"), ("TL", "NWB"),
                 ("OTU", "ONU"), ("OTL", "ONL")):
        line(a, b, p.Nx_near_wake)
    for a, b in (("NW", "FW"), ("NWT", "FWT"), ("NWB", "FWB"),
                 ("ONU", "OFT"), ("ONL", "OFB")):
        line(a, b, p.Nx_far_wake, p.y_outer_horizontal_progression)
    line("OU", "OTU", p.upper_surface)
    line("OL", "OTL", p.lower_surface)
    for a, b in (("U", "OU"), ("L", "OL"), ("TU", "OTU"), ("TL", "OTL"),
                 ("NWT", "ONU"), ("NWB", "ONL"), ("FWT", "OFT"), ("FWB", "OFB")):
        line(a, b, p.Ny_outer, p.y_outer_vertical_progression)

    blocks = {
        "nose_inner": ("A", "B", "L", "U"),
        "upper_inner": ("TE", "A", "U", "TU"),
        "lower_inner": ("B", "TE", "TL", "L"),
        "upper_near_wake": ("TE", "NW", "NWT", "TU"),
        "lower_near_wake": ("TE", "TL", "NWB", "NW"),
        "upper_far_wake": ("NW", "FW", "FWT", "NWT"),
        "lower_far_wake": ("NW", "NWB", "FWB", "FW"),
        "upper_outer": ("U", "TU", "OTU", "OU"),
        "lower_outer": ("L", "OL", "OTL", "TL"),
        "nose_outer": ("U", "OU", "OL", "L"),
        "upper_outer_near_wake": ("TU", "NWT", "ONU", "OTU"),
        "lower_outer_near_wake": ("TL", "OTL", "ONL", "NWB"),
        "upper_outer_far_wake": ("NWT", "FWT", "OFT", "ONU"),
        "lower_outer_far_wake": ("NWB", "ONL", "OFB", "FWB"),
    }
    surfaces = []
    for names in blocks.values():
        loop = geo.addCurveLoop([curves[a, b] for a, b in zip(names, names[1:]+names[:1])])
        surface = geo.addPlaneSurface([loop])
        geo.mesh.setTransfiniteSurface(surface, cornerTags=[points[n] for n in names])
        geo.mesh.setRecombine(2, surface)
        surfaces.append(surface)
    geo.synchronize()
    groups = {
        "Inlet": [inlet],
        "Outlet": [curves[a, b] for a, b in (("FW", "FWT"), ("FW", "FWB"),
                                             ("FWT", "OFT"), ("FWB", "OFB"))],
        "Airfoil": [upper, nose, lower],
        "Bottom": [curves[a, b] for a, b in (("OL", "OTL"), ("OTL", "ONL"), ("ONL", "OFB"))],
        "Top": [curves[a, b] for a, b in (("OU", "OTU"), ("OTU", "ONU"), ("ONU", "OFT"))],
    }
    for name, tags in groups.items():
        gmsh.model.addPhysicalGroup(1, tags, name=name)
    gmsh.model.addPhysicalGroup(2, surfaces, name="fluid")
    return surfaces, groups
