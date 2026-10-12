"""Headless PNG previews of the actual generated surface meshes."""

from pathlib import Path

import gmsh
import numpy as np


def write_previews(p, mode, directory, prefix, dpi=160):
    # Import plotting only when requested; Agg works without the Gmsh GUI.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tags, coordinates, _ = gmsh.model.mesh.getNodes()
    index = np.argsort(tags)
    tags = np.asarray(tags)[index]
    coordinates = np.asarray(coordinates).reshape(-1, 3)[index]

    def physical(name, dim=2):
        for d, tag in gmsh.model.getPhysicalGroups(dim):
            if gmsh.model.getPhysicalName(d, tag) == name:
                return list(gmsh.model.getEntitiesForPhysicalGroup(d, tag))
        return []

    def segments(entities, dim=2):
        pieces = []
        for entity in sorted(set(int(t) for t in entities)):
            types, _, node_arrays = gmsh.model.mesh.getElements(dim, entity)
            for kind, nodes in zip(types, node_arrays):
                _, _, order, width, _, corners = gmsh.model.mesh.getElementProperties(kind)
                if (dim == 2 and corners != 4) or (dim == 1 and corners != 2):
                    continue
                connectivity = np.asarray(nodes).reshape(-1, width)
                xyz = coordinates[np.searchsorted(tags, connectivity)]
                for edge in range(corners if dim == 2 else 1):
                    a, b = xyz[:, edge], xyz[:, (edge+1) % corners]
                    if order == 2:
                        mid = xyz[:, corners+edge]
                        t = np.linspace(-1, 1, 5)[None, :, None]
                        line = a[:, None]*t*(t-1)/2 + mid[:, None]*(1-t*t) + b[:, None]*t*(t+1)/2
                        pieces.append(np.stack((line[:, :-1], line[:, 1:]), axis=2).reshape(-1, 2, 3))
                    else:
                        pieces.append(np.stack((a, b), axis=1))
        if not pieces:
            raise ValueError("preview selection contains no mesh edges")
        return np.concatenate(pieces)

    def figure(size=(10, 7)):
        fig = Figure(figsize=size, facecolor="white")
        FigureCanvasAgg(fig)
        return fig

    paths = {}

    def save(fig, view):
        path = directory / f"{prefix}_{view}.png"
        fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
        fig.clear()
        paths[f"png_{view}"] = path

    planar = [t for _, t in gmsh.model.getEntities(2)] if mode == "2d" else physical("Side1")
    xy = segments(planar)
    wall = segments(physical("Airfoil", 1), dim=1) if mode == "2d" else None
    for view, bounds in (("xy_domain", (-p.Ly, p.Lx, -p.Ly, p.Ly)),
                         ("xy_airfoil", (-0.3, 1.6, -0.4, 0.4))):
        fig = figure()
        ax = fig.add_subplot()
        ax.add_collection(LineCollection(xy[:, :, :2], colors="#245c7c", linewidths=0.35))
        if wall is not None:
            ax.add_collection(LineCollection(wall[:, :, :2], colors="#d66b21", linewidths=1.2))
        ax.set(xlim=bounds[:2], ylim=bounds[2:], xlabel="X / chord", ylabel="Y / chord",
               title=f"NACA {p.airfoil} | {mode} | XY grid at Z = 0")
        ax.set_aspect("equal")
        fig.tight_layout()
        save(fig, view)

    if mode != "2d":
        boundary = set()
        for _, tag in gmsh.model.getPhysicalGroups(2):
            boundary.update(gmsh.model.getEntitiesForPhysicalGroup(2, tag))
        lines = segments(boundary)
        # Crop the physical mesh for an informative airfoil view; no Z scaling
        # is applied to the coordinates. The display aspect expands the span.
        middle = lines.mean(axis=1)
        lines = lines[(middle[:, 0] > -0.3) & (middle[:, 0] < 1.6) & (abs(middle[:, 1]) < 0.4)]
        fig = figure()
        ax = fig.add_subplot(projection="3d")
        ax.add_collection3d(Line3DCollection(lines, colors="#245c7c", linewidths=0.3, alpha=0.7))
        ax.add_collection3d(Line3DCollection(segments(physical("Airfoil")),
                                            colors="#d66b21", linewidths=0.45, alpha=0.9))
        ax.set(xlim=(-0.3, 1.6), ylim=(-0.4, 0.4), zlim=(0, p.W),
               xlabel="X / chord", ylabel="Y / chord",
               title=f"NACA {p.airfoil} | {mode} | airfoil perspective\nSpan visually expanded for readability")
        ax.set_box_aspect((2.4, 1.1, 0.8))
        ax.view_init(elev=25, azim=-65)
        fig.tight_layout()
        # Matplotlib's tight bounding box can omit a projected Z-axis label.
        fig.text(0.97, 0.48, "Z / chord", rotation=90, va="center")
        save(fig, "perspective")

        section = []
        for _, tag in gmsh.model.getEntities(2):
            bbox = gmsh.model.getBoundingBox(2, tag)
            if abs(bbox[0]-p.c) < 1e-6 and abs(bbox[3]-p.c) < 1e-6:
                section.append(tag)
        yz = segments(section)
        edges = {abs(int(t)) for _, t in gmsh.model.getBoundary(
            [(2, t) for t in section], combined=False, oriented=True)}
        outlines = segments(edges, dim=1)
        fig = figure((7, 8))
        ax = fig.add_subplot()
        ax.add_collection(LineCollection(yz[:, :, [2, 1]], colors="#245c7c", linewidths=0.5))
        ax.add_collection(LineCollection(outlines[:, :, [2, 1]], colors="#d66b21", linewidths=1))
        ax.set(xlim=(-0.05*p.W, 1.05*p.W), ylim=(-1.1*p.H, 1.1*p.H),
               xlabel="Z / chord", ylabel="Y / chord",
               title=f"NACA {p.airfoil} | {mode}\nYZ block section at X / chord = {p.c:g}")
        ax.set_aspect("equal")
        fig.tight_layout()
        save(fig, "yz_section")
    return paths
