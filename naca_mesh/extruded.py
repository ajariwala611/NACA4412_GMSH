"""Uniform-Z extrusion of the shared planar XY block grid."""

import gmsh


def _boundary(entities, combined=False):
    return {abs(int(tag)) for _, tag in gmsh.model.getBoundary(
        entities, combined=combined, oriented=True)}


def extrude_planar_xy(surfaces, groups, span, z_elements):
    """Extrude all planar blocks together into conforming uniform-Z hex layers."""
    gmsh.model.removePhysicalGroups()
    result = gmsh.model.geo.extrude([(2, tag) for tag in surfaces], 0, 0, span,
                                    numElements=[z_elements], recombine=True)
    gmsh.model.geo.synchronize()
    top = set()
    sides = {name: set() for name in groups}
    tolerance = max(1, span) * 1e-7
    for dim, tag in set(result):
        if dim != 2:
            continue
        bbox = gmsh.model.getBoundingBox(2, tag)
        if abs(bbox[2] - span) < tolerance and abs(bbox[5] - span) < tolerance:
            top.add(tag)
            continue
        edges = _boundary([(2, tag)])
        for name, curves in groups.items():
            if edges.intersection(curves):
                sides[name].add(tag)
    for name, tags in sides.items():
        if not tags:
            raise ValueError(f"extruded boundary {name} is empty")
        gmsh.model.addPhysicalGroup(2, sorted(tags), name=name)
    gmsh.model.addPhysicalGroup(2, surfaces, name="Side1")
    gmsh.model.addPhysicalGroup(2, sorted(top), name="Side2")
    volumes = [tag for _, tag in gmsh.model.getEntities(3)]
    gmsh.model.addPhysicalGroup(3, volumes, name="fluid")
