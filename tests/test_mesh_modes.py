"""Verify full XY grids and uniform spanwise extrusion independently of nesting."""

from collections import Counter
import importlib.util
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("gmsh"), "Gmsh Python package is required")
class MeshModeTests(unittest.TestCase):
    def test_xy_and_extruded_modes(self):
        import gmsh
        import generate_mesh as generator
        from naca_mesh import Params

        original_initialize = gmsh.initialize
        original_report = generator.write_mesh_quality_report

        def quiet():
            original_initialize()
            gmsh.option.setNumber("General.Terminal", 0)

        with tempfile.TemporaryDirectory() as out:
            for code in ("4412", "0012"):
                for mode in ("2d", "3d-extrude"):
                    for order in (1, 2):
                        layers = 1 if order == 1 else 7
                        params = Params(airfoil=code, ny=6, nz_mid=3, nz_left_right=4,
                                        upper_surface=31, lower_surface=16, nose_surface=21,
                                        Ny_outer=6, Nx_far_wake=8, Nx_near_wake=8,
                                        z_elements=layers)

                        def check(path, dim=3):
                            types, tags, nodes = gmsh.model.mesh.getElements(dim)
                            expected_type = {("2d", 1): 3, ("2d", 2): 10,
                                             ("3d-extrude", 1): 5, ("3d-extrude", 2): 12}[mode, order]
                            self.assertEqual(list(types), [expected_type])
                            self.assertEqual(len(tags[0]), 930 if mode == "2d" else 930*layers)
                            self.assertTrue(np.all(np.asarray(gmsh.model.mesh.getElementQualities(tags[0], "minSJ")) > 0))
                            node_tags, xyz, _ = gmsh.model.mesh.getNodes()
                            xyz = np.asarray(xyz).reshape(-1, 3)
                            corners = np.asarray(nodes[0]).reshape(-1, gmsh.model.mesh.getElementProperties(types[0])[3])
                            corner_count = 4 if dim == 2 else 8
                            if mode == "2d":
                                self.assertEqual(gmsh.model.getEntities(3), [])
                                np.testing.assert_allclose(xyz[:, 2], 0, atol=1e-14)
                                self.assertEqual(len(gmsh.model.getEntities(2)), 14)
                                wall_tags = next(t for d, t in gmsh.model.getPhysicalGroups(1)
                                                if gmsh.model.getPhysicalName(d, t) == "Airfoil")
                                curve_counts = []
                                for curve in gmsh.model.getEntitiesForPhysicalGroup(1, wall_tags):
                                    curve_counts.append(sum(len(t) for t in gmsh.model.mesh.getElements(1, int(curve))[1]))
                                self.assertEqual(sorted(curve_counts), [15, 20, 30])
                            else:
                                corner_tags = set(corners[:, :8].ravel())
                                zs = xyz[[int(t) in corner_tags for t in node_tags], 2]
                                np.testing.assert_allclose(np.unique(np.round(zs, 12)),
                                                           np.linspace(0, params.W, layers+1), atol=1e-12)
                            indices = [(0, 1), (1, 2), (2, 3), (3, 0)] if dim == 2 else [
                                (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
                                (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
                            incidence = Counter(tuple(sorted(cell[list(face)]))
                                                for cell in corners[:, :corner_count] for face in indices)
                            self.assertTrue(all(v in (1, 2) for v in incidence.values()))
                            exterior = {key for key, n in incidence.items() if n == 1}
                            labeled = []
                            for _, tag in gmsh.model.getPhysicalGroups(dim-1):
                                for entity in gmsh.model.getEntitiesForPhysicalGroup(dim-1, tag):
                                    btypes, _, bnodes = gmsh.model.mesh.getElements(dim-1, int(entity))
                                    for bt, bn in zip(btypes, bnodes):
                                        width = gmsh.model.mesh.getElementProperties(bt)[3]
                                        bcorner = 2 if dim == 2 else 4
                                        labeled.extend(tuple(sorted(row[:bcorner])) for row in np.asarray(bn).reshape(-1, width))
                            self.assertEqual(len(labeled), len(set(labeled)))
                            self.assertEqual(set(labeled), exterior)
                            original_report(path, dim=dim)

                        with self.subTest(code=code, mode=mode, order=order), patch.object(
                                gmsh, "initialize", quiet), patch.object(generator, "write_mesh_quality_report", check), patch.object(
                                generator, "build_nested_geometry", side_effect=AssertionError("planar modes must not build nested volumes")):
                            generator.generate_mesh(params, mode=mode, order=order,
                                                    output_dir=out, prefix=f"{code}_{mode}_{order}", images=False)
                        self.assertFalse(gmsh.isInitialized())


if __name__ == "__main__":
    unittest.main()
