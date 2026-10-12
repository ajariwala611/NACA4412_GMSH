"""Small end-to-end meshes: element quality, conformity, and solver boundaries."""

from collections import Counter
import importlib.util
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("gmsh"), "Gmsh Python package is required")
class NestedMeshTests(unittest.TestCase):
    def test_import_has_no_mesh_side_effects(self):
        import gmsh
        import nested_z_3D_blocks_geo_normal
        self.assertFalse(gmsh.isInitialized())

    def test_coarse_airfoils_have_conforming_hexes_and_complete_boundaries(self):
        import gmsh
        import nested_z_3D_blocks_geo_normal as generator
        from naca_mesh import Params

        original_initialize = gmsh.initialize
        original_report = generator.write_mesh_quality_report

        def initialize_quietly():
            original_initialize()
            gmsh.option.setNumber("General.Terminal", 0)

        def check_mesh(path):
            types, tags, nodes = gmsh.model.mesh.getElements(3)
            self.assertEqual(list(types), [5])  # linear hexahedra only
            self.assertTrue(np.all(np.asarray(gmsh.model.mesh.getElementQualities(tags[0], "minSJ")) > 0))
            faces = Counter()
            corners = np.asarray(nodes[0]).reshape(-1, 8)
            indices = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
                       (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
            for cell in corners:
                for face in indices:
                    faces[tuple(sorted(cell[list(face)]))] += 1
            self.assertTrue(all(n in (1, 2) for n in faces.values()))
            exterior = {f for f, n in faces.items() if n == 1}
            boundary_faces = []
            names = set()
            for dim, tag in gmsh.model.getPhysicalGroups(2):
                name = gmsh.model.getPhysicalName(dim, tag)
                names.add(name)
                entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
                if name == "Airfoil":
                    self.assertEqual(len(entities), 9)
                for entity in entities:
                    stypes, _, snodes = gmsh.model.mesh.getElements(2, int(entity))
                    self.assertEqual(list(stypes), [3])  # quadrilateral boundary patches
                    boundary_faces.extend(tuple(sorted(face)) for face in np.asarray(snodes[0]).reshape(-1, 4))
            self.assertEqual(names, {"Airfoil", "Inlet", "Outlet", "Top", "Bottom", "Side1", "Side2"})
            self.assertEqual(len(boundary_faces), len(set(boundary_faces)))
            self.assertEqual(set(boundary_faces), exterior)
            original_report(path)

        with tempfile.TemporaryDirectory() as out:
            for code in ("4412", "0012", "4415"):
                with self.subTest(airfoil=code):
                    params = Params(airfoil=code, ny=6, nz_mid=3, nz_left_right=4,
                                    upper_surface=31, lower_surface=16, nose_surface=21,
                                    Ny_outer=6, Nx_far_wake=8, Nx_near_wake=8)
                    with patch.object(gmsh, "initialize", initialize_quietly), patch.object(
                            generator, "write_mesh_quality_report", check_mesh):
                        paths = generator.generate_nested(params, output_dir=out, prefix=code, order=1)
                    self.assertFalse(gmsh.isInitialized())
                    self.assertTrue(all(p.is_file() for p in paths.values()))


if __name__ == "__main__":
    unittest.main()
