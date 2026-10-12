"""PNG export smoke checks across all mesh modes."""

import importlib.util
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ("gmsh", "matplotlib")),
                     "Gmsh and Matplotlib are required")
class PreviewTests(unittest.TestCase):
    def test_real_mesh_previews_are_readable_pngs(self):
        import gmsh
        import matplotlib.image as mpimg
        from generate_mesh import generate_mesh
        from naca_mesh import Params

        old_initialize = gmsh.initialize

        def quiet():
            old_initialize()
            gmsh.option.setNumber("General.Terminal", 0)

        params = Params(ny=6, nz_mid=3, nz_left_right=4, upper_surface=31,
                        lower_surface=16, nose_surface=21, Ny_outer=6,
                        Nx_far_wake=8, Nx_near_wake=8, z_elements=4)
        with tempfile.TemporaryDirectory() as out:
            for mode in ("2d", "3d-extrude", "3d-nested"):
                with self.subTest(mode=mode), patch.object(gmsh, "initialize", quiet):
                    paths = generate_mesh(params, mode=mode, output_dir=out, order=2, image_dpi=80)
                    pngs = [p for name, p in paths.items() if name.startswith("png_")]
                    self.assertEqual(len(pngs), 2 if mode == "2d" else 4)
                    for path in pngs:
                        self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                        pixels = mpimg.imread(path)
                        self.assertGreater(pixels.shape[0], 200)
                        self.assertGreater(pixels.shape[1], 200)
                        self.assertGreater(float(np.std(pixels[:, :, :3])), 0.03)
                    self.assertFalse(gmsh.isInitialized())
