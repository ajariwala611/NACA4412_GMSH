"""Mode selection and user-facing element-count controls."""

import contextlib
import io
import unittest
from unittest.mock import patch

from naca_mesh.cli import main


class CliTests(unittest.TestCase):
    def capture(self, args):
        result = {}

        def generator(params, **kwargs):
            result.update(params=params, **kwargs)
            return {}

        with patch("sys.argv", ["generate_mesh.py", *args]):
            main(generator)
        return result

    def test_xy_surface_counts_are_elements(self):
        result = self.capture(["--mode", "2d", "--upper-elements", "30",
                               "--lower-elements", "15", "--nose-elements", "20",
                               "--wall-elements", "5", "--outer-elements", "5", "--no-gui"])
        p = result["params"]
        self.assertEqual((p.upper_surface, p.lower_surface, p.nose_surface, p.ny, p.Ny_outer),
                         (31, 16, 21, 6, 6))
        self.assertEqual(result["mode"], "2d")
        self.assertFalse(result["gui"])

    def test_extrusion_layers_and_nested_segments(self):
        result = self.capture(["--mode", "3d-extrude", "--z-elements", "32", "--span", "0.2"])
        self.assertEqual(result["params"].z_elements, 32)
        self.assertEqual(result["params"].W, 0.2)
        result = self.capture(["--z-mid-elements", "4", "--z-side-elements", "8"])
        self.assertEqual(result["mode"], "3d-nested")
        self.assertEqual((result["params"].nz_mid, result["params"].nz_left_right), (5, 9))

    def test_incompatible_or_invalid_z_controls(self):
        for args in (["--mode", "2d", "--z-elements", "4"],
                     ["--mode", "3d-extrude", "--z-mid-elements", "4"],
                     ["--mode", "3d-extrude", "--z-elements", "0"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.capture(args)
