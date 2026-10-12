"""Numerical and configuration checks without running Gmsh."""

import unittest
import numpy as np

from naca_mesh.airfoil import coordinates, decode_airfoil, outward_normal_on_outer_surface
from naca_mesh.config import Params
from naca_mesh.grading import create_geometric_grading, first_interval


class AirfoilTests(unittest.TestCase):
    def test_four_digit_parameters(self):
        self.assertEqual(decode_airfoil("4412"), (0.04, 0.4, 0.12))
        self.assertEqual(decode_airfoil("0012"), (0, 0, 0.12))
        for invalid in ("12", 4412, "abcd", "4012", "4400"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                decode_airfoil(invalid)

    def test_symmetric_coordinates_and_normals(self):
        x = np.linspace(0, 1, 101)
        with np.errstate(all="raise"):
            xu, yu, xl, yl = coordinates(x, *decode_airfoil("0012"))
            upper, lower = outward_normal_on_outer_surface(x, *decode_airfoil("0012"))
        np.testing.assert_array_equal(xu, xl)
        np.testing.assert_array_equal(yu, -yl)
        np.testing.assert_allclose(upper[2], lower[2])
        np.testing.assert_allclose(upper[3], -lower[3])
        np.testing.assert_allclose(np.hypot(upper[2], upper[3]), 1)

    def test_normal_perpendicular_to_surface(self):
        x = np.array([0.05, 0.2, 0.6, 0.9])
        upper, lower = outward_normal_on_outer_surface(x)
        plus, minus = coordinates(x + 1e-6), coordinates(x - 1e-6)
        for surface, offset in ((upper, 0), (lower, 2)):
            dx = (plus[offset] - minus[offset]) / 2e-6
            dy = (plus[offset+1] - minus[offset+1]) / 2e-6
            np.testing.assert_allclose(dx*surface[2] + dy*surface[3], 0, atol=1e-8)

    def test_closed_trailing_edge_and_scalar(self):
        values = coordinates(1.0)
        self.assertTrue(all(isinstance(v, float) for v in values))
        np.testing.assert_allclose(values, [1, 0, 1, 0], atol=1e-15)

    def test_sampling_range(self):
        for invalid in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                coordinates(invalid)


class ConfigurationTests(unittest.TestCase):
    def test_defaults_preserve_resolution(self):
        p = Params()
        self.assertEqual((p.upper_surface, p.lower_surface, p.nose_surface), (275, 50, 150))
        self.assertAlmostEqual(p.dz, 0.004)
        self.assertAlmostEqual(2*p.z1 + p.w_in, p.W)

    def test_overrides_recompute_geometry_and_span(self):
        p = Params(airfoil="0012", W=0.2, upper_surface=40)
        np.testing.assert_array_equal(p.xu, p.xU_cos)
        self.assertAlmostEqual(p.dz, 0.008)
        self.assertEqual(p.upper_surface, 40)
        cambered = Params(airfoil="2412")
        expected = coordinates(cambered.xU_cos, 0.02, 0.4, 0.12)
        np.testing.assert_array_equal(cambered.yu, expected[1])

    def test_invalid_nested_configuration(self):
        for kwargs in ({"ny": 1}, {"W": -1}, {"h_in": 0.01}, {"Lx": 2},
                       {"c": 2}, {"closed_TE": False}, {"x_th": 0}, {"airfoil": "4012"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Params(**kwargs)

    def test_uniform_outer_grading(self):
        counts, heights = create_geometric_grading(4, 1.0)
        self.assertEqual(counts, [1, 1, 1, 1])
        np.testing.assert_allclose(heights, [0.25, 0.5, 0.75, 1])
        self.assertEqual(create_geometric_grading(1, 1.0), ([1], [1.0]))

    def test_first_wall_interval_matches(self):
        p = Params()
        self.assertAlmostEqual(first_interval(p.h_in, p.r_near_wake_in, p.ny-1),
                               p.a_out_near_wake, places=13)
        self.assertEqual(first_interval(1, 1e6, 500), 0.0)


if __name__ == "__main__":
    unittest.main()
