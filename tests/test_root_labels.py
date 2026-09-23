import unittest

import numpy as np

from algebraics.root_labels import (
    extract_square_factor,
    find_hotspots,
    format_rational,
    pretty_root,
    primitive_coeffs,
)


class FormatTests(unittest.TestCase):
    def test_rational(self):
        self.assertEqual(format_rational(1, 2), r"\frac{1}{2}")
        self.assertEqual(format_rational(-3, 2), r"\frac{-3}{2}")
        self.assertEqual(format_rational(4, 2), "2")
        self.assertEqual(format_rational(-2, -1), "2")

    def test_square_factor(self):
        self.assertEqual(extract_square_factor(4), (2, 1))
        self.assertEqual(extract_square_factor(12), (2, 3))
        self.assertEqual(extract_square_factor(5), (1, 5))

    def test_degree_one(self):
        self.assertEqual(pretty_root(0, (0, 1)), "0")
        self.assertEqual(pretty_root(1, (-1, 1)), "1")
        self.assertEqual(pretty_root(-1, (1, 1)), "-1")
        self.assertEqual(pretty_root(0.5, (-1, 2)), r"\frac{1}{2}")
        self.assertEqual(pretty_root(1.5, (-3, 2)), r"\frac{3}{2}")
        self.assertEqual(pretty_root(2, (-2, 1)), "2")

    def test_imaginary_unit(self):
        self.assertEqual(pretty_root(1j, (1, 0, 1)), r"\mathrm{i}")
        self.assertEqual(pretty_root(-1j, (1, 0, 1)), r"-\mathrm{i}")

    def test_sixth_roots(self):
        z = 0.5 + 1j * np.sqrt(3) / 2
        self.assertEqual(pretty_root(z, (1, -1, 1)), r"\frac{1+\mathrm{i}\sqrt{3}}{2}")
        zc = 0.5 - 1j * np.sqrt(3) / 2
        self.assertEqual(pretty_root(zc, (1, -1, 1)), r"\frac{1-\mathrm{i}\sqrt{3}}{2}")

    def test_golden_ratio(self):
        phi = (1 + np.sqrt(5)) / 2
        self.assertEqual(pretty_root(phi, (-1, -1, 1)), r"\frac{1+\sqrt{5}}{2}")

    def test_eighth_roots(self):
        z = np.exp(1j * np.pi / 4)
        self.assertEqual(pretty_root(z, (1, 0, 0, 0, 1)), r"e^{i\pi/4}")

    def test_primitive_sign(self):
        self.assertEqual(primitive_coeffs((2, 0, 2)), (1, 0, 1))
        self.assertEqual(primitive_coeffs((-1, 0, -1)), (1, 0, 1))


class HotspotTests(unittest.TestCase):
    def test_picks_high_mass_simple_roots(self):
        xs = [0.0] * 200 + [1.0] * 80 + [0.5] * 40 + [0.31] * 5
        ys = [0.0] * 200 + [0.0] * 80 + [0.0] * 40 + [0.22] * 5
        pts = np.zeros(len(xs), dtype=[("x", "f8"), ("y", "f8"), ("h", "i4"), ("o", "i4")])
        pts["x"] = xs
        pts["y"] = ys
        pts["h"] = 4
        pts["o"] = 1
        labels = find_hotspots(pts, min_mass_frac=0.01, min_hits=10, max_labels=10)
        texts = {lab.text for lab in labels}
        self.assertIn("0", texts)
        self.assertIn("1", texts)
        self.assertIn(r"\frac{1}{2}", texts)
        self.assertNotIn("0.31", texts)

    def test_empty(self):
        pts = np.zeros(0, dtype=[("x", "f8"), ("y", "f8"), ("h", "i4"), ("o", "i4")])
        self.assertEqual(find_hotspots(pts), ())


if __name__ == "__main__":
    unittest.main()
