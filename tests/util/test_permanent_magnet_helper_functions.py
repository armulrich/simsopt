import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from simsopt.util.permanent_magnet_helper_functions import (
    calculate_modB_on_major_radius,
    initialize_coils,
    initialize_default_kwargs,
    read_focus_coils,
)


class PermanentMagnetHelperFunctionsTests(unittest.TestCase):
    def test_initialize_default_kwargs_rs(self):
        kwargs = initialize_default_kwargs("RS")
        self.assertTrue(kwargs["verbose"])
        self.assertIn("nu", kwargs)
        self.assertIn("max_iter", kwargs)
        self.assertIn("max_iter_RS", kwargs)

    def test_initialize_default_kwargs_gpmo_variants(self):
        for alg in ["GPMO", "GPMO_Backtracking", "GPMOmr", "GPMO_py", "GPMO_ArbVec"]:
            with self.subTest(algorithm=alg):
                kwargs = initialize_default_kwargs(alg)
                self.assertTrue(kwargs["verbose"])
                self.assertIn("K", kwargs)
                self.assertIn("reg_l2", kwargs)
                self.assertIn("nhistory", kwargs)

    def test_read_focus_coils_wrapper(self):
        base = Path(__file__).resolve().parents[1] / "test_files"
        coils_file = base / "muse_tf_coils.focus"
        curves, currents, ncoils = read_focus_coils(str(coils_file))
        self.assertEqual(len(curves), ncoils)
        self.assertEqual(len(currents), ncoils)

    def test_calculate_modB_wrapper_dispatch(self):
        from simsopt.util import coil_optimization_helper_functions as cohf

        with patch.object(cohf, "calculate_modB_on_major_radius", return_value=123.0) as f:
            out = calculate_modB_on_major_radius(bs=object(), s=object())
            self.assertEqual(out, 123.0)
            self.assertTrue(f.called)

    def test_initialize_coils_muse_smoke(self):
        base = Path(__file__).resolve().parents[1] / "test_files"
        with TemporaryDirectory() as td:
            base_curves, curves, coils = initialize_coils(
                "muse", base, s=None, out_dir=td
            )
            self.assertGreater(len(coils), 0)
            self.assertEqual(len(curves), len(coils))
            self.assertEqual(len(base_curves), len(coils))

            vtk_files = list(Path(td).glob("curves_init*.vtu"))
            self.assertTrue(vtk_files)


if __name__ == "__main__":
    unittest.main()

