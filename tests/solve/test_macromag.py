import unittest
import json
import numpy as np
from pathlib import Path
from typing import Optional
from unittest.mock import patch

from simsopt.solve.macromag import (
    MacroMag,
    Tiles,
    assemble_blocks_subset,
    build_prism,
    rotation_angle,
    _rotation_matrix,
    muse2tiles,
)


def make_trivial_tiles(n: int, *, cube_dim: float = 1.0, offsets: Optional[np.ndarray] = None) -> Tiles:
    """
    Small synthetic Tiles instance for unit tests.

    - Cubic prisms with the same size and rotation.
    - Easy axis along +z.
    - mu_r_ea = mu_r_oa = 1 (chi = 0).
    - M_rem = [1, 2, ..., n].
    """
    tiles = Tiles(n)
    tiles.tile_type = 2  # prism

    if offsets is None:
        offsets = np.zeros((n, 3), dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    if offsets.shape != (n, 3):
        raise ValueError(f"offsets must have shape ({n}, 3), got {offsets.shape}")

    dims = np.array([cube_dim, cube_dim, cube_dim], dtype=np.float64)
    for i in range(n):
        tiles.offset = (offsets[i], i)
        tiles.size = (dims, i)
        tiles.rot = ((0.0, 0.0, 0.0), i)
        tiles.M_rem = (float(i + 1), i)
        tiles.mu_r_ea = (1.0, i)
        tiles.mu_r_oa = (1.0, i)
        tiles.u_ea = ((0.0, 0.0, 1.0), i)
    return tiles


def make_coupled_tiles(
    n: int,
    *,
    cube_dim: float = 0.01,
    offsets: Optional[np.ndarray] = None,
    mu_r: float = 2.0,
) -> Tiles:
    """
    Like :func:`make_trivial_tiles`, but with finite susceptibility (mu_r != 1).
    """
    tiles = make_trivial_tiles(n, cube_dim=cube_dim, offsets=offsets)
    for i in range(n):
        tiles.mu_r_ea = (float(mu_r), i)
        tiles.mu_r_oa = (float(mu_r), i)
    return tiles


class MacroMagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path(__file__).parent.parent / "test_files"
        cls.csv_path = base / "magtense_zot80_3d.csv"
        cls.ref_subset_path = base / "muse_tensor_subset.json"

    # --- basic validation tests ---

    def test_demag_tensor_small_cube_properties(self):
        """
        Demag tensor sanity checks on a tiny synthetic system.

        For a cube, symmetry implies the self-demag block has equal diagonal
        entries and trace(N_ii) = -1 (our convention returns -N in H_d = -N M).
        """
        offsets = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=np.float64)
        tiles = make_trivial_tiles(2, cube_dim=0.01, offsets=offsets)
        macro = MacroMag(tiles)

        N = macro.fast_get_demag_tensor(cache=False)
        self.assertEqual(N.shape, (2, 2, 3, 3))

        # Symmetry: each 3×3 block is symmetric.
        for i in range(2):
            for j in range(2):
                with self.subTest(i=i, j=j):
                    np.testing.assert_allclose(N[i, j], N[i, j].T, atol=1e-14, rtol=0.0)

        # Cube self-demag: equal diagonal, trace ~ -1, diag ~ -1/3.
        diag = np.diag(N[0, 0])
        self.assertAlmostEqual(float(np.trace(N[0, 0])), -1.0, places=10)
        np.testing.assert_allclose(diag, (-1.0 / 3.0) * np.ones(3), atol=5e-6, rtol=0.0)

        # Reciprocity for identical cubes in identical orientation: N_01 == N_10.
        np.testing.assert_allclose(N[0, 1], N[1, 0], atol=1e-14, rtol=0.0)

    def test_demag_tensor_cache_reuse(self):
        offsets = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=np.float64)
        tiles = make_trivial_tiles(2, cube_dim=0.01, offsets=offsets)
        macro = MacroMag(tiles)

        N1 = macro.fast_get_demag_tensor(cache=True)
        N2 = macro.fast_get_demag_tensor(cache=True)
        self.assertIs(N1, N2)

    def test_direct_solve_magnitude_constant(self):
        """
        When chi = 0 and H_coil = 0, direct solve should yield M = M_rem * u_ea.
        This ensures equivalence between MacroMag and uncoupled GPMO for μ=1.
        """
        tiles = make_trivial_tiles(4)
        macro = MacroMag(tiles)

        # Any valid neighbors array is sufficient here: chi=0 short-circuits the solve.
        neighbors = np.tile(np.arange(tiles.n, dtype=np.int64), (tiles.n, 1))
        macro, A = macro.direct_solve_neighbor_sparse(
            neighbors=neighbors,
            use_coils=False,
            krylov_tol=1e-8,
            krylov_it=200,
            print_progress=False,
            x0=None,
            H_a_override=None,
            drop_tol=0.0,
        )
        self.assertIsNone(A)

        norms = np.linalg.norm(tiles.M, axis=1)
        expected = tiles.M_rem
        for idx, (n_val, e_val) in enumerate(zip(norms, expected)):
            with self.subTest(tile=idx):
                self.assertAlmostEqual(
                    n_val, e_val, places=8,
                    msg=f"Tile {idx}: |M| = {n_val:.12f}, expected {e_val:.12f}"
                )

    def test_direct_solve_chi_zero_short_circuit(self):
        tiles = make_trivial_tiles(2, cube_dim=0.01)
        macro = MacroMag(tiles)

        N_dummy = np.zeros((tiles.n, tiles.n, 3, 3), dtype=np.float64)
        _, A = macro.direct_solve(
            use_coils=False,
            krylov_tol=1e-10,
            krylov_it=10,
            N_new_rows=N_dummy,
            print_progress=False,
            x0=None,
            H_a_override=None,
            A_prev=None,
            prev_n=0,
        )
        self.assertIsNone(A)
        np.testing.assert_allclose(tiles.M, tiles.M_rem[:, None] * tiles.u_ea, atol=1e-14, rtol=0.0)

    def test_direct_solve_use_coils_requires_field(self):
        tiles = make_coupled_tiles(1, cube_dim=0.01, mu_r=2.0)
        macro = MacroMag(tiles)
        N_dummy = np.zeros((tiles.n, tiles.n, 3, 3), dtype=np.float64)

        with self.assertRaises(ValueError):
            macro.direct_solve(
                use_coils=True,
                krylov_tol=1e-10,
                krylov_it=10,
                N_new_rows=N_dummy,
                print_progress=False,
                x0=None,
                H_a_override=None,
                A_prev=None,
                prev_n=0,
            )

    def test_direct_solve_print_progress_callback(self):
        tiles = make_coupled_tiles(1, cube_dim=0.01, mu_r=2.0)
        macro = MacroMag(tiles)
        N_dummy = np.zeros((tiles.n, tiles.n, 3, 3), dtype=np.float64)

        with patch("builtins.print") as p:
            macro.direct_solve(
                use_coils=False,
                krylov_tol=1e-12,
                krylov_it=3,
                N_new_rows=N_dummy,
                print_progress=True,
                x0=None,
                H_a_override=None,
                A_prev=None,
                prev_n=0,
            )
        self.assertTrue(p.called)

    def test_muse_reference_tensor_subset(self):
        """
        Regression check against a fixed reference subset of the MUSE demag tensor computed via MAGTENSE.

        This avoids storing the full 9736×9736 tensor in the repository while still
        pinning a handful of representative demag blocks to known values.
        """
        with open(self.ref_subset_path, "r", encoding="utf-8") as f:
            ref = json.load(f)
        idx = np.asarray(ref["indices"], dtype=np.int64)
        ref_blocks = np.asarray(ref["blocks"], dtype=np.float64)

        tiles = muse2tiles(str(self.csv_path), magnetization=1.1658e6)
        macro = MacroMag(tiles)

        blocks = assemble_blocks_subset(macro.centres, macro.half, macro.Rg2l, idx, idx)
        np.testing.assert_allclose(blocks, ref_blocks, atol=1e-12, rtol=0.0)

    def test_prism_demag_local_singular_points(self):
        a = b = c = 0.5

        N_self = MacroMag._prism_N_local_nb(a, b, c, 0.0, 0.0, 0.0)
        self.assertEqual(N_self.shape, (3, 3))
        self.assertTrue(np.all(np.isfinite(N_self)))
        np.testing.assert_allclose(N_self, N_self.T, atol=1e-14, rtol=0.0)
        self.assertAlmostEqual(float(np.trace(N_self)), -1.0, places=10)

        # Evaluate at a corner-like point to exercise dx==0/dy==0/dz==0 and getF_limit paths.
        N_corner = MacroMag._prism_N_local_nb(a, b, c, a, b, c)
        self.assertTrue(np.all(np.isfinite(N_corner)))
        np.testing.assert_allclose(N_corner, N_corner.T, atol=1e-12, rtol=0.0)

    def test_demag_field_at_rotation_matches_manual(self):
        tiles = make_trivial_tiles(1, cube_dim=0.01)
        tiles.rot = ((0.2, -0.1, 0.05), 0)
        tiles.M = ((1.0, 2.0, 3.0), 0)
        macro = MacroMag(tiles)

        pts = np.zeros((2, 3), dtype=np.float64)
        demag_tensor = np.zeros((1, 2, 3, 3), dtype=np.float64)
        demag_tensor[0, :, 0, 0] = 1.0
        demag_tensor[0, :, 1, 1] = 2.0
        demag_tensor[0, :, 2, 2] = 3.0

        H = macro._demag_field_at(pts, demag_tensor)

        Rg2l, Rl2g = MacroMag.get_rotation_matrices(*tiles.rot[0])
        M_loc = Rg2l @ tiles.M[0]
        H_loc = -np.einsum("pqr,r->pq", demag_tensor[0], M_loc)
        H_expected = (Rl2g @ H_loc.T).T
        np.testing.assert_allclose(H, H_expected, atol=1e-14, rtol=0.0)

    # minimal tests for Tiles API

    def test_tile_proxy_and_setters(self):
        """Basic Tiles set/get consistency."""
        tiles = make_trivial_tiles(2)
        self.assertEqual(tiles[0].M_rem, 1.0)
        self.assertEqual(tiles[1].M_rem, 2.0)
        tiles[1].M_rem = 5.5
        self.assertEqual(tiles.M_rem[1], 5.5)
        self.assertEqual(tiles.M_rem[0], 1.0)
        with self.assertRaises(IndexError):
            _ = tiles[2]
        s = str(tiles)
        self.assertIn("Tile_0", s)
        self.assertIn("Tile_1", s)

    def test_tiles_bulk_setters_and_vertices_validation(self):
        tiles = Tiles(2)
        tiles.tile_type = 2
        tiles.center_pos = [1.0, 2.0, 3.0]
        tiles.dev_center = [0.1, 0.2, 0.3]
        tiles.size = [[0.01, 0.02, 0.03], [0.02, 0.03, 0.04]]
        tiles.offset = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]
        tiles.rot = [[0.0, 0.0, 0.0], [0.1, -0.2, 0.3]]
        tiles.M_rem = [1.0, 2.0]
        tiles.mu_r_ea = 1.5
        tiles.mu_r_oa = 1.2
        tiles.M = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        tiles.u_ea = [[0.0, 0.0, 1.0], [0.0, 0.6, 0.8]]

        with self.assertRaises(ValueError):
            tiles.vertices = (np.zeros((3, 3)), 0)

    def test_rotation_matrices(self):
        """Rotation matrices should be orthonormal inverses."""
        Rg2l, Rl2g = MacroMag.get_rotation_matrices(0.1, 0.2, 0.3)
        I = np.eye(3)
        self.assertTrue(np.allclose(Rl2g @ Rg2l, I, atol=1e-12))

    def test_set_easy_axis_error(self):
        tiles = Tiles(1)
        with self.assertRaises(TypeError):
            tiles.set_easy_axis(val=1.23, idx=0)

    def test_rotation_angle_roundtrip(self):
        angles = (0.3, -0.2, 0.5)

        R_xyz = _rotation_matrix(*angles, xyz=True)
        a2, b2, c2 = rotation_angle(R_xyz, xyz=True)
        R_xyz2 = _rotation_matrix(a2, b2, c2, xyz=True)
        np.testing.assert_allclose(R_xyz, R_xyz2, atol=1e-14, rtol=0.0)

        R_zyx = _rotation_matrix(*angles, xyz=False)
        a3, b3, c3 = rotation_angle(R_zyx, xyz=False)
        R_zyx2 = _rotation_matrix(a3, b3, c3, xyz=False)
        np.testing.assert_allclose(R_zyx, R_zyx2, atol=1e-14, rtol=0.0)

        with self.assertRaises(ValueError):
            rotation_angle(np.eye(2))

    def test_assemble_blocks_subset_empty(self):
        tiles = make_trivial_tiles(1, cube_dim=0.01)
        macro = MacroMag(tiles)
        centres = macro.centres
        half = macro.half
        Rg2l = macro.Rg2l
        out = assemble_blocks_subset(centres, half, Rg2l, np.array([], dtype=np.int32), np.array([0], dtype=np.int32))
        self.assertEqual(out.shape, (0, 1, 3, 3))
        out = assemble_blocks_subset(centres, half, Rg2l, np.array([0], dtype=np.int32), np.array([], dtype=np.int32))
        self.assertEqual(out.shape, (1, 0, 3, 3))

    def test_load_coils_and_coil_field(self):
        tiles = make_trivial_tiles(1, cube_dim=0.01)
        macro = MacroMag(tiles)
        with self.assertRaises(ValueError):
            macro.coil_field_at(np.zeros((1, 3)))

        coil_file = self.csv_path.parent / "muse_tf_coils.focus"
        macro.load_coils(coil_file, current_scale=1.1)
        H = macro.coil_field_at(np.zeros((2, 3)))
        self.assertEqual(H.shape, (2, 3))
        self.assertTrue(np.all(np.isfinite(H)))

    def test_direct_solve_finite_mu_zero_demag(self):
        offsets = np.array([[0.0, 0.0, 0.0], [0.03, 0.0, 0.0]], dtype=np.float64)
        tiles = make_coupled_tiles(2, cube_dim=0.01, offsets=offsets, mu_r=2.0)
        macro = MacroMag(tiles)

        # Use a zero demag operator so A=I but still exercise the full assembly path.
        N_zero = np.zeros((tiles.n, tiles.n, 3, 3), dtype=np.float64)
        macro, A = macro.direct_solve(
            use_coils=False,
            krylov_tol=1e-10,
            krylov_it=50,
            N_new_rows=N_zero,
            print_progress=False,
            x0=None,
            H_a_override=None,
            A_prev=None,
            prev_n=0,
        )
        self.assertIsNotNone(A)
        self.assertEqual(A.shape, (3 * tiles.n, 3 * tiles.n))

        # With H_a=0 and N=0, the solve reduces to M = M_rem * u_ea.
        np.testing.assert_allclose(tiles.M, tiles.M_rem[:, None] * tiles.u_ea, atol=1e-10, rtol=0.0)

    def test_direct_solve_incremental_matches_full(self):
        offsets = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=np.float64)

        tiles_full = make_coupled_tiles(2, cube_dim=0.01, offsets=offsets, mu_r=2.0)
        macro_full = MacroMag(tiles_full)
        N_full = macro_full.fast_get_demag_tensor(cache=False)
        _, _ = macro_full.direct_solve(
            use_coils=False,
            krylov_tol=1e-10,
            krylov_it=200,
            N_new_rows=N_full,
            print_progress=False,
            x0=None,
            H_a_override=None,
            A_prev=None,
            prev_n=0,
        )
        M_full = tiles_full.M.copy()

        # Build A_prev from the 1-tile system
        tiles_prev = make_coupled_tiles(1, cube_dim=0.01, offsets=offsets[:1], mu_r=2.0)
        macro_prev = MacroMag(tiles_prev)
        N_prev = N_full[:1, :1]
        _, A_prev = macro_prev.direct_solve(
            use_coils=False,
            krylov_tol=1e-10,
            krylov_it=200,
            N_new_rows=N_prev,
            print_progress=False,
            x0=None,
            H_a_override=None,
            A_prev=None,
            prev_n=0,
        )

        tiles_inc = make_coupled_tiles(2, cube_dim=0.01, offsets=offsets, mu_r=2.0)
        macro_inc = MacroMag(tiles_inc)
        N_new_rows = N_full[1:, :1]
        N_new_cols = N_full[:1, 1:]
        N_new_diag = N_full[1:, 1:]

        _, A_inc = macro_inc.direct_solve(
            use_coils=False,
            krylov_tol=1e-10,
            krylov_it=200,
            N_new_rows=N_new_rows,
            N_new_cols=N_new_cols,
            N_new_diag=N_new_diag,
            print_progress=False,
            x0=None,
            H_a_override=None,
            A_prev=A_prev,
            prev_n=1,
        )
        self.assertIsNotNone(A_inc)
        self.assertEqual(A_inc.shape, (6, 6))
        np.testing.assert_allclose(tiles_inc.M, M_full, atol=1e-8, rtol=1e-8)

    def test_direct_solve_neighbor_sparse_nontrivial(self):
        offsets = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.04, 0.0, 0.0]], dtype=np.float64)
        tiles = make_coupled_tiles(3, cube_dim=0.01, offsets=offsets, mu_r=2.0)
        macro = MacroMag(tiles)

        # neighbors[i,:] must include valid indices; include self + one neighbor
        neighbors = np.array([[0, 1], [1, 2], [2, 1]], dtype=np.int64)
        macro, A_sparse = macro.direct_solve_neighbor_sparse(
            neighbors=neighbors,
            use_coils=False,
            krylov_tol=1e-8,
            krylov_it=200,
            print_progress=False,
            x0=None,
            H_a_override=None,
            drop_tol=0.0,
        )
        self.assertIsNotNone(A_sparse)
        self.assertEqual(A_sparse.shape, (9, 9))
        self.assertEqual(tiles.M.shape, (3, 3))

        with self.assertRaises(ValueError):
            macro.direct_solve_neighbor_sparse(neighbors=np.zeros((2, 2), dtype=np.int64))

    def test_direct_solve_neighbor_sparse_drop_tol_and_progress(self):
        offsets = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=np.float64)
        tiles = make_coupled_tiles(2, cube_dim=0.01, offsets=offsets, mu_r=2.0)
        macro = MacroMag(tiles)

        neighbors = np.array([[0, 1], [1, 0]], dtype=np.int64)
        with patch("builtins.print") as p:
            _, A_sparse = macro.direct_solve_neighbor_sparse(
                neighbors=neighbors,
                use_coils=False,
                krylov_tol=1e-8,
                krylov_it=10,
                print_progress=True,
                x0=None,
                H_a_override=None,
                drop_tol=1e9,
            )
        self.assertIsNotNone(A_sparse)
        self.assertEqual(A_sparse.shape, (6, 6))
        self.assertTrue(p.called)

    def test_tiles_refine_prism(self):
        prism = build_prism(
            lwh=[0.02, 0.02, 0.02],
            center=[0.0, 0.0, 0.0],
            rot=[0.0, 0.0, 0.0],
            mag_angle=[0.0, 0.0],
            mu=[1.0, 1.0],
            remanence=[1.0],
        )
        self.assertEqual(prism.n, 1)
        prism.refine_prism(idx=0, mat=[2, 1, 1])
        self.assertEqual(prism.n, 2)
        # refined tiles should still be prisms and have reduced size
        self.assertTrue(np.all(prism.tile_type == 2))
        np.testing.assert_allclose(prism.size[0], np.array([0.01, 0.02, 0.02]), atol=1e-14, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
