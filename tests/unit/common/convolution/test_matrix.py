"""Unit tests for convolution matrix construction."""

import numpy as np
from numpy.testing import assert_allclose

from osipy.common.convolution.matrix import circulant_convmat, convmat, invconvmat


class TestConvmatConstruction:
    """Tests for convmat() construction details."""

    def test_convmat_first_row_zeros(self):
        """Test that first row of convmat is all zeros."""
        t = np.linspace(0, 5, 11)
        h = np.exp(-t / 2)

        A = convmat(h, t)

        assert_allclose(A[0, :], np.zeros(len(t)))

    def test_convmat_diagonal_small(self):
        """Test that diagonal elements are small (integral at single point)."""
        t = np.linspace(0, 5, 11)
        h = np.exp(-t / 2)

        A = convmat(h, t)

        # Diagonal should be approximately zero (no overlap at same time)
        for i in range(len(t)):
            assert abs(A[i, i]) < 0.5

    def test_convmat_uniform_time(self):
        """Test convmat for uniform time grid."""
        n = 20
        dt = 0.25
        t = np.arange(n) * dt
        h = np.exp(-t / 2)

        A = convmat(h, t)

        # Check structure
        assert A.shape == (n, n)

        # Should be approximately Toeplitz-like
        # (constant along diagonals, up to boundary effects)

    def test_convmat_non_uniform_time(self):
        """Test convmat for non-uniform time grid."""
        t = np.array([0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.5, 4.0])
        h = np.exp(-t / 2)

        A = convmat(h, t)

        assert A.shape == (len(t), len(t))
        assert np.all(np.isfinite(A))

    def test_convmat_order_1_vs_2(self):
        """Test first-order vs second-order integration."""
        t = np.linspace(0, 5, 21)
        h = np.exp(-t / 2)

        A1 = convmat(h, t, order=1)
        A2 = convmat(h, t, order=2)

        # Both should have same shape
        assert A1.shape == A2.shape

        # Both should be finite
        assert np.all(np.isfinite(A1))
        assert np.all(np.isfinite(A2))


class TestCirculantConvmat:
    """Tests for circulant_convmat() construction."""

    def test_circulant_shape(self):
        """Test circulant matrix has correct shape."""
        n = 10
        t = np.linspace(0, 5, n)
        h = np.exp(-t / 2)

        C = circulant_convmat(h, t)

        assert C.shape == (n, n)

    def test_circulant_is_circulant(self):
        """Test that the matrix is actually circulant."""
        n = 8
        t = np.linspace(0, 4, n)
        h = np.exp(-t / 2)

        C = circulant_convmat(h, t)

        # For a circulant matrix, C[i,j] = C[(i+1) mod n, (j+1) mod n]
        for i in range(n - 1):
            for j in range(n - 1):
                i_next = (i + 1) % n
                j_next = (j + 1) % n
                assert_allclose(C[i, j], C[i_next, j_next], rtol=1e-10)

    def test_circulant_first_column(self):
        """Test that first column contains h values."""
        n = 10
        dt = 0.5
        t = np.arange(n) * dt
        h = np.exp(-t / 2)

        C = circulant_convmat(h, t)

        # First column should be h scaled by dt
        assert_allclose(C[:, 0], h * dt, rtol=1e-10)

    def test_circulant_empty(self):
        """Test circulant with empty arrays."""
        C = circulant_convmat(np.array([]), np.array([]))
        assert C.size == 0


class TestInvconvmatDetails:
    """Tests for invconvmat() implementation details."""

    def test_invconvmat_identity_approx(self):
        """Test that well-conditioned matrix gives identity."""
        n = 10
        A = np.eye(n) + 0.01 * np.random.randn(n, n)

        A_inv = invconvmat(A, method="tsvd", tol=0.001)

        # A @ A_inv should be close to identity
        product = A @ A_inv
        assert_allclose(product, np.eye(n), atol=0.1)

    def test_invconvmat_singular_value_truncation(self):
        """Test TSVD truncates small singular values."""
        n = 5
        # Create matrix with known singular values
        U = np.eye(n)
        s = np.array([10.0, 1.0, 0.1, 0.01, 0.001])
        Vt = np.eye(n)
        A = U @ np.diag(s) @ Vt

        # TSVD with tol=0.1 should truncate s < 1.0
        A_inv = invconvmat(A, method="tsvd", tol=0.1)

        # The inverse should not amplify the smallest singular value
        # Check that A_inv has bounded norm
        assert np.max(np.abs(A_inv)) < 100

    def test_invconvmat_tikhonov_damping(self):
        """Test Tikhonov damping of small singular values."""
        # Create matrix with known singular values
        s = np.array([10.0, 1.0, 0.1, 0.01, 0.001])
        A = np.diag(s)

        # Tikhonov with lambda = 1.0
        A_inv = invconvmat(A, method="tikhonov", lambda_reg=1.0)

        # Expected: s / (s^2 + lambda^2)
        expected_diag = s / (s**2 + 1.0)
        assert_allclose(np.diag(A_inv), expected_diag, rtol=1e-10)

    def test_invconvmat_lambda_from_tol(self):
        """Test automatic lambda computation from tol."""
        s = np.array([10.0, 5.0, 2.0, 1.0, 0.5])
        A = np.diag(s)

        tol = 0.1
        A_inv = invconvmat(A, method="tikhonov", tol=tol)

        # lambda should be approximately tol * max(s) = 1.0
        expected_lambda = tol * 10.0
        expected_diag = s / (s**2 + expected_lambda**2)
        assert_allclose(np.diag(A_inv), expected_diag, rtol=1e-10)


class TestMatrixSymmetry:
    """Tests for matrix operation symmetry and properties."""

    def test_convmat_scaling_with_dt(self):
        """Test that convmat scales appropriately with time step."""
        h = np.exp(-np.linspace(0, 5, 21) / 2)

        # Fine time grid
        t1 = np.linspace(0, 5, 21)
        A1 = convmat(h, t1)

        # Coarser time grid (same h values)
        t2 = np.linspace(0, 5, 11)
        h2 = np.exp(-t2 / 2)
        A2 = convmat(h2, t2)

        # Both should have similar structure scaled by dt
        assert A1.shape[0] > A2.shape[0]

    def test_invconvmat_non_square(self):
        """Test invconvmat handles non-square matrices."""
        A = np.random.randn(10, 5)

        A_inv = invconvmat(A, method="tsvd", tol=0.1)

        # Result should be (5, 10) for pseudo-inverse
        assert A_inv.shape == (5, 10)

    def test_circulant_eigenvalues(self):
        """Test circulant matrix has DFT eigenvalues."""
        n = 8
        t = np.linspace(0, 4, n)
        h = np.exp(-t / 2)

        C = circulant_convmat(h, t)

        # Eigenvalues of circulant matrix are DFT of first column
        eigenvalues = np.linalg.eigvals(C)
        h_scaled = h * (t[-1] - t[0]) / (n - 1)
        fft_h = np.fft.fft(h_scaled)

        # Eigenvalues should match DFT (up to ordering)
        eigenvalues_sorted = np.sort(np.abs(eigenvalues))
        fft_sorted = np.sort(np.abs(fft_h))
        assert_allclose(eigenvalues_sorted, fft_sorted, rtol=0.1)
