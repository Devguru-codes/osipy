"""Shared pytest fixtures for osipy tests.

This module provides DRO fixtures for all modalities.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Path to fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"
OSIPI_DRO_DIR = FIXTURES_DIR / "osipi_dro"


# =============================================================================
# Synthetic DRO Generation Helpers
# =============================================================================


def generate_synthetic_dce_data(
    shape: tuple[int, int, int] = (32, 32, 8),
    n_timepoints: int = 50,
    noise_level: float = 0.01,
    seed: int = 42,
) -> dict[str, NDArray[np.floating]]:
    """Generate synthetic DCE-MRI data with known ground truth.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Spatial dimensions (x, y, z).
    n_timepoints : int
        Number of time points.
    noise_level : float
        Relative noise level.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
        - signal: 4D array (x, y, z, t)
        - t1_map: 3D array
        - ktrans: 3D array (ground truth)
        - ve: 3D array (ground truth)
        - vp: 3D array (ground truth)
        - aif: 1D array
        - time: 1D array
        - mask: 3D boolean array
    """
    rng = np.random.default_rng(seed)

    # Generate ground truth parameter maps with realistic spatial variation
    ktrans = np.zeros(shape)
    ve = np.zeros(shape)
    vp = np.zeros(shape)

    # Create regions with different parameters
    x, y, _z = shape
    ktrans[x // 4 : 3 * x // 4, y // 4 : 3 * y // 4, :] = 0.1
    ktrans[x // 3 : 2 * x // 3, y // 3 : 2 * y // 3, :] = 0.2

    ve[x // 4 : 3 * x // 4, y // 4 : 3 * y // 4, :] = 0.2
    ve[x // 3 : 2 * x // 3, y // 3 : 2 * y // 3, :] = 0.3

    vp[x // 4 : 3 * x // 4, y // 4 : 3 * y // 4, :] = 0.02
    vp[x // 3 : 2 * x // 3, y // 3 : 2 * y // 3, :] = 0.05

    # Add small random variation
    ktrans += rng.uniform(-0.01, 0.01, shape)
    ktrans = np.clip(ktrans, 0.01, 0.5)

    ve += rng.uniform(-0.02, 0.02, shape)
    ve = np.clip(ve, 0.1, 0.5)

    vp += rng.uniform(-0.005, 0.005, shape)
    vp = np.clip(vp, 0.01, 0.1)

    # T1 map (ms)
    t1_map = 1000 + rng.uniform(-100, 100, shape)

    # Time vector (minutes)
    time = np.linspace(0, 5, n_timepoints)

    # Parker AIF model (simplified)
    def parker_aif(t: NDArray[np.floating]) -> NDArray[np.floating]:
        """Simplified Parker AIF."""
        a1, a2 = 0.809, 0.330
        t1, t2 = 0.17046, 0.365
        sigma1, sigma2 = 0.0563, 0.132
        alpha, beta = 1.05, 0.1685
        s, tau = 38.078, 0.483

        # Gaussian components
        g1 = (
            a1
            / (sigma1 * np.sqrt(2 * np.pi))
            * np.exp(-((t - t1) ** 2) / (2 * sigma1**2))
        )
        g2 = (
            a2
            / (sigma2 * np.sqrt(2 * np.pi))
            * np.exp(-((t - t2) ** 2) / (2 * sigma2**2))
        )

        # Exponential component (simplified)
        exp_comp = alpha * np.exp(-beta * t) / (1 + np.exp(-s * (t - tau)))

        return g1 + g2 + exp_comp

    aif = parker_aif(time)

    # Generate signal using extended Tofts model
    signal = np.zeros((*shape, n_timepoints))

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                kt = ktrans[i, j, k]
                v_e = ve[i, j, k]
                v_p = vp[i, j, k]

                if kt > 0 and v_e > 0:
                    # Convolution of AIF with exponential
                    kep = kt / v_e
                    dt = time[1] - time[0]

                    # Impulse response
                    impulse = kt * np.exp(-kep * time)

                    # Convolution
                    conv = np.convolve(aif, impulse)[:n_timepoints] * dt

                    # Extended Tofts: C(t) = vp*Cp(t) + Ktrans * int(Cp*exp(-kep*tau))
                    signal[i, j, k, :] = v_p * aif + conv

    # Add noise
    signal += rng.normal(0, noise_level * np.max(signal), signal.shape)
    signal = np.maximum(signal, 0)

    # Create mask
    mask = ktrans > 0.05

    return {
        "signal": signal.astype(np.float32),
        "t1_map": t1_map.astype(np.float32),
        "ktrans": ktrans.astype(np.float32),
        "ve": ve.astype(np.float32),
        "vp": vp.astype(np.float32),
        "aif": aif.astype(np.float32),
        "time": time.astype(np.float32),
        "mask": mask,
    }


def generate_synthetic_dsc_data(
    shape: tuple[int, int, int] = (32, 32, 8),
    n_timepoints: int = 60,
    noise_level: float = 0.02,
    seed: int = 42,
) -> dict[str, NDArray[np.floating]]:
    """Generate synthetic DSC-MRI data with known ground truth.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Spatial dimensions (x, y, z).
    n_timepoints : int
        Number of time points.
    noise_level : float
        Relative noise level.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
        - signal: 4D array (x, y, z, t)
        - delta_r2: 4D array (x, y, z, t)
        - cbv: 3D array (ground truth)
        - cbf: 3D array (ground truth)
        - mtt: 3D array (ground truth)
        - aif: 1D array
        - time: 1D array
        - mask: 3D boolean array
    """
    rng = np.random.default_rng(seed)

    # Generate ground truth parameter maps
    cbv = np.zeros(shape)
    cbf = np.zeros(shape)

    x, y, _z = shape

    # Gray matter region
    cbv[x // 4 : 3 * x // 4, y // 4 : 3 * y // 4, :] = 4.0
    cbf[x // 4 : 3 * x // 4, y // 4 : 3 * y // 4, :] = 60.0

    # White matter region
    cbv[x // 3 : 2 * x // 3, y // 3 : 2 * y // 3, :] = 2.0
    cbf[x // 3 : 2 * x // 3, y // 3 : 2 * y // 3, :] = 25.0

    # Add variation
    cbv += rng.uniform(-0.5, 0.5, shape)
    cbv = np.clip(cbv, 1.0, 10.0)

    cbf += rng.uniform(-5, 5, shape)
    cbf = np.clip(cbf, 10.0, 100.0)

    # MTT = CBV / CBF (scaled)
    mtt = cbv / (cbf / 60.0)  # Convert CBF to per-second
    mtt = np.clip(mtt, 3.0, 12.0)

    # Time vector (seconds)
    time = np.linspace(0, 60, n_timepoints)

    # Gamma-variate AIF
    def gamma_variate(
        t: NDArray[np.floating],
        t0: float = 10.0,
        alpha: float = 3.0,
        beta: float = 1.5,
    ) -> NDArray[np.floating]:
        """Generate gamma-variate curve."""
        result = np.zeros_like(t)
        mask = t > t0
        t_shifted = t[mask] - t0
        result[mask] = t_shifted**alpha * np.exp(-t_shifted / beta)
        result = result / np.max(result) * 10.0  # Scale to reasonable delta R2*
        return result

    aif = gamma_variate(time)

    # Generate delta R2* and signal
    delta_r2 = np.zeros((*shape, n_timepoints))
    te = 0.025  # 25 ms echo time
    s0 = 1000  # Baseline signal

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if cbv[i, j, k] > 1.5:
                    # Scale AIF by local CBV
                    local_scale = cbv[i, j, k] / 4.0
                    local_mtt = mtt[i, j, k]

                    # Convolve with residue function (exponential)
                    residue = np.exp(-time / local_mtt)
                    residue = residue / np.sum(residue)

                    tissue_curve = np.convolve(aif, residue, mode="full")[:n_timepoints]
                    delta_r2[i, j, k, :] = tissue_curve * local_scale

    # Convert to signal
    signal = s0 * np.exp(-te * delta_r2)
    signal += rng.normal(0, noise_level * s0, signal.shape)
    signal = np.maximum(signal, 1)

    # Create mask
    mask = cbv > 1.5

    return {
        "signal": signal.astype(np.float32),
        "delta_r2": delta_r2.astype(np.float32),
        "cbv": cbv.astype(np.float32),
        "cbf": cbf.astype(np.float32),
        "mtt": mtt.astype(np.float32),
        "aif": aif.astype(np.float32),
        "time": time.astype(np.float32),
        "mask": mask,
    }


def generate_synthetic_asl_data(
    shape: tuple[int, int, int] = (32, 32, 8),
    n_plds: int = 6,
    noise_level: float = 0.05,
    seed: int = 42,
) -> dict[str, NDArray[np.floating]]:
    """Generate synthetic ASL data with known ground truth.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Spatial dimensions (x, y, z).
    n_plds : int
        Number of post-labeling delays.
    noise_level : float
        Relative noise level.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
        - control: 4D array (x, y, z, pld)
        - label: 4D array (x, y, z, pld)
        - difference: 4D array (x, y, z, pld)
        - m0: 3D array
        - cbf: 3D array (ground truth)
        - att: 3D array (ground truth)
        - plds: 1D array
        - mask: 3D boolean array
    """
    rng = np.random.default_rng(seed)

    # Generate ground truth parameter maps
    cbf = np.zeros(shape)
    att = np.zeros(shape)
    m0 = np.zeros(shape)

    x, y, _z = shape

    # Gray matter
    cbf[x // 4 : 3 * x // 4, y // 4 : 3 * y // 4, :] = 60.0
    att[x // 4 : 3 * x // 4, y // 4 : 3 * y // 4, :] = 1.2
    m0[x // 4 : 3 * x // 4, y // 4 : 3 * y // 4, :] = 1000.0

    # White matter
    cbf[x // 3 : 2 * x // 3, y // 3 : 2 * y // 3, :] = 25.0
    att[x // 3 : 2 * x // 3, y // 3 : 2 * y // 3, :] = 1.8
    m0[x // 3 : 2 * x // 3, y // 3 : 2 * y // 3, :] = 800.0

    # Add variation
    cbf += rng.uniform(-5, 5, shape)
    cbf = np.clip(cbf, 10.0, 100.0)

    att += rng.uniform(-0.1, 0.1, shape)
    att = np.clip(att, 0.5, 2.5)

    m0 += rng.uniform(-50, 50, shape)
    m0 = np.clip(m0, 500.0, 1500.0)

    # PLDs (seconds)
    plds = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0])[:n_plds]

    # ASL parameters
    label_duration = 1.8  # seconds
    t1_blood = 1.65  # seconds
    t1_tissue = 1.3  # seconds
    labeling_efficiency = 0.85

    # Generate ASL signal using Buxton model
    control = np.zeros((*shape, n_plds))
    label = np.zeros((*shape, n_plds))

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if cbf[i, j, k] > 15:
                    local_cbf = cbf[i, j, k]
                    local_att = att[i, j, k]
                    local_m0 = m0[i, j, k]

                    for p, pld in enumerate(plds):
                        # Simplified Buxton model for pCASL
                        pld + label_duration

                        if pld < local_att:
                            # Before arrival
                            delta_m = 0
                        elif pld < local_att + label_duration:
                            # During bolus arrival
                            delta_m = (
                                2
                                * labeling_efficiency
                                * local_m0
                                * local_cbf
                                / 6000.0  # Convert CBF to per-second
                                * t1_blood
                                * np.exp(-pld / t1_blood)
                                * (1 - np.exp(-(pld - local_att) / t1_tissue))
                            )
                        else:
                            # After bolus
                            delta_m = (
                                2
                                * labeling_efficiency
                                * local_m0
                                * local_cbf
                                / 6000.0
                                * t1_blood
                                * np.exp(-pld / t1_blood)
                                * np.exp(
                                    -(pld - local_att - label_duration) / t1_tissue
                                )
                                * (1 - np.exp(-label_duration / t1_tissue))
                            )

                        control[i, j, k, p] = local_m0
                        label[i, j, k, p] = local_m0 - delta_m

    # Add noise
    noise_scale = noise_level * np.mean(m0[m0 > 0])
    control += rng.normal(0, noise_scale, control.shape)
    label += rng.normal(0, noise_scale, label.shape)

    difference = control - label

    # Create mask
    mask = cbf > 15

    return {
        "control": control.astype(np.float32),
        "label": label.astype(np.float32),
        "difference": difference.astype(np.float32),
        "m0": m0.astype(np.float32),
        "cbf": cbf.astype(np.float32),
        "att": att.astype(np.float32),
        "plds": plds.astype(np.float32),
        "mask": mask,
    }


def generate_synthetic_ivim_data(
    shape: tuple[int, int, int] = (32, 32, 8),
    n_bvalues: int = 10,
    noise_level: float = 0.01,
    seed: int = 42,
) -> dict[str, NDArray[np.floating]]:
    """Generate synthetic IVIM data with known ground truth.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Spatial dimensions (x, y, z).
    n_bvalues : int
        Number of b-values.
    noise_level : float
        Relative noise level.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
        - signal: 4D array (x, y, z, b)
        - d: 3D array (ground truth D, x10^-3 mm^2/s)
        - d_star: 3D array (ground truth D*, x10^-3 mm^2/s)
        - f: 3D array (ground truth f)
        - b_values: 1D array
        - mask: 3D boolean array
    """
    rng = np.random.default_rng(seed)

    # Generate ground truth parameter maps
    d = np.zeros(shape)
    d_star = np.zeros(shape)
    f = np.zeros(shape)

    x, y, _z = shape

    # Tissue region 1
    d[x // 4 : 3 * x // 4, y // 4 : 3 * y // 4, :] = 1.0e-3
    d_star[x // 4 : 3 * x // 4, y // 4 : 3 * y // 4, :] = 10.0e-3
    f[x // 4 : 3 * x // 4, y // 4 : 3 * y // 4, :] = 0.1

    # Tissue region 2
    d[x // 3 : 2 * x // 3, y // 3 : 2 * y // 3, :] = 0.8e-3
    d_star[x // 3 : 2 * x // 3, y // 3 : 2 * y // 3, :] = 15.0e-3
    f[x // 3 : 2 * x // 3, y // 3 : 2 * y // 3, :] = 0.15

    # Add variation
    d += rng.uniform(-0.1e-3, 0.1e-3, shape)
    d = np.clip(d, 0.5e-3, 2.0e-3)

    d_star += rng.uniform(-2e-3, 2e-3, shape)
    d_star = np.clip(d_star, 5e-3, 20e-3)

    f += rng.uniform(-0.02, 0.02, shape)
    f = np.clip(f, 0.05, 0.30)

    # B-values (s/mm^2)
    b_values = np.array([0, 10, 20, 50, 100, 200, 400, 600, 800, 1000])[:n_bvalues]

    # Generate IVIM signal
    s0 = 1.0
    signal = np.zeros((*shape, n_bvalues))

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if d[i, j, k] > 0.4e-3:
                    local_d = d[i, j, k]
                    local_d_star = d_star[i, j, k]
                    local_f = f[i, j, k]

                    # Bi-exponential IVIM model
                    for b_idx, b in enumerate(b_values):
                        signal[i, j, k, b_idx] = s0 * (
                            local_f * np.exp(-b * local_d_star)
                            + (1 - local_f) * np.exp(-b * local_d)
                        )

    # Add Rician noise (approximated as Gaussian for high SNR)
    signal += rng.normal(0, noise_level, signal.shape)
    signal = np.maximum(signal, 0.01)

    # Create mask
    mask = d > 0.4e-3

    return {
        "signal": signal.astype(np.float32),
        "d": d.astype(np.float32),
        "d_star": d_star.astype(np.float32),
        "f": f.astype(np.float32),
        "b_values": b_values.astype(np.float32),
        "mask": mask,
    }


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def dce_dro() -> dict[str, NDArray[np.floating]]:
    """Provide synthetic DCE-MRI DRO data.

    Returns
    -------
    dict
        DCE-MRI data with signal, parameters, and ground truth.
    """
    return generate_synthetic_dce_data()


@pytest.fixture
def dce_dro_large() -> dict[str, NDArray[np.floating]]:
    """Provide larger DCE-MRI DRO for integration tests.

    Returns
    -------
    dict
        DCE-MRI data with larger spatial dimensions.
    """
    return generate_synthetic_dce_data(shape=(64, 64, 16), n_timepoints=80)


@pytest.fixture
def dsc_dro() -> dict[str, NDArray[np.floating]]:
    """Provide synthetic DSC-MRI DRO data.

    Returns
    -------
    dict
        DSC-MRI data with signal, parameters, and ground truth.
    """
    return generate_synthetic_dsc_data()


@pytest.fixture
def dsc_dro_large() -> dict[str, NDArray[np.floating]]:
    """Provide larger DSC-MRI DRO for integration tests.

    Returns
    -------
    dict
        DSC-MRI data with larger spatial dimensions.
    """
    return generate_synthetic_dsc_data(shape=(64, 64, 16), n_timepoints=90)


@pytest.fixture
def asl_dro() -> dict[str, NDArray[np.floating]]:
    """Provide synthetic ASL DRO data.

    Returns
    -------
    dict
        ASL data with control/label images and ground truth.
    """
    return generate_synthetic_asl_data()


@pytest.fixture
def asl_dro_large() -> dict[str, NDArray[np.floating]]:
    """Provide larger ASL DRO for integration tests.

    Returns
    -------
    dict
        ASL data with larger spatial dimensions.
    """
    return generate_synthetic_asl_data(shape=(64, 64, 16), n_plds=8)


@pytest.fixture
def ivim_dro() -> dict[str, NDArray[np.floating]]:
    """Provide synthetic IVIM DRO data.

    Returns
    -------
    dict
        IVIM data with multi-b signal and ground truth.
    """
    return generate_synthetic_ivim_data()


@pytest.fixture
def ivim_dro_large() -> dict[str, NDArray[np.floating]]:
    """Provide larger IVIM DRO for integration tests.

    Returns
    -------
    dict
        IVIM data with larger spatial dimensions.
    """
    return generate_synthetic_ivim_data(shape=(64, 64, 16), n_bvalues=12)


@pytest.fixture
def affine_matrix() -> NDArray[np.floating]:
    """Provide standard affine transformation matrix.

    Returns
    -------
    NDArray
        4x4 identity affine matrix.
    """
    return np.eye(4, dtype=np.float32)


@pytest.fixture
def small_mask() -> NDArray[np.bool_]:
    """Provide small test mask.

    Returns
    -------
    NDArray
        Boolean mask with central region True.
    """
    mask = np.zeros((32, 32, 8), dtype=bool)
    mask[8:24, 8:24, 2:6] = True
    return mask


@pytest.fixture
def fixtures_dir() -> Path:
    """Provide path to fixtures directory.

    Returns
    -------
    Path
        Path to tests/fixtures directory.
    """
    return FIXTURES_DIR


@pytest.fixture
def osipi_dro_dir() -> Path:
    """Provide path to OSIPI DRO directory.

    Returns
    -------
    Path
        Path to tests/fixtures/osipi_dro directory.
    """
    return OSIPI_DRO_DIR


# =============================================================================
# Fixtures that load actual DRO files from disk
# =============================================================================


def _load_osipi_dce_dro() -> dict[str, NDArray[np.floating]] | None:
    """Load OSIPI DCE DRO from disk if available."""
    try:
        import nibabel as nib

        dce_dir = OSIPI_DRO_DIR / "dce"
        if not (dce_dir / "Ktrans.nii").exists():
            return None

        ktrans = nib.load(dce_dir / "Ktrans.nii").get_fdata()
        kep = nib.load(dce_dir / "kep.nii").get_fdata()
        vp = nib.load(dce_dir / "vp.nii").get_fdata()
        r10 = nib.load(dce_dir / "R10.nii").get_fdata()

        # Derive ve from Ktrans/kep (with safe division)
        ve = np.zeros_like(ktrans)
        mask = kep > 0
        ve[mask] = ktrans[mask] / kep[mask]
        ve = np.clip(ve, 0, 1)  # Physiological range

        # Load AIF if available
        aif = None
        if (dce_dir / "AIF.npz").exists():
            aif_data = np.load(dce_dir / "AIF.npz")
            aif = aif_data[next(iter(aif_data.keys()))]

        return {
            "ktrans": ktrans.astype(np.float32),
            "kep": kep.astype(np.float32),
            "ve": ve.astype(np.float32),
            "vp": vp.astype(np.float32),
            "r10": r10.astype(np.float32),
            "aif": aif,
            "mask": (ktrans > 0.01).astype(bool),
            "affine": np.eye(4, dtype=np.float32),
        }
    except Exception:
        return None


def _load_osipi_dsc_dro() -> dict[str, NDArray[np.floating]] | None:
    """Load OSIPI DSC DRO from disk if available."""
    dsc_dir = OSIPI_DRO_DIR / "dsc"
    if not (dsc_dir / "cbv.npy").exists():
        return None

    return {
        "cbv": np.load(dsc_dir / "cbv.npy"),
        "cbf": np.load(dsc_dir / "cbf.npy"),
        "mtt": np.load(dsc_dir / "mtt.npy"),
        "aif": np.load(dsc_dir / "aif.npy"),
        "time": np.load(dsc_dir / "time.npy"),
        "mask": np.load(dsc_dir / "mask.npy"),
        "affine": np.eye(4, dtype=np.float32),
    }


def _load_osipi_asl_dro() -> dict[str, NDArray[np.floating]] | None:
    """Load OSIPI ASL DRO from disk if available."""
    asl_dir = OSIPI_DRO_DIR / "asl"
    if not (asl_dir / "cbf.npy").exists():
        return None

    return {
        "cbf": np.load(asl_dir / "cbf.npy"),
        "att": np.load(asl_dir / "att.npy"),
        "m0": np.load(asl_dir / "m0.npy"),
        "plds": np.load(asl_dir / "plds.npy"),
        "mask": np.load(asl_dir / "mask.npy"),
        "affine": np.eye(4, dtype=np.float32),
    }


def _load_osipi_ivim_dro() -> dict[str, NDArray[np.floating]] | None:
    """Load OSIPI IVIM DRO from disk if available."""
    ivim_dir = OSIPI_DRO_DIR / "ivim"
    if not (ivim_dir / "d.npy").exists():
        return None

    return {
        "d": np.load(ivim_dir / "d.npy"),
        "d_star": np.load(ivim_dir / "d_star.npy"),
        "f": np.load(ivim_dir / "f.npy"),
        "b_values": np.load(ivim_dir / "b_values.npy"),
        "mask": np.load(ivim_dir / "mask.npy"),
        "affine": np.eye(4, dtype=np.float32),
    }


@pytest.fixture
def osipi_dce_dro() -> dict[str, NDArray[np.floating]]:
    """Load OSIPI DCE DRO from disk.

    Returns
    -------
    dict
        DCE parameter maps (Ktrans, ve, vp, kep, R10) from OSIPI challenge.
        Falls back to synthetic data if files not available.
    """
    data = _load_osipi_dce_dro()
    if data is None:
        pytest.skip("OSIPI DCE DRO files not available")
    return data


@pytest.fixture
def osipi_dsc_dro() -> dict[str, NDArray[np.floating]]:
    """Load OSIPI DSC DRO from disk.

    Returns
    -------
    dict
        DSC parameter maps (CBV, CBF, MTT) with ground truth values.
        Falls back to synthetic data if files not available.
    """
    data = _load_osipi_dsc_dro()
    if data is None:
        pytest.skip("OSIPI DSC DRO files not available")
    return data


@pytest.fixture
def osipi_asl_dro() -> dict[str, NDArray[np.floating]]:
    """Load OSIPI ASL DRO from disk.

    Returns
    -------
    dict
        ASL parameter maps (CBF, ATT) with ground truth values.
        Falls back to synthetic data if files not available.
    """
    data = _load_osipi_asl_dro()
    if data is None:
        pytest.skip("OSIPI ASL DRO files not available")
    return data


@pytest.fixture
def osipi_ivim_dro() -> dict[str, NDArray[np.floating]]:
    """Load OSIPI IVIM DRO from disk.

    Returns
    -------
    dict
        IVIM parameter maps (D, D*, f) with ground truth values.
        Falls back to synthetic data if files not available.
    """
    data = _load_osipi_ivim_dro()
    if data is None:
        pytest.skip("OSIPI IVIM DRO files not available")
    return data


# =============================================================================
# OSIPI Compliance Summary
# =============================================================================

_COMPLIANCE_RESULTS: list[dict[str, Any]] = []
_FIGURES_OUTPUT_DIR = Path("output/compliance")


def pytest_addoption(parser):
    """Register ``--generate-figures`` CLI flag."""
    parser.addoption(
        "--generate-figures",
        action="store_true",
        default=False,
        help="Generate compliance figures in output/compliance/",
    )


@pytest.fixture
def generate_figures(request):
    """Return True when ``--generate-figures`` was passed to pytest."""
    return request.config.getoption("--generate-figures")


def record_compliance_result(name: str, passed: bool, details: str = "") -> None:
    """Record a compliance test result for the summary table.

    Parameters
    ----------
    name : str
        Name of the compliance test.
    passed : bool
        Whether the test passed.
    details : str
        Optional details about the result.
    """
    _COMPLIANCE_RESULTS.append(
        {
            "name": name,
            "passed": passed,
            "details": details,
        }
    )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print OSIPI Tolerance Compliance summary at the end of test run."""
    if not _COMPLIANCE_RESULTS:
        return

    terminalreporter.write_sep("=", "OSIPI Tolerance Compliance Summary")

    passed = sum(1 for r in _COMPLIANCE_RESULTS if r["passed"])
    total = len(_COMPLIANCE_RESULTS)

    for result in _COMPLIANCE_RESULTS:
        status = "PASS" if result["passed"] else "FAIL"
        line = f"  [{status}] {result['name']}"
        if result["details"]:
            line += f" — {result['details']}"
        terminalreporter.write_line(line)

    terminalreporter.write_line("")
    terminalreporter.write_line(f"  {passed}/{total} compliance checks passed")

    if config.getoption("--generate-figures", default=False):
        out = _FIGURES_OUTPUT_DIR.resolve()
        terminalreporter.write_line(f"  Figures saved to: {out}")

    terminalreporter.write_sep("=", "")
