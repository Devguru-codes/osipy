"""Generate MR physics figures for osipy documentation.

Produces matplotlib figures used in the explanation/ docs. Run this script
to regenerate all figures into docs/assets/figures/.

    python docs/scripts/gen_physics_figures.py

Where possible, figures use osipy's own signal models to produce
realistic curves (IVIM bi-exponential, DSC concentration).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
FIGURES_DIR = Path(__file__).resolve().parents[1] / "assets" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared style helpers
# ---------------------------------------------------------------------------
STYLE = {
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "lines.linewidth": 2,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
}

COLORS = {
    "primary": "#5E35B1",  # deep purple (matches mkdocs theme)
    "accent": "#FFA000",  # amber
    "blue": "#1565C0",
    "red": "#C62828",
    "green": "#2E7D32",
    "gray": "#616161",
    "light_purple": "#D1C4E9",
    "light_blue": "#BBDEFB",
    "light_amber": "#FFECB3",
    "light_green": "#C8E6C9",
}


def _apply_style():
    plt.rcParams.update(STYLE)
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def _save(fig: plt.Figure, name: str) -> None:
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  saved {path.relative_to(FIGURES_DIR.parents[2])}")


# ===================================================================
# ASL FIGURES
# ===================================================================


def gen_asl_labeling_process() -> None:
    """Schematic of the ASL labeling process: label plane -> imaging volume."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Label region box
    label_box = Rectangle(
        (0.5, 1),
        3.5,
        3,
        linewidth=2,
        edgecolor=COLORS["red"],
        facecolor=COLORS["light_amber"],
        linestyle="-",
        zorder=2,
    )
    ax.add_patch(label_box)
    ax.text(
        2.25,
        3.6,
        "Label Region",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=COLORS["red"],
    )
    ax.text(
        2.25,
        2.5,
        "RF Labeling\n(Invert $M_z$)",
        ha="center",
        va="center",
        fontsize=10,
        color=COLORS["gray"],
    )

    # Imaging region box
    img_box = Rectangle(
        (6, 1),
        3.5,
        3,
        linewidth=2,
        edgecolor=COLORS["primary"],
        facecolor=COLORS["light_purple"],
        linestyle="-",
        zorder=2,
    )
    ax.add_patch(img_box)
    ax.text(
        7.75,
        3.6,
        "Imaging Region",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=COLORS["primary"],
    )
    ax.text(
        7.75,
        2.5,
        "Brain Tissue\n(Measure Signal)",
        ha="center",
        va="center",
        fontsize=10,
        color=COLORS["gray"],
    )

    # Blood flow arrow
    arrow = FancyArrowPatch(
        (4.2, 2.5),
        (5.8, 2.5),
        arrowstyle="-|>",
        mutation_scale=20,
        linewidth=2.5,
        color=COLORS["red"],
        zorder=3,
    )
    ax.add_patch(arrow)
    ax.text(
        5.0,
        3.0,
        "Blood Flow",
        ha="center",
        va="bottom",
        fontsize=10,
        fontstyle="italic",
        color=COLORS["red"],
    )

    # Labels below
    ax.text(
        2.25,
        0.6,
        "Labeling Plane",
        ha="center",
        va="top",
        fontsize=10,
        color=COLORS["gray"],
    )
    ax.text(
        7.75,
        0.6,
        "Imaging Volume",
        ha="center",
        va="top",
        fontsize=10,
        color=COLORS["gray"],
    )

    # Small arrows pointing up from labels
    ax.annotate(
        "",
        xy=(2.25, 1.0),
        xytext=(2.25, 0.65),
        arrowprops={"arrowstyle": "->", "color": COLORS["gray"], "lw": 1.2},
    )
    ax.annotate(
        "",
        xy=(7.75, 1.0),
        xytext=(7.75, 0.65),
        arrowprops={"arrowstyle": "->", "color": COLORS["gray"], "lw": 1.2},
    )

    fig.suptitle("ASL Labeling Process", fontsize=14, fontweight="bold", y=0.98)
    _save(fig, "asl_labeling_process")


def gen_asl_pld_timeline() -> None:
    """Timeline diagram showing labeling duration, PLD, and acquisition."""
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 2.5)
    ax.axis("off")

    y = 1.0
    h = 0.8

    # Labeling block
    label_rect = Rectangle(
        (0, y),
        3,
        h,
        linewidth=1.5,
        edgecolor=COLORS["red"],
        facecolor=COLORS["light_amber"],
    )
    ax.add_patch(label_rect)
    ax.text(
        1.5,
        y + h / 2,
        "Labeling ($\\tau$)",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=COLORS["red"],
    )

    # PLD gap
    ax.annotate(
        "",
        xy=(6.5, y + h + 0.15),
        xytext=(3, y + h + 0.15),
        arrowprops={"arrowstyle": "<->", "color": COLORS["gray"], "lw": 1.5},
    )
    ax.text(
        4.75,
        y + h + 0.3,
        "PLD (blood transit)",
        ha="center",
        va="bottom",
        fontsize=10,
        color=COLORS["gray"],
    )

    # Dashed line for transit
    ax.plot(
        [3.3, 6.2],
        [y + h / 2, y + h / 2],
        "--",
        color=COLORS["red"],
        linewidth=1.2,
        alpha=0.5,
    )
    ax.annotate(
        "",
        xy=(6.2, y + h / 2),
        xytext=(5.5, y + h / 2),
        arrowprops={
            "arrowstyle": "-|>",
            "color": COLORS["red"],
            "lw": 1.2,
            "alpha": 0.5,
        },
    )

    # Imaging block
    acq_rect = Rectangle(
        (6.5, y),
        3,
        h,
        linewidth=1.5,
        edgecolor=COLORS["primary"],
        facecolor=COLORS["light_purple"],
    )
    ax.add_patch(acq_rect)
    ax.text(
        8.0,
        y + h / 2,
        "Acquire Image",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=COLORS["primary"],
    )

    # Timeline axis
    ax.plot([-0.3, 10.3], [y - 0.15, y - 0.15], "-", color="black", lw=1)
    ax.text(10.4, y - 0.15, "Time", ha="left", va="center", fontsize=10)

    # Tick marks
    for x, label in [(0, "Label\nstarts"), (3, "Label\nends"), (6.5, "Readout")]:
        ax.plot([x, x], [y - 0.25, y - 0.05], "-", color="black", lw=1)
        ax.text(
            x, y - 0.35, label, ha="center", va="top", fontsize=9, color=COLORS["gray"]
        )

    fig.suptitle(
        "ASL Timing: Labeling, PLD, and Acquisition",
        fontsize=13,
        fontweight="bold",
        y=1.0,
    )
    _save(fig, "asl_pld_timeline")


# ===================================================================
# DSC FIGURES
# ===================================================================


def gen_dsc_signal_drop() -> None:
    """DSC signal-time curve showing signal drop during bolus passage."""
    t = np.linspace(0, 60, 300)

    # Simulate a gamma-variate-like bolus
    t_bolus = 20.0
    alpha, beta = 3.0, 1.5
    bolus = np.zeros_like(t)
    mask = t > t_bolus
    t_shifted = t[mask] - t_bolus
    bolus[mask] = (t_shifted**alpha) * np.exp(-t_shifted / beta)
    bolus /= bolus.max()

    # Signal = S0 * exp(-TE * r2* * C(t))
    s0 = 1000
    te_r2 = 2.5  # TE * r2* scaling
    signal = s0 * np.exp(-te_r2 * bolus)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, signal, color=COLORS["primary"], linewidth=2.5)

    # Shade baseline
    ax.axvspan(0, t_bolus - 1, alpha=0.08, color=COLORS["green"])
    ax.text(
        10,
        s0 * 0.98,
        "Baseline",
        ha="center",
        va="top",
        fontsize=10,
        color=COLORS["green"],
    )

    # Mark bolus region
    bolus_start = t_bolus
    bolus_end = t_bolus + 15
    ax.axvspan(bolus_start, bolus_end, alpha=0.08, color=COLORS["red"])
    ax.text(
        (bolus_start + bolus_end) / 2,
        s0 * 0.55,
        "Bolus\nPassage",
        ha="center",
        va="center",
        fontsize=10,
        color=COLORS["red"],
    )

    # Arrow showing signal drop
    drop_t = t_bolus + alpha * beta
    drop_idx = np.argmin(np.abs(t - drop_t))
    ax.annotate(
        "Signal drop\n($\\Delta R_2^*$ effect)",
        xy=(t[drop_idx], signal[drop_idx]),
        xytext=(t[drop_idx] + 10, signal[drop_idx] + 80),
        fontsize=9,
        color=COLORS["gray"],
        arrowprops={"arrowstyle": "->", "color": COLORS["gray"], "lw": 1.2},
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal Intensity (a.u.)")
    ax.set_title("DSC-MRI Signal During Bolus Passage", fontweight="bold")
    ax.set_ylim(0, s0 * 1.1)
    _save(fig, "dsc_signal_drop")


def gen_dsc_residue_function() -> None:
    """Residue function R(t) showing exponential-like decay."""
    t = np.linspace(0, 30, 200)

    # Exponential residue function with MTT = 4s
    mtt = 4.0
    R = np.exp(-t / mtt)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, R, color=COLORS["primary"], linewidth=2.5)
    ax.axhline(y=1.0, color=COLORS["gray"], linestyle=":", linewidth=0.8, alpha=0.5)

    # Mark R(0) = 1
    ax.plot(0, 1.0, "o", color=COLORS["primary"], markersize=8, zorder=5)
    ax.text(0.5, 1.02, "$R(0) = 1$", fontsize=10, color=COLORS["primary"])

    # Mark MTT
    r_at_mtt = np.exp(-1)
    ax.plot([mtt, mtt], [0, r_at_mtt], "--", color=COLORS["accent"], linewidth=1.5)
    ax.plot([0, mtt], [r_at_mtt, r_at_mtt], "--", color=COLORS["accent"], linewidth=1.5)
    ax.plot(mtt, r_at_mtt, "o", color=COLORS["accent"], markersize=8, zorder=5)
    ax.text(mtt + 0.3, 0.05, f"MTT = {mtt:.0f} s", fontsize=10, color=COLORS["accent"])

    # Shade tail region
    tail_start = 12
    ax.axvspan(tail_start, t[-1], alpha=0.06, color=COLORS["gray"])
    ax.text(21, 0.15, "Tail", fontsize=10, color=COLORS["gray"], fontstyle="italic")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("$R(t)$")
    ax.set_title("Residue Function", fontweight="bold")
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlim(-0.5, t[-1])
    _save(fig, "dsc_residue_function")


def gen_dsc_bolus_delay() -> None:
    """Side-by-side AIF and delayed tissue curves."""
    t = np.linspace(0, 60, 300)

    # Gamma-variate AIF
    t0_aif = 10.0
    alpha, beta = 3.0, 1.5
    aif = np.zeros_like(t)
    mask_a = t > t0_aif
    ts_a = t[mask_a] - t0_aif
    aif[mask_a] = (ts_a**alpha) * np.exp(-ts_a / beta)
    aif /= aif.max()

    # Delayed + dispersed tissue curve
    delay = 6.0
    tissue = np.zeros_like(t)
    mask_t = t > (t0_aif + delay)
    ts_t = t[mask_t] - (t0_aif + delay)
    tissue[mask_t] = (ts_t ** (alpha + 0.5)) * np.exp(-ts_t / (beta + 0.3))
    tissue /= tissue.max() / 0.7  # lower peak

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)

    # AIF
    ax1.plot(t, aif, color=COLORS["red"], linewidth=2.5)
    ax1.fill_between(t, aif, alpha=0.15, color=COLORS["red"])
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Concentration (a.u.)")
    ax1.set_title("Arterial Input (AIF)", fontweight="bold")
    ax1.axvline(x=t0_aif, color=COLORS["gray"], linestyle=":", linewidth=1)
    ax1.text(t0_aif + 0.5, 0.9, "Arrival", fontsize=9, color=COLORS["gray"])

    # Tissue with delay
    ax2.plot(t, tissue, color=COLORS["blue"], linewidth=2.5)
    ax2.fill_between(t, tissue, alpha=0.15, color=COLORS["blue"])
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Tissue (with delay)", fontweight="bold")
    ax2.axvline(x=t0_aif + delay, color=COLORS["gray"], linestyle=":", linewidth=1)
    ax2.text(t0_aif + delay + 0.5, 0.9, "Arrival", fontsize=9, color=COLORS["gray"])

    # Annotate delay
    ax2.annotate(
        "",
        xy=(t0_aif + delay, -0.08),
        xytext=(t0_aif, -0.08),
        arrowprops={"arrowstyle": "<->", "color": COLORS["accent"], "lw": 2},
        annotation_clip=False,
    )
    ax2.annotate(
        f"Delay = {delay:.0f} s",
        xy=(t0_aif + delay / 2, -0.15),
        ha="center",
        va="top",
        fontsize=10,
        color=COLORS["accent"],
        annotation_clip=False,
    )

    fig.suptitle(
        "Effect of Bolus Delay on Tissue Curve", fontsize=13, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    _save(fig, "dsc_bolus_delay")


# ===================================================================
# IVIM FIGURES
# ===================================================================


def gen_ivim_compartments() -> None:
    """IVIM two-compartment voxel schematic."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Outer voxel box
    voxel = Rectangle(
        (0.3, 0.5),
        9.4,
        4.2,
        linewidth=2.5,
        edgecolor="black",
        facecolor="white",
        zorder=1,
    )
    ax.add_patch(voxel)
    ax.text(
        5.0,
        4.9,
        "TISSUE VOXEL",
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
    )

    # Tissue compartment
    tissue = Rectangle(
        (0.8, 1.0),
        3.8,
        3.0,
        linewidth=1.5,
        edgecolor=COLORS["blue"],
        facecolor=COLORS["light_blue"],
        zorder=2,
    )
    ax.add_patch(tissue)
    ax.text(
        2.7,
        3.6,
        "Tissue Water",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=COLORS["blue"],
    )

    # Draw slow random walk arrows in tissue
    rng = np.random.default_rng(42)
    for _ in range(6):
        cx = rng.uniform(1.3, 4.1)
        cy = rng.uniform(1.5, 3.0)
        dx = rng.uniform(-0.3, 0.3)
        dy = rng.uniform(-0.2, 0.2)
        ax.annotate(
            "",
            xy=(cx + dx, cy + dy),
            xytext=(cx, cy),
            arrowprops={
                "arrowstyle": "->",
                "color": COLORS["blue"],
                "lw": 1,
                "alpha": 0.6,
            },
        )

    ax.text(
        2.7,
        1.35,
        "Slow $D$\nFraction: $(1-f)$",
        ha="center",
        va="center",
        fontsize=10,
        color=COLORS["blue"],
    )

    # Blood compartment
    blood = Rectangle(
        (5.4, 1.0),
        3.8,
        3.0,
        linewidth=1.5,
        edgecolor=COLORS["red"],
        facecolor=COLORS["light_amber"],
        zorder=2,
    )
    ax.add_patch(blood)
    ax.text(
        7.3,
        3.6,
        "Blood Water",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=COLORS["red"],
    )

    # Draw fast pseudo-random arrows in blood
    for _ in range(8):
        cx = rng.uniform(5.9, 8.7)
        cy = rng.uniform(1.5, 3.0)
        dx = rng.uniform(-0.5, 0.5)
        dy = rng.uniform(-0.3, 0.3)
        ax.annotate(
            "",
            xy=(cx + dx, cy + dy),
            xytext=(cx, cy),
            arrowprops={
                "arrowstyle": "->",
                "color": COLORS["red"],
                "lw": 1.2,
                "alpha": 0.6,
            },
        )

    ax.text(
        7.3,
        1.35,
        "Fast $D^*$\nFraction: $f$",
        ha="center",
        va="center",
        fontsize=10,
        color=COLORS["red"],
    )

    fig.suptitle("IVIM Two-Compartment Model", fontsize=14, fontweight="bold", y=0.98)
    _save(fig, "ivim_compartments")


def gen_ivim_signal_decay() -> None:
    """IVIM bi-exponential signal decay vs b-value.

    Uses the IVIM signal equation: S/S0 = f*exp(-b*Dstar) + (1-f)*exp(-b*D)
    """
    b = np.linspace(0, 800, 500)

    # Physiological parameters
    D = 1.0e-3  # mm^2/s
    Dstar = 15.0e-3  # mm^2/s
    f = 0.15

    # Bi-exponential signal
    signal = f * np.exp(-b * Dstar) + (1 - f) * np.exp(-b * D)

    # Individual components
    perfusion_component = f * np.exp(-b * Dstar)
    diffusion_component = (1 - f) * np.exp(-b * D)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Plot on semi-log scale
    ax.semilogy(
        b,
        signal,
        color=COLORS["primary"],
        linewidth=2.5,
        label="Total signal",
        zorder=3,
    )
    ax.semilogy(
        b,
        diffusion_component,
        "--",
        color=COLORS["blue"],
        linewidth=1.5,
        label="Diffusion: $(1-f) \\cdot e^{-bD}$",
        alpha=0.7,
    )
    ax.semilogy(
        b,
        perfusion_component,
        "--",
        color=COLORS["red"],
        linewidth=1.5,
        label="Perfusion: $f \\cdot e^{-bD^*}$",
        alpha=0.7,
    )

    # Shade regions
    ax.axvspan(0, 100, alpha=0.06, color=COLORS["red"])
    ax.axvspan(200, 800, alpha=0.06, color=COLORS["blue"])

    ax.text(
        40,
        0.03,
        "Perfusion\nsensitive",
        ha="center",
        fontsize=9,
        color=COLORS["red"],
        fontstyle="italic",
    )
    ax.text(
        500,
        0.03,
        "Diffusion\ndominated",
        ha="center",
        fontsize=9,
        color=COLORS["blue"],
        fontstyle="italic",
    )

    # Annotate parameters
    param_text = f"$D$ = {D * 1e3:.1f} $\\times 10^{{-3}}$ mm$^2$/s\n"
    param_text += f"$D^*$ = {Dstar * 1e3:.0f} $\\times 10^{{-3}}$ mm$^2$/s\n"
    param_text += f"$f$ = {f:.2f}"
    ax.text(
        0.97,
        0.05,
        param_text,
        fontsize=9,
        color=COLORS["gray"],
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": COLORS["gray"],
            "alpha": 0.8,
        },
    )

    ax.set_xlabel("b-value (s/mm$^2$)")
    ax.set_ylabel("$S / S_0$")
    ax.set_title("IVIM Bi-Exponential Signal Decay", fontweight="bold")
    ax.set_ylim(0.02, 1.2)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2, which="both")
    _save(fig, "ivim_signal_decay")


# ===================================================================
# DCE FIGURES
# ===================================================================


def gen_dce_compartments() -> None:
    """DCE two-compartment exchange model schematic."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Outer voxel box
    voxel = Rectangle(
        (0.3, 0.5),
        9.4,
        4.2,
        linewidth=2.5,
        edgecolor="black",
        facecolor="white",
        zorder=1,
    )
    ax.add_patch(voxel)
    ax.text(
        5.0,
        4.9,
        "TISSUE VOXEL",
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
    )

    # Plasma compartment
    plasma = Rectangle(
        (0.8, 1.0),
        3.5,
        3.0,
        linewidth=1.5,
        edgecolor=COLORS["red"],
        facecolor=COLORS["light_amber"],
        zorder=2,
    )
    ax.add_patch(plasma)
    ax.text(
        2.55,
        3.55,
        "Blood Plasma",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=COLORS["red"],
    )
    ax.text(
        2.55,
        2.5,
        "$C_p(t)$",
        ha="center",
        va="center",
        fontsize=14,
        color=COLORS["red"],
    )
    ax.text(
        2.55,
        1.5,
        "Volume: $v_p$",
        ha="center",
        va="center",
        fontsize=10,
        color=COLORS["gray"],
    )

    # EES compartment
    ees = Rectangle(
        (5.7, 1.0),
        3.5,
        3.0,
        linewidth=1.5,
        edgecolor=COLORS["primary"],
        facecolor=COLORS["light_purple"],
        zorder=2,
    )
    ax.add_patch(ees)
    ax.text(
        7.45,
        3.55,
        "EES",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=COLORS["primary"],
    )
    ax.text(
        7.45,
        2.5,
        "$C_e(t)$",
        ha="center",
        va="center",
        fontsize=14,
        color=COLORS["primary"],
    )
    ax.text(
        7.45,
        1.5,
        "Volume: $v_e$",
        ha="center",
        va="center",
        fontsize=10,
        color=COLORS["gray"],
    )

    # Ktrans arrow (forward)
    arrow_fwd = FancyArrowPatch(
        (4.45, 2.9),
        (5.55, 2.9),
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2.5,
        color=COLORS["green"],
        zorder=3,
    )
    ax.add_patch(arrow_fwd)
    ax.text(
        5.0,
        3.2,
        "$K^{trans}$",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=COLORS["green"],
    )

    # kep arrow (reverse)
    arrow_rev = FancyArrowPatch(
        (5.55, 2.1),
        (4.45, 2.1),
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2.5,
        color=COLORS["accent"],
        zorder=3,
    )
    ax.add_patch(arrow_rev)
    ax.text(
        5.0,
        1.75,
        "$k_{ep}$",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=COLORS["accent"],
    )

    fig.suptitle(
        "DCE Two-Compartment Exchange Model", fontsize=14, fontweight="bold", y=0.98
    )
    _save(fig, "dce_compartments")


# ===================================================================
# MAIN
# ===================================================================


def main() -> None:
    _apply_style()
    print("Generating physics figures ...")

    # ASL
    gen_asl_labeling_process()
    gen_asl_pld_timeline()

    # DSC
    gen_dsc_signal_drop()
    gen_dsc_residue_function()
    gen_dsc_bolus_delay()

    # IVIM
    gen_ivim_compartments()
    gen_ivim_signal_decay()

    # DCE
    gen_dce_compartments()

    print("Done.")


if __name__ == "__main__":
    main()
