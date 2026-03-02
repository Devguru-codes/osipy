"""Plotting helpers for OSIPI compliance test figures.

Generates visual artifacts from compliance test data. Only called
when pytest is invoked with ``--generate-figures``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend — no display required
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


def plot_forward_model(
    model_name: str,
    cases_data: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Plot predicted vs reference concentration curves.

    Parameters
    ----------
    model_name : str
        Registry name of the model (e.g. ``"tofts"``).
    cases_data : list[dict]
        Each dict has keys: name, time, aif, expected, predicted, params.
    output_dir : Path
        Directory for the saved PNG.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    n_cases = len(cases_data)
    fig, axes = plt.subplots(1, n_cases, figsize=(5 * n_cases, 4), squeeze=False)
    fig.suptitle(f"{model_name} — forward model compliance", fontsize=13)

    for idx, case in enumerate(cases_data):
        ax = axes[0, idx]
        time = case["time"]
        aif = case["aif"]
        expected = case["expected"]
        predicted = case["predicted"]

        # AIF as faint reference
        ax.plot(time, aif, color="gray", alpha=0.3, linewidth=1, label="AIF")
        ax.plot(
            time,
            expected,
            "--",
            color="tab:blue",
            linewidth=1.5,
            label="Reference",
        )
        ax.plot(
            time,
            predicted,
            "-",
            color="tab:orange",
            linewidth=1.5,
            label="Predicted",
        )

        ax.set_title(case["name"], fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Concentration")
        ax.legend(fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_dir / f"{model_name}_forward.png", dpi=150)
    plt.close(fig)


def plot_parameter_recovery(
    model_name: str,
    true_params: dict[str, float],
    recovered_params: dict[str, float],
    tolerances: dict[str, dict[str, float]],
    output_dir: Path,
) -> None:
    """Plot true vs recovered parameters with tolerance bands.

    Parameters
    ----------
    model_name : str
        Registry name of the model.
    true_params : dict[str, float]
        Ground-truth parameter values.
    recovered_params : dict[str, float]
        Fitted parameter values.
    tolerances : dict[str, dict[str, float]]
        Per-parameter tolerance dicts with ``"absolute"`` and ``"relative"`` keys.
    output_dir : Path
        Directory for the saved PNG.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Only plot parameters present in both dicts
    param_names = [p for p in true_params if p in recovered_params]
    if not param_names:
        return

    n = len(param_names)
    x = np.arange(n)
    width = 0.3

    fig, ax = plt.subplots(figsize=(max(4, 2 * n), 4))

    true_vals = [true_params[p] for p in param_names]
    rec_vals = [recovered_params[p] for p in param_names]

    # Compute tolerance bands and pass/fail colours
    colours = []
    tol_lo = []
    tol_hi = []
    for p, tv in zip(param_names, true_vals, strict=True):
        tol = tolerances.get(p, {"absolute": 0.1, "relative": 0.1})
        abs_tol = tol["absolute"]
        rel_tol = tol["relative"]
        band = max(abs_tol, rel_tol * abs(tv))
        tol_lo.append(tv - band)
        tol_hi.append(tv + band)

        rv = recovered_params[p]
        abs_err = abs(rv - tv)
        rel_err = abs_err / (abs(tv) + 1e-10)
        passed = (abs_err <= tol["absolute"]) or (rel_err <= tol["relative"])
        colours.append("tab:green" if passed else "tab:red")

    # Bar chart
    ax.bar(x - width / 2, true_vals, width, label="True", color="tab:blue", alpha=0.8)
    ax.bar(x + width / 2, rec_vals, width, label="Recovered", color=colours, alpha=0.8)

    # Tolerance bands as error bars on the "true" bars
    for i in range(n):
        band = (tol_hi[i] - tol_lo[i]) / 2
        ax.errorbar(
            x[i] - width / 2,
            true_vals[i],
            yerr=band,
            fmt="none",
            ecolor="black",
            capsize=4,
            linewidth=1,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(param_names)
    ax.set_ylabel("Value")
    ax.set_title(f"{model_name} — parameter recovery")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / f"{model_name}_recovery.png", dpi=150)
    plt.close(fig)


def plot_osipi_dro_recovery(
    model_name: str,
    recovery_data: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Scatter plot of true vs recovered parameters from OSIPI DRO data.

    One subplot per parameter. Identity line with tolerance band shading.
    Green/red markers for pass/fail cases.

    Parameters
    ----------
    model_name : str
        Registry name of the model (e.g. ``"tofts"``).
    recovery_data : list[dict]
        Each dict has keys: label, true_params, recovered_params, passed.
    output_dir : Path
        Directory for the saved PNG.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not recovery_data:
        return

    # Collect all parameter names across cases
    param_names: list[str] = []
    for case in recovery_data:
        for p in case["true_params"]:
            if p not in param_names and p in case.get("recovered_params", {}):
                param_names.append(p)

    if not param_names:
        return

    n_params = len(param_names)
    fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 4.5), squeeze=False)
    fig.suptitle(
        f"{model_name} — OSIPI DRO parameter recovery ({len(recovery_data)} cases)",
        fontsize=13,
    )

    for col, param in enumerate(param_names):
        ax = axes[0, col]
        true_vals: list[float] = []
        rec_vals: list[float] = []
        colors: list[str] = []

        for case in recovery_data:
            tv = case["true_params"].get(param)
            rv = case["recovered_params"].get(param)
            if tv is None or rv is None:
                continue
            true_vals.append(tv)
            rec_vals.append(rv)
            colors.append("tab:green" if case["passed"] else "tab:red")

        if not true_vals:
            ax.set_visible(False)
            continue

        true_arr = np.array(true_vals)
        rec_arr = np.array(rec_vals)

        # Identity line
        lo = min(true_arr.min(), rec_arr.min())
        hi = max(true_arr.max(), rec_arr.max())
        margin = (hi - lo) * 0.1 + 1e-10
        lim = (lo - margin, hi + margin)
        ax.plot(lim, lim, "k--", linewidth=0.8, alpha=0.5, label="Identity")

        # Scatter
        ax.scatter(
            true_arr,
            rec_arr,
            c=colors,
            s=40,
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )

        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"True {param}")
        ax.set_ylabel(f"Recovered {param}")
        ax.set_title(param, fontsize=10)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_dir / f"{model_name}_dro_recovery.png", dpi=150)
    plt.close(fig)


def plot_delay_aif_comparison(
    model_name: str,
    delay_cases: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Plot original AIF vs delayed AIF with true and estimated delay markers.

    For each test case, shows the original AIF and the AIF shifted by the
    estimated delay, with vertical lines indicating true and recovered delay.

    Parameters
    ----------
    model_name : str
        Registry name of the model (e.g. ``"tofts"``).
    delay_cases : list[dict]
        Each dict has keys: label, time, aif, true_delay, recovered_delay,
        passed.
    output_dir : Path
        Directory for the saved PNG.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not delay_cases:
        return

    # Limit to a grid of cases (max 4 cols)
    n_cases = len(delay_cases)
    n_cols = min(4, n_cases)
    n_rows = (n_cases + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )
    fig.suptitle(
        f"{model_name} — AIF delay comparison ({n_cases} cases)",
        fontsize=13,
    )

    for idx, case in enumerate(delay_cases):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        time = case["time"]
        aif = case["aif"]
        true_delay = case["true_delay"]
        rec_delay = case.get("recovered_delay")

        # Original AIF
        ax.plot(time, aif, "-", color="tab:blue", linewidth=1.5, label="Original AIF")

        # AIF shifted by estimated delay (what the fitter used)
        if rec_delay is not None:
            shifted_time = time - rec_delay
            aif_shifted = np.interp(shifted_time, time, aif, left=0.0, right=aif[-1])
            ax.plot(
                time,
                aif_shifted,
                "-",
                color="tab:orange",
                linewidth=1.5,
                label=f"Delayed AIF (est={rec_delay:.2f}s)",
            )

        # True delay vertical line
        ax.axvline(
            true_delay,
            color="tab:blue",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label=f"True delay={true_delay:.2f}s",
        )

        # Estimated delay vertical line
        if rec_delay is not None:
            ax.axvline(
                rec_delay,
                color="tab:orange",
                linestyle=":",
                linewidth=1,
                alpha=0.7,
                label=f"Est delay={rec_delay:.2f}s",
            )

        ax.set_title(case["label"], fontsize=8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Concentration")
        ax.legend(fontsize=6, loc="upper right")

    # Hide unused subplots
    for idx in range(n_cases, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_dir / f"{model_name}_delay_aif.png", dpi=150)
    plt.close(fig)
