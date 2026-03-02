#!/usr/bin/env python3
"""Generate OSIPI CodeCollection reference data.

Generates JSON reference files containing forward model predictions
for each DCE pharmacokinetic model using the Parker AIF and standard
parameter combinations from OSIPI CodeCollection patterns.

Run: python3 scripts/generate_osipi_reference.py
"""

import json
import sys
from pathlib import Path

import numpy as np

# Compatibility shim: numpy < 2.0 uses trapz, >= 2.0 uses trapezoid
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapezoid  # type: ignore[attr-defined]

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from osipy.common.aif.population import ParkerAIF
from osipy.dce.models import get_model

OUTPUT_DIR = (
    Path(__file__).parent.parent / "tests" / "fixtures" / "osipi_codecollection" / "dce"
)

# Model configurations: model_name -> list of {name, params}
MODEL_CONFIGS = {
    "tofts": [
        {"name": "case1_low_permeability", "params": {"Ktrans": 0.1, "ve": 0.2}},
        {"name": "case2_moderate_permeability", "params": {"Ktrans": 0.25, "ve": 0.4}},
        {"name": "case3_very_low_permeability", "params": {"Ktrans": 0.05, "ve": 0.1}},
    ],
    "extended_tofts": [
        {
            "name": "case1_low_permeability",
            "params": {"Ktrans": 0.1, "ve": 0.2, "vp": 0.02},
        },
        {
            "name": "case2_moderate_permeability",
            "params": {"Ktrans": 0.25, "ve": 0.4, "vp": 0.05},
        },
        {
            "name": "case3_very_low_permeability",
            "params": {"Ktrans": 0.05, "ve": 0.1, "vp": 0.01},
        },
    ],
    "patlak": [
        {"name": "case1_low_permeability", "params": {"Ktrans": 0.1, "vp": 0.02}},
        {"name": "case2_moderate_permeability", "params": {"Ktrans": 0.25, "vp": 0.05}},
        {"name": "case3_very_low_permeability", "params": {"Ktrans": 0.05, "vp": 0.01}},
    ],
    "2cxm": [
        {
            "name": "case1_moderate_flow",
            "params": {"Fp": 50.0, "PS": 5.0, "ve": 0.2, "vp": 0.02},
        },
        {
            "name": "case2_low_flow_high_ps",
            "params": {"Fp": 30.0, "PS": 10.0, "ve": 0.3, "vp": 0.05},
        },
        {
            "name": "case3_high_flow_low_ps",
            "params": {"Fp": 80.0, "PS": 2.0, "ve": 0.15, "vp": 0.01},
        },
    ],
    "2cum": [
        {"name": "case1_moderate_flow", "params": {"Fp": 50.0, "PS": 5.0, "vp": 0.02}},
        {
            "name": "case2_low_flow_high_ps",
            "params": {"Fp": 30.0, "PS": 10.0, "vp": 0.05},
        },
        {
            "name": "case3_high_flow_low_ps",
            "params": {"Fp": 80.0, "PS": 2.0, "vp": 0.01},
        },
    ],
}


def generate_reference(model_name: str, cases: list[dict]) -> dict:
    """Generate reference data for a single model."""
    model = get_model(model_name)
    parker = ParkerAIF()

    # 300 seconds, 60 timepoints (OSIPI convention)
    time = np.linspace(0, 300, 60)
    aif_result = parker(time)
    aif = aif_result.concentration

    test_cases = []
    for case in cases:
        concentration = model.predict(time, aif, case["params"])
        test_cases.append(
            {
                "name": case["name"],
                "params": case["params"],
                "time_seconds": time.tolist(),
                "aif": aif.tolist(),
                "concentration": concentration.tolist(),
            }
        )

    return {
        "model": model_name,
        "model_display_name": model.name,
        "aif_model": "Parker",
        "time_range_seconds": [0, 300],
        "n_timepoints": 60,
        "test_cases": test_cases,
    }


def main() -> None:
    """Generate all reference JSON files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, cases in MODEL_CONFIGS.items():
        print(f"Generating {model_name}...")
        ref_data = generate_reference(model_name, cases)

        output_path = OUTPUT_DIR / f"{model_name}.json"
        with output_path.open("w") as f:
            json.dump(ref_data, f, indent=2)

        print(f"  -> {output_path} ({len(cases)} test cases)")

    print("Done.")


if __name__ == "__main__":
    main()
