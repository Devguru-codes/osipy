# How to Add a Custom ASL Quantification Model

Add a new ASL labeling scheme and quantification equation in a single file.

## Steps

### 1. Create your model file

Create a new file, e.g., `osipy/asl/quantification/my_labeling.py`:

!!! example "Define a custom ASL quantification model"

    ```python
    """Custom ASL quantification model for a new labeling scheme."""

    from typing import TYPE_CHECKING, Any

    import numpy as np

    from osipy.common.backend.array_module import get_array_module
    from osipy.asl.quantification import BaseQuantificationModel
    from osipy.asl.quantification.registry import register_quantification_model

    if TYPE_CHECKING:
        from numpy.typing import NDArray


    @register_quantification_model("my_labeling_single_pld")
    class MyLabelingSinglePLDModel(BaseQuantificationModel):
        """Custom quantification model for my labeling scheme."""

        @property
        def name(self) -> str:
            return "my_labeling_single_pld"

        @property
        def labeling_type(self) -> str:
            return "my_labeling"

        def quantify(self, delta_m, m0, params):
            xp = get_array_module(delta_m, m0)

            # Your quantification equation here
            # Example: CBF = (6000 * lambda * delta_m) / (2 * alpha * T1b * M0 * ...)
            cbf = (
                6000.0
                * params.partition_coefficient
                * delta_m
                * xp.exp(params.pld / params.t1_blood)
            ) / (
                2.0
                * params.labeling_efficiency
                * params.t1_blood
                * m0
            )

            return cbf
    ```

### 2. Use it

!!! example "Use the registered quantification model"

    ```python
    # Import to trigger registration
    import osipy.asl.quantification.my_labeling

    from osipy.asl.quantification.registry import (
        get_quantification_model,
        list_quantification_models,
    )

    # Verify registration
    print(list_quantification_models())
    # ['casl_single_pld', 'my_labeling_single_pld', 'pasl_single_pld', 'pcasl_single_pld']

    # Use directly
    model = get_quantification_model("my_labeling_single_pld")
    cbf = model.quantify(delta_m, m0, params)
    ```

## Adding a custom ATT estimation model

For multi-PLD support, create a model inheriting from `BaseATTModel`:

!!! example "Define a custom ATT estimation model"

    ```python
    from osipy.asl.quantification.att_base import BaseATTModel
    from osipy.asl.quantification.att_registry import register_att_model


    @register_att_model("my_att_model")
    class MyATTModel(BaseATTModel):
        """BaseATTModel is an alias for BaseASLModel(BaseSignalModel)."""

        @property
        def name(self) -> str:
            return "My ATT Model"

        @property
        def reference(self) -> str:
            return "Author et al. (2025). Journal Name."

        @property
        def labeling_type(self) -> str:
            return "pcasl"

        @property
        def parameters(self) -> list[str]:
            return ["CBF", "ATT"]

        @property
        def parameter_units(self) -> dict[str, str]:
            return {"CBF": "mL/100g/min", "ATT": "ms"}

        def get_bounds(self) -> dict[str, tuple[float, float]]:
            return {"CBF": (0.0, 150.0), "ATT": (0.0, 4000.0)}
    ```

!!! example "Use custom ATT model for multi-PLD quantification"

    ```python
    from osipy.asl.quantification import quantify_multi_pld

    result = quantify_multi_pld(data, m0, params, mask=mask)
    ```

## Adding a custom M0 calibration method

!!! example "Define a custom M0 calibration method"

    ```python
    from osipy.asl.calibration.base import BaseM0Calibration
    from osipy.asl.calibration.registry import register_m0_calibration


    @register_m0_calibration("my_calibration")
    class MyM0Calibration(BaseM0Calibration):
        @property
        def name(self) -> str:
            return "my_calibration"

        def calibrate(self, asl_data, m0_image, params, mask=None):
            # Your calibration logic
            # Return (calibrated_data, m0_values)
            ...
    ```
