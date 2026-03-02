# How to Add a Custom DSC Deconvolution Method

Add a new deconvolution method to osipy in a single file.

## Steps

### 1. Create your deconvolver

Create a new file, e.g., `osipy/dsc/deconvolution/my_deconv.py`:

!!! example "Define a custom deconvolver"

    ```python
    """Custom deconvolution method for DSC-MRI."""

    from typing import TYPE_CHECKING, Any

    import numpy as np

    from osipy.common.backend.array_module import get_array_module
    from osipy.dsc.deconvolution.base import BaseDeconvolver
    from osipy.dsc.deconvolution.registry import register_deconvolver
    from osipy.dsc.deconvolution.svd import DeconvolutionResult

    if TYPE_CHECKING:
        from numpy.typing import NDArray


    @register_deconvolver("my_method")
    class MyDeconvolver(BaseDeconvolver):
        """Custom deconvolution method."""

        @property
        def name(self) -> str:
            return "My Custom Deconvolution"

        def deconvolve(self, concentration, aif, time, mask=None, **kwargs):
            xp = get_array_module(concentration, aif, time)

            spatial_shape = concentration.shape[:-1]
            n_timepoints = concentration.shape[-1]
            dt = float(time[1] - time[0]) if len(time) > 1 else 1.0

            # Your deconvolution algorithm here
            # ...

            # Return a DeconvolutionResult
            return DeconvolutionResult(
                residue_function=residue,
                cbf=cbf,
                mtt=mtt,
                delay=delay,
                threshold_used=0.0,
            )
    ```

### 2. Use it

!!! example "Use the registered deconvolver"

    ```python
    # Import to trigger registration
    import osipy.dsc.deconvolution.my_deconv

    from osipy.dsc.deconvolution.registry import (
        get_deconvolver,
        list_deconvolvers,
    )

    # Verify registration
    print(list_deconvolvers())
    # ['cSVD', 'my_method', 'oSVD', 'sSVD']

    # Use it
    deconvolver = get_deconvolver("my_method")
    result = deconvolver.deconvolve(concentration, aif, time, mask)

    print(result.cbf)  # CBF map
    print(result.mtt)  # MTT map
    ```

## Adding a custom leakage correction method

The same pattern applies:

!!! example "Define a custom leakage corrector"

    ```python
    from osipy.dsc.leakage.base import BaseLeakageCorrector
    from osipy.dsc.leakage.registry import register_leakage_corrector


    @register_leakage_corrector("my_correction")
    class MyLeakageCorrector(BaseLeakageCorrector):
        @property
        def name(self) -> str:
            return "My Leakage Correction"

        def correct(self, delta_r2, aif, time, mask=None, **kwargs):
            # Your correction algorithm
            # Return a LeakageCorrectionResult
            ...
    ```

## Key conventions

- Use `xp = get_array_module(...)` for GPU/CPU agnostic code
- Return the standard result dataclasses (`DeconvolutionResult`, `LeakageCorrectionResult`)
- Use `DataValidationError` from `osipy.common.exceptions` for input validation errors
- Do not use scipy — use `xp.linalg` for linear algebra
