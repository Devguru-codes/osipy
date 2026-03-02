# The xp Abstraction Pattern

osipy runs the same numerical code on CPU (NumPy) and GPU (CuPy) using the `xp = get_array_module()` pattern.

## The Problem

MRI perfusion analysis involves heavy numerical computation:

- Thousands to millions of voxels
- Iterative fitting algorithms
- Matrix operations and convolutions

GPUs can accelerate this, but the challenge is:

- **NumPy** works on CPU, has vast ecosystem
- **CuPy** works on GPU, mirrors NumPy's API
- Maintaining two codebases is impractical

## The Solution: xp Abstraction

### Core Idea

Instead of writing:

!!! example "CPU-only approach with numpy"

    ```python
    import numpy as np
    result = np.exp(-data)  # CPU only
    ```

Or:

!!! example "GPU-only approach with cupy"

    ```python
    import cupy as cp
    result = cp.exp(-data)  # GPU only
    ```

The xp pattern offers a third option:

!!! example "Backend-agnostic approach with xp"

    ```python
    from osipy.common.backend.array_module import get_array_module

    def my_function(data):
        xp = get_array_module(data)  # Returns numpy or cupy
        result = xp.exp(-data)       # Works on both!
        return result
    ```

### How It Works

`get_array_module()` inspects the input arrays:

!!! example "How get_array_module detects the backend"

    ```python
    def get_array_module(*arrays):
        """Return numpy or cupy depending on input array type."""
        for arr in arrays:
            if hasattr(arr, '__cuda_array_interface__'):
                import cupy
                return cupy
        return numpy
    ```

If any input is a CuPy array (GPU), it returns CuPy. Otherwise, NumPy.

## The Pattern in Practice

### Required Structure

Every numerical function in osipy follows this pattern:

!!! example "Standard xp pattern for numerical functions"

    ```python
    from osipy.common.backend.array_module import get_array_module

    def fit_model(data, aif, time):
        """Fit pharmacokinetic model.

        Works transparently on CPU (numpy) or GPU (cupy).
        """
        # FIRST LINE: Get the array module
        xp = get_array_module(data, aif, time)

        # ALL array operations use xp:
        n_voxels = data.shape[0]
        result = xp.zeros(n_voxels)

        # Mathematical operations
        exp_decay = xp.exp(-time / tau)
        convolved = xp.convolve(aif, exp_decay)

        # Linear algebra
        solution = xp.linalg.lstsq(A, b)

        # Reductions
        mean_val = xp.mean(result)

        return result
    ```

### Common xp Replacements

| NumPy | xp Equivalent |
|-------|---------------|
| `np.array()` | `xp.array()` |
| `np.zeros()` | `xp.zeros()` |
| `np.ones()` | `xp.ones()` |
| `np.exp()` | `xp.exp()` |
| `np.log()` | `xp.log()` |
| `np.sqrt()` | `xp.sqrt()` |
| `np.sum()` | `xp.sum()` |
| `np.mean()` | `xp.mean()` |
| `np.std()` | `xp.std()` |
| `np.where()` | `xp.where()` |
| `np.clip()` | `xp.clip()` |
| `np.abs()` | `xp.abs()` |
| `np.linalg.*` | `xp.linalg.*` |
| `np.fft.*` | `xp.fft.*` |
| `np.concatenate()` | `xp.concatenate()` |
| `np.stack()` | `xp.stack()` |

## When to Convert

### Converting to CPU

Some operations require CPU arrays:

!!! example "Convert GPU arrays to CPU for I/O and plotting"

    ```python
    from osipy.common.backend import to_numpy

    def save_result(data, filename):
        """Save data to file (requires numpy)."""
        data_cpu = to_numpy(data)  # Ensure numpy array
        np.save(filename, data_cpu)

    def plot_data(data):
        """Matplotlib requires numpy."""
        import matplotlib.pyplot as plt
        plt.plot(to_numpy(data))
    ```

### Converting to GPU

To move data to GPU:

!!! example "Explicitly move data to GPU"

    ```python
    from osipy.common.backend import to_gpu

    def process_on_gpu(data):
        """Explicitly move to GPU."""
        if is_gpu_available():
            data_gpu = to_gpu(data)
            ...
    ```

## Exceptions to the Pattern

### Type Hints

NumPy typing is OK (no runtime operations):

!!! example "Type hints with numpy.typing"

    ```python
    from numpy.typing import NDArray
    import numpy as np

    def my_function(data: NDArray[np.float64]) -> NDArray[np.float64]:
        xp = get_array_module(data)
        # ...
    ```

### Mathematical Constants

Constants like `np.pi` are fine:

!!! example "Using numpy constants with xp pattern"

    ```python
    def compute_phase(angle):
        xp = get_array_module(angle)
        # np.pi is a constant, not an operation
        return xp.sin(angle * np.pi / 180)
    ```

### I/O Operations

File I/O always uses numpy (via nibabel, pydicom):

!!! example "Load, process on GPU, and save with nibabel"

    ```python
    import nibabel as nib

    def load_and_process(filename):
        # Load with nibabel (returns numpy)
        img = nib.load(filename)
        data = img.get_fdata()  # numpy array

        # Convert to GPU if available
        data_gpu = to_gpu(data) if is_gpu_available() else data

        # Process with xp pattern
        result = process(data_gpu)

        # Convert back for saving
        result_cpu = to_numpy(result)
        nib.save(nib.Nifti1Image(result_cpu, img.affine), "output.nii.gz")
    ```

## Why This Matters

### Performance Impact

GPU acceleration varies by operation:

| Operation | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| 300k voxel fitting | 45s | 3s | 15x |
| FFT convolution | 2.1s | 0.1s | 21x |
| Matrix operations | 1.5s | 0.08s | 19x |

*Benchmarks measured with specific dataset sizes and hardware. Actual speedups depend on data size, GPU model, and which code path is used. Only xp-compliant code paths benefit from GPU acceleration.*

### Code Maintainability

One codebase serves both:

- **Researchers**: Can run on laptops (CPU)
- **Production**: Can scale with GPUs
- **No divergence**: Same algorithm, verified on both

## Common Mistakes

### Forgetting xp for New Operations

!!! example "Wrong vs correct use of xp for new operations"

    ```python
    # WRONG: Added np.maximum without xp
    def clip_positive(data):
        xp = get_array_module(data)
        result = np.maximum(data, 0)  # BUG: uses np directly
        return result

    # CORRECT:
    def clip_positive(data):
        xp = get_array_module(data)
        result = xp.maximum(data, 0)  # Uses xp
        return result
    ```

### Mixing Array Types

!!! example "Wrong vs correct array creation in xp functions"

    ```python
    # WRONG: Creates numpy array in xp function
    def bad_function(data):
        xp = get_array_module(data)
        mask = np.zeros(data.shape, dtype=bool)  # numpy on GPU data!
        result = xp.where(mask, data, 0)  # Type mismatch
        return result

    # CORRECT:
    def good_function(data):
        xp = get_array_module(data)
        mask = xp.zeros(data.shape, dtype=bool)  # Same backend
        result = xp.where(mask, data, 0)
        return result
    ```

### Using scipy

!!! example "Wrong vs correct approach for optimization"

    ```python
    # WRONG: scipy doesn't work with CuPy
    from scipy.optimize import minimize

    def bad_fit(data):
        xp = get_array_module(data)
        result = minimize(objective, x0)  # scipy can't use GPU arrays

    # CORRECT: Use osipy's Levenberg-Marquardt fitter
    from osipy.common.fitting import LevenbergMarquardtFitter

    def good_fit(data, bound_model, mask):
        fitter = LevenbergMarquardtFitter()
        result = fitter.fit_image(bound_model, data, mask)  # xp-compatible
    ```

## Testing xp Code

### Test on Both Backends

!!! example "Separate CPU and GPU tests"

    ```python
    import pytest
    import numpy as np

    def test_my_function_cpu():
        data = np.array([1.0, 2.0, 3.0])
        result = my_function(data)
        assert isinstance(result, np.ndarray)

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="No GPU")
    def test_my_function_gpu():
        import cupy as cp
        data = cp.array([1.0, 2.0, 3.0])
        result = my_function(data)
        assert isinstance(result, cp.ndarray)
    ```

### Parametrize for Both

!!! example "Parametrized fixture for both backends"

    ```python
    @pytest.fixture(params=['numpy', 'cupy'])
    def xp(request):
        if request.param == 'cupy':
            pytest.importorskip('cupy')
            import cupy
            return cupy
        return numpy

    def test_my_function(xp):
        data = xp.array([1.0, 2.0, 3.0])
        result = my_function(data)
        # Works for both backends
    ```

## Design Principles

1. **Input determines backend**: First array argument sets the context
2. **All operations through xp**: No direct numpy/cupy calls in core code
3. **Explicit conversion**: Use `to_numpy()`/`to_gpu()` at boundaries
4. **Test both paths**: CI runs on CPU, GPU tests where available

## See Also

- [Architecture Overview](architecture.md)
- [How to Enable GPU Acceleration](../how-to/enable-gpu-acceleration.md)
