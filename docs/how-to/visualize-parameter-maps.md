# How to Visualize Parameter Maps

Create visualizations of perfusion parameter maps.

## Basic Parameter Map

Display a single parameter map:

!!! example "Display a single Ktrans map"

    ```python
    import matplotlib.pyplot as plt
    import osipy

    # Load or compute parameter map
    ktrans = result.parameter_maps['Ktrans'].values

    # Select slice
    slice_idx = ktrans.shape[2] // 2

    # Display
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(ktrans[:, :, slice_idx], cmap='hot', vmin=0, vmax=0.5)
    ax.set_title('Ktrans (min⁻¹)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.savefig('ktrans_map.png', dpi=300, bbox_inches='tight')
    plt.show()
    ```

## Multi-Parameter Display

Show multiple parameters together:

!!! example "Display multiple parameter maps in a grid"

    ```python
    import matplotlib.pyplot as plt

    # Parameters to display
    params = {
        'Ktrans': (result.parameter_maps['Ktrans'].values, 'hot', 0, 0.5, 'min⁻¹'),
        've': (result.parameter_maps['ve'].values, 'viridis', 0, 1, 'fraction'),
        'vp': (result.parameter_maps['vp'].values, 'plasma', 0, 0.2, 'fraction'),
        'R²': (result.r_squared_map, 'gray', 0, 1, ''),
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    slice_idx = result.parameter_maps['Ktrans'].values.shape[2] // 2

    for ax, (name, (data, cmap, vmin, vmax, units)) in zip(axes.flat, params.items()):
        im = ax.imshow(data[:, :, slice_idx], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f'{name} ({units})' if units else name)
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig('parameter_maps.png', dpi=300, bbox_inches='tight')
    plt.show()
    ```

## Slice Montage

Display all slices in a grid:

!!! example "Create a slice montage"

    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    def plot_montage(data, title, cmap='hot', vmin=None, vmax=None, n_cols=6):
        """Create a slice montage."""
        n_slices = data.shape[2]
        n_rows = int(np.ceil(n_slices / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
        axes = axes.flatten()

        for i in range(n_slices):
            axes[i].imshow(data[:, :, i], cmap=cmap, vmin=vmin, vmax=vmax)
            axes[i].axis('off')
            axes[i].set_title(f'z={i}', fontsize=8)

        # Hide unused axes
        for i in range(n_slices, len(axes)):
            axes[i].axis('off')

        fig.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()
        return fig

    # Create montage
    fig = plot_montage(result.parameter_maps['Ktrans'].values, 'Ktrans (min⁻¹)', vmin=0, vmax=0.5)
    plt.savefig('ktrans_montage.png', dpi=200, bbox_inches='tight')
    ```

## Overlay on Anatomy

Overlay parameter map on anatomical image:

!!! example "Overlay parameter map on anatomy"

    ```python
    import matplotlib.pyplot as plt
    import numpy as np

    def overlay_map(anatomy, parameter, mask=None, slice_idx=None,
                    cmap='hot', alpha=0.7, vmin=None, vmax=None):
        """Overlay parameter map on anatomy."""
        if slice_idx is None:
            slice_idx = anatomy.shape[2] // 2

        fig, ax = plt.subplots(figsize=(8, 8))

        # Show anatomy
        ax.imshow(anatomy[:, :, slice_idx], cmap='gray')

        # Overlay parameter (masked)
        param_slice = parameter[:, :, slice_idx].copy()
        if mask is not None:
            param_slice = np.ma.masked_where(~mask[:, :, slice_idx], param_slice)

        im = ax.imshow(param_slice, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.axis('off')

        return fig

    # Use
    anatomy = dce_data.mean(axis=-1)  # Mean DCE signal as anatomy
    fig = overlay_map(anatomy, result.parameter_maps['Ktrans'].values, mask=result.quality_mask > 0,
                      vmin=0, vmax=0.5)
    plt.savefig('ktrans_overlay.png', dpi=300, bbox_inches='tight')
    ```

## Histogram Analysis

Plot parameter distributions:

!!! example "Plot parameter histograms"

    ```python
    import matplotlib.pyplot as plt
    import numpy as np

    def plot_histograms(result, mask):
        """Plot histograms of all parameters."""
        params = ['Ktrans', 've', 'vp']

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        for ax, param in zip(axes, params):
            values = result.parameter_maps[param].values[mask > 0]

            ax.hist(values, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
            ax.axvline(values.mean(), color='red', linestyle='--',
                       label=f'Mean: {values.mean():.3f}')
            ax.axvline(np.median(values), color='green', linestyle=':',
                       label=f'Median: {np.median(values):.3f}')

            ax.set_xlabel(param)
            ax.set_ylabel('Voxel Count')
            ax.legend(fontsize=8)

        plt.tight_layout()
        return fig

    fig = plot_histograms(result, result.quality_mask)
    plt.savefig('parameter_histograms.png', dpi=150)
    ```

## Time Course Plots

Visualize concentration time courses:

!!! example "Plot AIF and sample tissue time courses"

    ```python
    import matplotlib.pyplot as plt
    import numpy as np

    def plot_time_courses(concentration, time, aif, mask, n_samples=5):
        """Plot AIF and sample tissue curves."""
        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot AIF
        ax.plot(time, aif.concentration, 'r-', linewidth=2, label='AIF')

        # Sample tissue curves
        valid_indices = np.where(mask > 0)
        n_valid = len(valid_indices[0])
        sample_idx = np.linspace(0, n_valid-1, n_samples, dtype=int)

        for i, idx in enumerate(sample_idx):
            x, y, z = valid_indices[0][idx], valid_indices[1][idx], valid_indices[2][idx]
            ax.plot(time, concentration[x, y, z, :], alpha=0.6, label=f'Tissue {i+1}')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Concentration (mM)')
        ax.set_title('Concentration Time Curves')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        return fig

    fig = plot_time_courses(concentration, time, aif, result.quality_mask)
    plt.savefig('time_courses.png', dpi=150)
    ```

## Quality Mask Visualization

Show which voxels passed quality control:

!!! example "Visualize quality mask and R-squared distribution"

    ```python
    import matplotlib.pyplot as plt

    def plot_quality_mask(quality_mask, r_squared, slice_idx=None):
        """Visualize quality mask and R² distribution."""
        if slice_idx is None:
            slice_idx = quality_mask.shape[2] // 2

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Quality mask
        axes[0].imshow(quality_mask[:, :, slice_idx], cmap='binary')
        axes[0].set_title('Quality Mask')
        axes[0].axis('off')

        # R² map
        im = axes[1].imshow(r_squared[:, :, slice_idx], cmap='viridis', vmin=0, vmax=1)
        axes[1].set_title('R² Map')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], shrink=0.8)

        # R² histogram
        valid_r2 = r_squared[quality_mask > 0]
        axes[2].hist(valid_r2, bins=50, color='steelblue', edgecolor='white')
        axes[2].axvline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
        axes[2].set_xlabel('R²')
        axes[2].set_ylabel('Count')
        axes[2].set_title(f'R² Distribution\n(n={len(valid_r2)} voxels)')
        axes[2].legend()

        plt.tight_layout()
        return fig

    fig = plot_quality_mask(result.quality_mask, result.r_squared_map)
    plt.savefig('quality_metrics.png', dpi=150)
    ```

## Publication-Ready Figure

Create a complete figure for publication:

!!! example "Create publication-ready multi-panel figure"

    ```python
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    def create_publication_figure(result, concentration, time, aif, anatomy):
        """Create publication-ready figure."""
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        slice_idx = result.parameter_maps['Ktrans'].values.shape[2] // 2
        mask = result.quality_mask > 0

        # Row 1: Parameter maps
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(result.parameter_maps['Ktrans'].values[:, :, slice_idx], cmap='hot', vmin=0, vmax=0.5)
        ax1.set_title('Ktrans (min⁻¹)')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, shrink=0.6)

        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(result.parameter_maps['ve'].values[:, :, slice_idx], cmap='viridis', vmin=0, vmax=1)
        ax2.set_title('ve')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, shrink=0.6)

        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(result.parameter_maps['vp'].values[:, :, slice_idx], cmap='plasma', vmin=0, vmax=0.2)
        ax3.set_title('vp')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, shrink=0.6)

        # Row 2: Overlay and quality
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.imshow(anatomy[:, :, slice_idx], cmap='gray')
        ax4.set_title('Anatomy')
        ax4.axis('off')

        ax5 = fig.add_subplot(gs[1, 1])
        ax5.imshow(anatomy[:, :, slice_idx], cmap='gray')
        ktrans_masked = np.ma.masked_where(~mask[:, :, slice_idx], result.parameter_maps['Ktrans'].values[:, :, slice_idx])
        im5 = ax5.imshow(ktrans_masked, cmap='hot', alpha=0.7, vmin=0, vmax=0.5)
        ax5.set_title('Ktrans Overlay')
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5, shrink=0.6)

        ax6 = fig.add_subplot(gs[1, 2])
        im6 = ax6.imshow(result.r_squared_map[:, :, slice_idx], cmap='gray', vmin=0, vmax=1)
        ax6.set_title('R²')
        ax6.axis('off')
        plt.colorbar(im6, ax=ax6, shrink=0.6)

        # Row 3: Time courses and histograms
        ax7 = fig.add_subplot(gs[2, :2])
        ax7.plot(time, aif.concentration, 'r-', linewidth=2, label='AIF')
        sample_idx = np.where(mask)
        for i in range(3):
            idx = i * len(sample_idx[0]) // 4
            x, y, z = sample_idx[0][idx], sample_idx[1][idx], sample_idx[2][idx]
            ax7.plot(time, concentration[x, y, z, :], alpha=0.6)
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Concentration (mM)')
        ax7.set_title('Time Curves')
        ax7.grid(True, alpha=0.3)

        ax8 = fig.add_subplot(gs[2, 2])
        ax8.hist(result.parameter_maps['Ktrans'].values[mask], bins=30, color='steelblue', alpha=0.8)
        ax8.set_xlabel('Ktrans (min⁻¹)')
        ax8.set_ylabel('Count')
        ax8.set_title('Ktrans Distribution')

        plt.suptitle('DCE-MRI Analysis Results', fontsize=14, y=1.02)
        return fig

    fig = create_publication_figure(result, concentration, time, aif, dce_data.mean(axis=-1))
    plt.savefig('publication_figure.png', dpi=300, bbox_inches='tight')
    ```

## Colormaps for Perfusion

Recommended colormaps:

| Parameter | Colormap | Range |
|-----------|----------|-------|
| Ktrans | `hot` | 0-0.5 min⁻¹ |
| ve | `viridis` | 0-1 |
| vp | `plasma` | 0-0.2 |
| CBV | `hot` | 0-10 ml/100g |
| CBF | `jet` | 0-100 ml/100g/min |
| MTT | `viridis` | 0-10 s |
| R² | `gray` | 0-1 |

## See Also

- [DCE-MRI Tutorial](../tutorials/dce-analysis.md)
- [How to Export BIDS Results](export-bids.md)
