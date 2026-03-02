# How to Run a Pipeline from YAML (CLI)

Run osipy pipelines from the command line using YAML configuration files.

## Create a Config File

### Interactive wizard

The wizard walks you through every setting interactively and writes a
validated config file.  It adapts its questions based on your earlier
answers -- for example, it only asks for a T1 mapping method if you have
T1-weighted data, and only prompts for an AIF file path when you choose
manual AIF:

```bash
osipy --help-me-pls
```

### Template

Alternatively, dump a fully-commented template and edit it by hand:

```bash
osipy --dump-defaults dce > config.yaml
```

Available modalities: `dce`, `dsc`, `asl`, `ivim`.

## Run a Pipeline

Point the CLI at your config and data:

```bash
osipy config.yaml /path/to/data.nii.gz
```

Override the output directory:

```bash
osipy config.yaml /path/to/data.nii.gz -o results/
```

Enable verbose logging:

```bash
osipy config.yaml /path/to/data.nii.gz -v
```

## Validate a Config

Check a config file for errors without running the pipeline:

```bash
osipy --validate config.yaml
```

## YAML Config Structure

Every config has these top-level sections:

| Section    | Purpose                                        |
|------------|------------------------------------------------|
| `modality` | Which pipeline to run (`dce`, `dsc`, `asl`, `ivim`) |
| `pipeline` | Modality-specific parameters                    |
| `data`     | Input data paths and format settings            |
| `output`   | Output format and directory                     |
| `backend`  | GPU/CPU settings                                |
| `logging`  | Log level configuration                         |

### Common Fields

**data**:

- `format` -- Input format: `auto`, `nifti`, `dicom`, or `bids`
- `mask` -- Path to brain/tissue mask
- `t1_map` -- Pre-computed T1 map (DCE)
- `aif_file` -- Measured AIF file
- `m0_data` -- M0 calibration image (ASL)
- `b_values` -- b-value list (IVIM)
- `b_values_file` -- Path to b-values file (IVIM)

**output**:

- `format` -- Output format (`nifti`)

**backend**:

- `force_cpu` -- Force CPU execution even if GPU is available (`true`/`false`)

**logging**:

- `level` -- Log verbosity: `INFO`, `DEBUG`, or `WARNING`

## Modality Examples

### DCE-MRI

With T1 mapping from VFA data:

```yaml
modality: dce
pipeline:
  model: extended_tofts
  t1_mapping_method: vfa
  aif_source: population
  population_aif: parker
  acquisition:
    tr: 5.0
    flip_angles: [2, 5, 10, 15]
    baseline_frames: 5
    relaxivity: 4.5
data:
  mask: brain_mask.nii.gz
output:
  format: nifti
```

Without T1 data (assumed T1):

```yaml
modality: dce
pipeline:
  model: extended_tofts
  aif_source: population
  population_aif: parker
  acquisition:
    t1_assumed: 1400.0
    baseline_frames: 5
    relaxivity: 4.5
data:
  mask: brain_mask.nii.gz
output:
  format: nifti
```

### DSC-MRI

```yaml
modality: dsc
pipeline:
  te: 30.0
  deconvolution_method: oSVD
  apply_leakage_correction: true
  svd_threshold: 0.2
  baseline_frames: 10
data:
  mask: brain_mask.nii.gz
output:
  format: nifti
```

### ASL

```yaml
modality: asl
pipeline:
  labeling_scheme: pcasl
  pld: 1800.0
  label_duration: 1800.0
  t1_blood: 1650.0
  labeling_efficiency: 0.85
  m0_method: single
data:
  mask: brain_mask.nii.gz
  m0_data: m0.nii.gz
output:
  format: nifti
```

### IVIM

```yaml
modality: ivim
pipeline:
  fitting_method: segmented
  b_threshold: 200.0
  normalize_signal: true
data:
  mask: brain_mask.nii.gz
  b_values: [0, 10, 20, 50, 100, 200, 400, 800]
output:
  format: nifti
```

## See Also

- [YAML Configuration Reference](../reference/cli-config.md) — Complete field-by-field reference (auto-generated)
- [How to Run a Complete Pipeline](run-complete-pipeline.md) — Python API for programmatic control
- [Architecture](../explanation/architecture.md) — Design context and pipeline internals
