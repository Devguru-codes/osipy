# How to Load Perfusion Data

Load perfusion MRI data from different vendors and formats.

## Quick Start

Auto-detect and load perfusion data:

```python
from osipy.common.io import load_perfusion
from osipy.common.types import Modality

# Auto-detect format (NIfTI, DICOM, or BIDS)
dataset = load_perfusion("path/to/data", modality=Modality.DCE)

# Explicit format
dataset = load_perfusion("path/to/bids", format="bids", subject="01", modality=Modality.ASL)
```

## Which Function to Use

`load_perfusion()` auto-detects your file format and delegates to the right loader. Use this by default.

The format-specific functions (`load_nifti()`, `load_dicom()`, `load_bids()`) are also exported for when you already know your format or need features that don't fit the generic interface — for example, `load_bids_with_m0()` returns the ASL data and M0 calibration image together, and `load_asl_context()` reads the control/label ordering from BIDS sidecars.

| Format | Extension | Generic | Direct | Notes |
|--------|-----------|---------|--------|-------|
| NIfTI | `.nii`, `.nii.gz` | `load_perfusion()` | `load_nifti()` | Optionally with JSON sidecar |
| DICOM | `.dcm`, `.IMA` | `load_perfusion()` | `load_dicom()` | Vendor metadata extracted |
| BIDS | Directory | `load_perfusion()` | `load_bids()`, `load_bids_with_m0()`, `load_asl_context()` | Full metadata from sidecars |

## Loading by Modality

### DCE-MRI

Load DCE-MRI data from NIfTI or DICOM:

```python
from osipy.common.io import load_perfusion
from osipy.common.types import Modality

# From NIfTI with sidecar
dataset = load_perfusion(
    "data/sub-01_dce.nii.gz",
    modality=Modality.DCE,
    sidecar_json="data/sub-01_dce.json",  # Optional
)

# From DICOM
dataset = load_perfusion(
    "data/dicom/dce_series/",
    modality=Modality.DCE,
    format="dicom",
)

print(f"Shape: {dataset.shape}")  # (x, y, z, t)
print(f"TR: {dataset.acquisition_params.tr} ms")
print(f"Flip angle: {dataset.acquisition_params.flip_angles}")
```

**Required Metadata:**
- TR (RepetitionTime)
- Flip angle (FlipAngle)

**Vendor Differences:**
- **GE**: TR in `RepetitionTime` tag (0018,0080)
- **Siemens**: TR in standard tag, may also have CSA header
- **Philips**: TR may be in private tag (2005,1030) for enhanced DICOM

### DSC-MRI

Load DSC-MRI data:

```python
from osipy.common.io import load_perfusion
from osipy.common.types import Modality

dataset = load_perfusion(
    "data/dsc_data.nii.gz",
    modality=Modality.DSC,
)

print(f"TE: {dataset.acquisition_params.te} ms")
```

**Required Metadata:**
- TE (EchoTime)
- TR (RepetitionTime)

### ASL

Load ASL data with M0 calibration from BIDS:

```python
from osipy.common.io import load_bids, load_bids_with_m0, load_asl_context
from osipy.common.types import Modality

# Load ASL with M0 calibration
asl_data, m0_data = load_bids_with_m0(
    "data/bids_dataset/",
    subject="01",
)

# Load ASL context (control/label order)
context = load_asl_context("data/bids_dataset/", subject="01")
print(f"Volume types: {set(context)}")  # {'control', 'label'}

# Access acquisition parameters
params = asl_data.acquisition_params
print(f"Labeling type: {params.labeling_type}")  # PCASL, PASL, etc.
print(f"PLD: {params.pld} ms")
print(f"Labeling duration: {params.labeling_duration} ms")
```

**Required Metadata:**
- ArterialSpinLabelingType (PCASL, PASL, CASL)
- PostLabelingDelay
- LabelingDuration (for pCASL/CASL)

**ASL-BIDS Files:**
- `*_asl.nii.gz` - ASL timeseries
- `*_asl.json` - Sidecar with parameters
- `*_aslcontext.tsv` - Volume types (control, label, m0scan, deltam)
- `*_m0scan.nii.gz` - Separate M0 calibration (optional)

### IVIM

Load IVIM data:

```python
from osipy.common.io import load_perfusion
from osipy.common.types import Modality

dataset = load_perfusion(
    "data/dwi_ivim.nii.gz",
    modality=Modality.IVIM,
)

print(f"B-values: {dataset.acquisition_params.b_values}")
```

**Required Metadata:**
- B-values (DiffusionBValue or bValues)

## Vendor-Specific Considerations

### GE Medical Systems

**DICOM Characteristics:**
- Manufacturer tag: "GE MEDICAL SYSTEMS"
- Private groups: 0027, 0043
- B-values: Often in private tag (0043,1039)
- ASL: 3D spiral pCASL product sequence common

Extract GE vendor metadata:

```python
import pydicom
from osipy.common.io.vendors import detect_vendor
from osipy.common.io.vendors.detection import extract_vendor_metadata

# Automatic vendor detection
ds = pydicom.dcmread("ge_dicom.dcm")
vendor = detect_vendor(ds)  # Returns "GE"

# Extract vendor-specific metadata (recommended one-step approach)
metadata = extract_vendor_metadata(ds)
print(f"Vendor: {metadata.vendor}")
print(f"TR: {metadata.tr} ms")
```

### Siemens

**DICOM Characteristics:**
- Manufacturer tag: "SIEMENS"
- Private groups: 0019, 0021, 0051
- CSA headers contain extended sequence information
- B-values: CSA header or private tag (0019,100c)
- ASL: WIP sequences common, PASL and pCASL available

**CSA Header Extraction:**
The Siemens parser automatically extracts:
- MrPhoenixProtocol parameters
- Sequence timing information
- Diffusion encoding details

### Philips

**DICOM Characteristics:**
- Manufacturer tag: "Philips Medical Systems" or "Philips"
- Private groups: 2001, 2005
- Enhanced DICOM format common
- B-values: Private tag (2001,1003)
- ASL: 2D EPI pCASL product sequence

**Important Notes:**
- Philips DICOM headers often lack complete ASL parameters
- Document sequence settings at acquisition time
- Control/label order may need manual specification

## Data Structures

### PerfusionDataset

All loading functions return a `PerfusionDataset`:

```python
@dataclass
class PerfusionDataset:
    data: NDArray[np.floating]      # Image data (3D or 4D)
    affine: NDArray[np.floating]    # 4x4 voxel-to-world transform
    modality: Modality              # DCE, DSC, ASL, or IVIM
    time_points: NDArray | None     # Time points for 4D data
    acquisition_params: AcquisitionParams  # Modality-specific params
    source_path: Path               # Original file path
    source_format: str              # "nifti", "dicom", or "bids"
```

### Acquisition Parameters

Each modality has specific acquisition parameters:

```python
# DCE
@dataclass
class DCEAcquisitionParams:
    tr: float                  # Repetition time (ms)
    te: float | None          # Echo time (ms)
    flip_angles: list[float]  # Flip angles (degrees)
    temporal_resolution: float | None

# ASL
@dataclass
class ASLAcquisitionParams:
    labeling_type: LabelingType  # PCASL, PASL_FAIR, etc.
    pld: float | list[float]     # Post-labeling delay (ms)
    labeling_duration: float     # Labeling duration (ms)
    background_suppression: bool
    m0_scale: float | None        # M0 scaling factor

# IVIM
@dataclass
class IVIMAcquisitionParams:
    b_values: NDArray[np.floating]  # B-values (s/mm²)
    tr: float | None
    te: float | None
```

## Metadata Priority

When loading data, metadata is resolved in this priority order:

1. **BIDS sidecar JSON** (highest priority)
2. **Vendor-specific DICOM tags**
3. **Standard DICOM tags**
4. **User-provided values**
5. **Default values** (lowest priority)

## Interactive Mode

By default, loading functions prompt for missing required parameters. To disable prompting:

```python
# Disable prompting (use defaults or raise error)
dataset = load_perfusion(
    "data.nii.gz",
    modality=Modality.ASL,
    interactive=False,  # Don't prompt for missing params
)
```

## Loading DICOM via the CLI

The `osipy` CLI supports DICOM input directly. Set `data.format` to `dicom`
(or `auto` for automatic detection):

YAML config for DICOM input:

```yaml
modality: dce
data:
  format: auto          # auto-detects DICOM directories
pipeline:
  model: extended_tofts
  aif_source: population
  population_aif: parker
  acquisition:
    tr: 5.0
    flip_angles: [2, 5, 10, 15]
```

Run CLI with DICOM directories:

```bash
# Single-series directory
osipy config.yaml /path/to/dicom_series/

# Multi-series directory (e.g., QIN-Breast with MR_* subdirs)
osipy config.yaml /path/to/study_dir/
```

Multi-series layouts are detected automatically. The loader delegates to
`load_dicom_multi_series()`, which auto-orders timepoints from DICOM metadata.

## BIDS Batch Processing

Process multiple subjects from a BIDS dataset:

```python
from bids import BIDSLayout
import osipy

layout = BIDSLayout("path/to/bids")

for subject in layout.get_subjects():
    dce_files = layout.get(subject=subject, suffix="dce", extension="nii.gz")
    if not dce_files:
        continue
    data = osipy.load_nifti(dce_files[0].path)
    metadata = dce_files[0].get_metadata()
    # result = osipy.fit_model("extended_tofts", ...)
```

### Multi-PLD ASL from BIDS

Load multi-PLD ASL from BIDS:

```python
from bids import BIDSLayout
import numpy as np

layout = BIDSLayout("path/to/bids")
asl_file = layout.get(subject="01", suffix="asl", extension="nii.gz")[0]
metadata = asl_file.get_metadata()

plds = metadata.get('PostLabelingDelay')  # May be array
if isinstance(plds, list):
    plds = np.array(plds)
```

### Common BIDS Fields for Perfusion

| Field | Description | Modality |
|-------|-------------|----------|
| `RepetitionTime` | TR in seconds | All |
| `EchoTime` | TE in seconds | All |
| `FlipAngle` | Flip angle in degrees | DCE |
| `ArterialSpinLabelingType` | PASL, CASL, PCASL | ASL |
| `LabelingDuration` | Label duration in seconds | ASL |
| `PostLabelingDelay` | PLD in seconds | ASL |
| `M0Type` | Included, Separate, Estimate | ASL |
| `DiffusionBValue` | b-values in s/mm^2 | IVIM |

## Troubleshooting

### Common Issues

**"Data directory not found"**
- Check that subject/session directories exist
- osipy checks both `sub-XX/perf/` and `sub-XX/` for ASL data

**"4D data requires time_points array"**
- Ensure TR is specified in sidecar JSON
- Provide `RepetitionTime` or `RepetitionTimePreparation` in metadata

**"No vendor parser available"**
- DICOM may be from unsupported vendor
- Standard DICOM tags will still be extracted

**DICOM loads as 3D instead of 4D**
- Known limitation: DICOM loader doesn't yet sort temporal dimension
- Workaround: Use dcm2niix to convert to NIfTI first

## See Also

- [Vendor DICOM Tag Reference](../reference/vendor-dicom-tags.md)
- [BIDS Specification](https://bids-specification.readthedocs.io/)
