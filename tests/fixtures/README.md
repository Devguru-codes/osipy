# Test Data Sources for osipy

This document describes the data sources used for testing osipy MRI perfusion analysis pipelines.

## Overview

osipy uses a combination of:
1. **Synthetic Digital Reference Objects (DROs)** - Generated programmatically with known ground truth
2. **Public DRO datasets** - From OSIPI, QIBA, and TCIA
3. **Community challenge data** - From ISMRM and OSIPI challenges

## Data Sources by Modality

### DCE-MRI

#### OSIPI TF6.2 DCE Challenge Data
- **Source**: [OSIPI TF6.2 DCE-DSC-MRI Challenges](https://github.com/OSIPI/TF6.2_DCE-DSC-MRI_Challenges)
- **OSF Repository**: https://osf.io/u7a6f
- **Contains**: Synthetic DCE-MRI data with known Ktrans, ve, vp values
- **Format**: NIfTI
- **License**: Open access for research

#### QIBA DCE-MRI DRO
- **Source**: [Duke Barboriak Lab](https://sites.duke.edu/dblab/qibacontent/)
- **Contains**: QIBA-compliant DCE-MRI phantom data
- **Parameters**: Ktrans (0.01-0.35 min^-1), ve (0.1-0.5), vp (0.01-0.1)
- **License**: Public domain

#### RSNA QIDW
- **Source**: https://qidw.rsna.org
- **Contains**: Additional QIBA validation datasets

### DSC-MRI

#### QIN-BRAIN-DSC-MRI (TCIA)
- **Source**: https://www.cancerimagingarchive.net/collection/qin-brain-dsc-mri/
- **Contains**: Brain tumor DSC-MRI with clinical annotations
- **Format**: DICOM
- **License**: TCIA Data Usage Policy

#### GBM-DSC-MRI-DRO (TCIA)
- **Source**: https://www.cancerimagingarchive.net/collection/gbm-dsc-mri-dro/
- **Contains**: Glioblastoma DSC-MRI Digital Reference Objects
- **Parameters**: Ground truth CBV, CBF, MTT values
- **License**: TCIA Data Usage Policy

#### BRAIN-TUMOR-PROGRESSION (TCIA)
- **Source**: https://www.cancerimagingarchive.net/collection/brain-tumor-progression/
- **Contains**: Longitudinal brain tumor DSC-MRI
- **License**: TCIA Data Usage Policy

### ASL

#### OSIPI TF2.2 ASL Toolbox
- **Source**: [OSIPI TF2.2 ASL Toolbox](https://github.com/OSIPI/TF2.2_OSIPI-ASL-toolbox)
- **Contains**: ASL processing tools and reference data
- **License**: Open source

#### ISMRM-OSIPI ASL Challenge
- **Contains**: ASL DRO with known CBF and ATT values
- **Parameters**: CBF (0-100 mL/100g/min), ATT (0.5-2.5 s)

#### OpenNeuro ASL-BIDS
- **Source**: https://openneuro.org (search for ASL)
- **Contains**: ASL datasets in BIDS format
- **Format**: NIfTI + JSON sidecars

### IVIM

#### QIBA DWI Phantom
- **Source**: QIBA DWI committee
- **Contains**: Multi-b-value DWI phantom data
- **Parameters**: D, D*, f reference values

#### Synthetic IVIM DRO
- **Note**: Limited public IVIM DROs available
- **Solution**: osipy generates synthetic IVIM data using `create_synthetic_dro(modality="ivim")`

## Using Test Data in osipy

### Synthetic DRO Generation

osipy provides built-in synthetic DRO generation for all modalities:

```python
from osipy.common.validation import create_synthetic_dro

# Create DCE DRO
dce_dro = create_synthetic_dro(shape=(64, 64, 20), modality="dce", noise_level=0.01)

# Create DSC DRO
dsc_dro = create_synthetic_dro(shape=(64, 64, 20), modality="dsc", noise_level=0.02)

# Create ASL DRO
asl_dro = create_synthetic_dro(shape=(64, 64, 20), modality="asl", noise_level=0.05)

# Create IVIM DRO
ivim_dro = create_synthetic_dro(shape=(64, 64, 20), modality="ivim", noise_level=0.01)
```

### Loading External DROs

```python
from osipy.common.validation import load_dro

# Load from local path
dro = load_dro("/path/to/dro/directory")

# DRO contains:
# - dro.name: str
# - dro.parameters: dict[str, np.ndarray]
# - dro.mask: np.ndarray
# - dro.metadata: dict
```

## Directory Structure

```
tests/fixtures/
├── osipi_dro/          # OSIPI challenge DROs (downloaded)
│   ├── dce/
│   ├── dsc/
│   ├── asl/
│   └── ivim/
├── qiba/               # QIBA phantom data (downloaded)
│   ├── dce/
│   └── dwi/
└── synthetic/          # Generated synthetic data (created at runtime)
```

## Downloading Test Data

### Manual Download

1. **OSIPI DCE DRO**:
   ```bash
   # Clone OSIPI challenge repository
   git clone https://github.com/OSIPI/TF6.2_DCE-DSC-MRI_Challenges.git
   # Or download from OSF: https://osf.io/u7a6f
   ```

2. **TCIA DSC Data**:
   - Visit https://www.cancerimagingarchive.net
   - Search for collection (e.g., "QIN-BRAIN-DSC-MRI")
   - Use NBIA Data Retriever to download

3. **OpenNeuro ASL**:
   ```bash
   # Using datalad
   datalad install https://github.com/OpenNeuroDatasets/ds003474.git
   ```

### Automated Download (CI/Testing)

For CI pipelines, osipy uses synthetic DROs by default to avoid external dependencies.
The test fixtures automatically fall back to synthetic data when external DROs are not available.

## Ground Truth Parameters

### DCE-MRI Reference Values
| Parameter | Range | Units | Source |
|-----------|-------|-------|--------|
| Ktrans | 0.01 - 0.35 | min^-1 | QIBA Profile |
| ve | 0.1 - 0.5 | - | QIBA Profile |
| vp | 0.01 - 0.1 | - | QIBA Profile |

### DSC-MRI Reference Values
| Parameter | Range | Units | Source |
|-----------|-------|-------|--------|
| rCBV | 1.0 - 10.0 | - | Literature |
| CBF | 20 - 100 | mL/100g/min | Literature |
| MTT | 3 - 12 | s | Literature |

### ASL Reference Values
| Parameter | Range | Units | Source |
|-----------|-------|-------|--------|
| CBF | 20 - 80 | mL/100g/min | ISMRM Consensus |
| ATT | 0.5 - 2.5 | s | ISMRM Consensus |

### IVIM Reference Values
| Parameter | Range | Units | Source |
|-----------|-------|-------|--------|
| D | 0.5 - 2.0 | x10^-3 mm^2/s | Literature |
| D* | 5 - 20 | x10^-3 mm^2/s | Literature |
| f | 0.05 - 0.30 | - | Literature |

## Contributing Test Data

To contribute test data to osipy:

1. Ensure data is publicly available or has appropriate licensing
2. Provide ground truth values if available
3. Document the source and acquisition parameters
4. Submit a pull request with data loading code

## References

1. OSIPI: https://osipi.org
2. QIBA: https://qibawiki.rsna.org
3. TCIA: https://www.cancerimagingarchive.net
4. OpenNeuro: https://openneuro.org
