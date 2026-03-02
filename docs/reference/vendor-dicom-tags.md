# Vendor DICOM Tag Reference

DICOM tags used by each MRI vendor for perfusion imaging parameters. These differences affect how osipy extracts metadata during data loading.

## Standard DICOM Tags (All Vendors)

| Tag | Name | Description |
|-----|------|-------------|
| (0008,0070) | Manufacturer | Vendor identification |
| (0008,1090) | ManufacturerModelName | Scanner model |
| (0018,0080) | RepetitionTime | TR in ms |
| (0018,0081) | EchoTime | TE in ms |
| (0018,1314) | FlipAngle | Flip angle in degrees |
| (0018,9087) | DiffusionBValue | B-value for DWI |
| (0020,0013) | InstanceNumber | Volume/slice number |
| (0020,1041) | SliceLocation | Slice position |

## GE Medical Systems

### Identification
- **Manufacturer**: "GE MEDICAL SYSTEMS"
- **Private Groups**: 0027, 0043, 0019

### DCE-MRI Tags
| Tag | Name | Notes |
|-----|------|-------|
| (0018,0080) | RepetitionTime | Standard TR |
| (0018,0081) | EchoTime | Standard TE |
| (0018,1314) | FlipAngle | Standard |
| (0019,10a2) | NumberOfPhases | Number of dynamic timepoints |

### DWI/IVIM Tags
| Tag | Name | Notes |
|-----|------|-------|
| (0043,1039) | B-value | GE private b-value array |
| (0019,10bb) | DiffusionBValue | Alternative b-value location |
| (0019,10bc) | DiffusionDirection | Gradient direction |

### ASL Tags
GE ASL data is typically:
- 3D spiral pCASL (product sequence)
- Parameters in BIDS sidecar after conversion
- `M0Type: "Included"` common (M0 in same timeseries)

**Known Issues:**
- B-values may not populate standard DiffusionBValue tag
- Must check private tags for complete diffusion info
- ASL parameters often require BIDS sidecar

## Siemens

### Identification
- **Manufacturer**: "SIEMENS"
- **Private Groups**: 0019, 0021, 0029, 0051

### CSA Headers
Siemens stores extended metadata in CSA (Cine Supplementary Attributes) headers:
- `(0029,1010)` CSA Image Header Info
- `(0029,1020)` CSA Series Header Info

The osipy Siemens parser extracts:

!!! example "Extract metadata using Siemens parser"

    ```python
    from osipy.common.io.vendors import SiemensParser

    parser = SiemensParser()
    metadata = parser.extract_metadata(ds)
    # Accesses CSA headers for extended parameters
    ```

### DCE-MRI Tags
| Tag | Name | Notes |
|-----|------|-------|
| (0018,0080) | RepetitionTime | Standard TR |
| (0018,0081) | EchoTime | Standard TE |
| (0018,1314) | FlipAngle | Standard |
| (0051,100c) | NumberOfImages | May indicate timepoints |
| CSA Header | MrPhoenixProtocol | Detailed sequence parameters |

### DWI/IVIM Tags
| Tag | Name | Notes |
|-----|------|-------|
| (0019,100c) | DiffusionBValue | Siemens private b-value |
| (0019,100d) | DiffusionDirection | Gradient direction (x) |
| (0019,100e) | DiffusionDirection | Gradient direction (y) |
| (0019,100f) | DiffusionDirection | Gradient direction (z) |
| CSA Header | B_value | B-value from CSA |

### ASL Tags
Siemens ASL sequences include:
- **WIP pCASL**: 3D GRASE readout
- **PASL/FAIR**: Bremen sequence (multi-PLD)
- **2D EPI**: Research sequences

| Parameter | Source | Notes |
|-----------|--------|-------|
| ArterialSpinLabelingType | BIDS sidecar | PASL, PCASL |
| PostLabelingDelay | BIDS sidecar | May be array for multi-PLD |
| BolusCutOffFlag | BIDS sidecar | Q2TIPS for PASL |
| BolusCutOffDelayTime | BIDS sidecar | Q2TIPS timing |
| BolusCutOffTechnique | BIDS sidecar | "Q2TIPS" |

**Known Issues:**
- Multi-PLD data has array of PLDs (not single value)
- CSA header parsing may fail on some sequences
- Research sequences may have non-standard parameters

## Philips

### Identification
- **Manufacturer**: "Philips Medical Systems" or "Philips"
- **Private Groups**: 2001, 2005

### Enhanced DICOM
Philips commonly uses Enhanced Multi-frame DICOM:
- Single file contains all slices/timepoints
- Nested sequences for per-frame metadata
- Use dcm2niix for reliable conversion

### DCE-MRI Tags
| Tag | Name | Notes |
|-----|------|-------|
| (0018,0080) | RepetitionTime | Standard TR |
| (0018,0081) | EchoTime | Standard TE |
| (0018,1314) | FlipAngle | Standard |
| (2005,1030) | PrivateRepetitionTime | May differ from standard |
| (2001,1023) | NumberOfDynamics | Number of timepoints |

### DWI/IVIM Tags
| Tag | Name | Notes |
|-----|------|-------|
| (2001,1003) | DiffusionBValue | Philips private b-value |
| (2001,1004) | DiffusionDirection | Gradient direction |
| (0018,9087) | DiffusionBValue | May also be populated |

### ASL Tags
Philips ASL typically uses 2D EPI pCASL:

| Parameter | Source | Notes |
|-----------|--------|-------|
| ArterialSpinLabelingType | BIDS sidecar | Usually "PCASL" |
| PostLabelingDelay | BIDS sidecar | Single value typically |
| LabelingDuration | BIDS sidecar | Required for quantification |
| BackgroundSuppression | BIDS sidecar | Usually true |

**Known Issues:**
- DICOM headers often lack complete ASL parameters
- Control/label order not specified in headers
- Must document settings at acquisition time
- LabelingDuration may be missing

## Parameter Extraction Priority

The osipy metadata mapper uses this priority:

```text
1. BIDS sidecar JSON
   └─ Highest confidence, standardized format

2. Vendor private tags
   └─ Parsed by vendor-specific parser

3. Standard DICOM tags
   └─ (0018,xxxx) series

4. User-provided values
   └─ Via function parameters

5. Modality defaults
   └─ Reasonable assumptions per modality
```

## Verification Checklist

When loading multi-vendor data, verify:

### DCE
- [ ] TR is in expected range (typically 2-10 ms for fast DCE)
- [ ] Flip angle(s) recorded
- [ ] Number of timepoints matches expectation
- [ ] Temporal resolution calculated correctly

### DSC
- [ ] TE is appropriate (typically 25-50 ms for gradient echo)
- [ ] TR is in expected range (typically 1-2 s)
- [ ] Number of baseline volumes known

### ASL
- [ ] Labeling type identified (PCASL, PASL, CASL)
- [ ] PLD specified (or array for multi-PLD)
- [ ] Labeling duration specified (for pCASL/CASL)
- [ ] Control/label order known (from aslcontext.tsv)
- [ ] M0 type identified (Separate, Included, Absent)

### IVIM
- [ ] B-values extracted correctly
- [ ] B=0 acquisition present
- [ ] Sufficient b-values for IVIM (≥4, ideally 6+)
- [ ] Low b-values present (<200 s/mm²)

## Testing with Real Data

## See Also

- [How to Load Perfusion Data](../how-to/load-perfusion-data.md)
- [DICOM Standard](https://www.dicomstandard.org/)
- [BIDS Specification](https://bids-specification.readthedocs.io/)
