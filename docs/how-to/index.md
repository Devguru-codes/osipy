# How-To Guides

## Data Loading

- [Load Perfusion Data](load-perfusion-data.md) — Load NIfTI, DICOM, or BIDS data with vendor-specific metadata extraction.

## AIF Selection

- [Use Custom AIF](use-custom-aif.md) — Provide your own measured arterial input function.
- [Choose Population AIF](choose-population-aif.md) — Select the right population AIF model for your data.

## GPU Acceleration

- [Configure GPU/CPU Backend](enable-gpu-acceleration.md) — Enable GPU acceleration with CuPy or force CPU execution.

## Export & Visualization

- [Export BIDS Results](export-bids.md) — Save results in BIDS derivatives format.
- [Visualize Parameter Maps](visualize-parameter-maps.md) — Create parameter map visualizations.

## Model Fitting

- [Compare Multiple Models](fit-multiple-models.md) — Fit and compare different DCE models.
- [Handle Fitting Failures](handle-fitting-failures.md) — Diagnose and resolve fitting problems.

## Pipelines

- [Run Pipeline from YAML (CLI)](run-pipeline-cli.md) — Run pipelines from the command line with YAML configuration.
- [Run Complete Pipeline (Python API)](run-complete-pipeline.md) — End-to-end analysis via Python API.

## Extension

- [Add PK Model](add-pharmacokinetic-model.md) — Implement a custom pharmacokinetic model for DCE-MRI.
- [Add ASL Model](add-asl-quantification-model.md) — Add a custom labeling scheme or quantification model.
- [Add Deconvolution Method](add-deconvolution-method.md) — Implement a custom DSC deconvolution algorithm.
- [Add BBB ASL Model](add-bbb-asl-model.md) — Implement BBB water exchange measurement with multi-echo ASL.
