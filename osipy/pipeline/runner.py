"""Pipeline runner module.

This module provides a unified interface for running perfusion analysis
pipelines with automatic modality detection.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.common.exceptions import DataValidationError
from osipy.common.parameter_map import ParameterMap
from osipy.common.types import Modality

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class PipelineResult:
    """Generic pipeline result container.

    Attributes
    ----------
    modality : Modality
        Analyzed modality.
    parameter_maps : dict[str, ParameterMap]
        Output parameter maps.
    quality_mask : NDArray[np.bool_]
        Quality mask for valid results.
    metadata : dict[str, Any]
        Additional metadata and statistics.
    """

    modality: Modality
    parameter_maps: dict[str, ParameterMap]
    quality_mask: "NDArray[np.bool_]"
    metadata: dict[str, Any] = field(default_factory=dict)


def run_analysis(
    data: "NDArray[np.floating[Any]]",
    modality: Modality | str,
    **kwargs: Any,
) -> PipelineResult:
    """Run perfusion analysis with specified modality.

    This is a unified entry point that dispatches to the appropriate
    pipeline based on the modality.

    Parameters
    ----------
    data : NDArray
        Input data appropriate for the modality.
    modality : Modality or str
        Analysis modality: 'dce', 'dsc', 'asl', or 'ivim'.
    **kwargs
        Additional arguments passed to the specific pipeline.
        See individual pipeline documentation for details.

    Returns
    -------
    PipelineResult
        Analysis results.

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.pipeline import run_analysis
    >>> from osipy.common.types import Modality
    >>>
    >>> # Run DCE analysis
    >>> result = run_analysis(
    ...     dce_data,
    ...     modality=Modality.DCE,
    ...     time=time_vector,
    ...     model='extended_tofts',
    ... )
    >>>
    >>> # Run IVIM analysis
    >>> result = run_analysis(
    ...     dwi_data,
    ...     modality='ivim',
    ...     b_values=b_values,
    ... )
    """
    # Normalize modality
    if isinstance(modality, str):
        modality_map = {
            "dce": Modality.DCE,
            "dsc": Modality.DSC,
            "asl": Modality.ASL,
            "ivim": Modality.IVIM,
        }
        modality = modality_map.get(modality.lower(), Modality.DCE)

    if modality == Modality.DCE:
        return _run_dce_analysis(data, **kwargs)
    elif modality == Modality.DSC:
        return _run_dsc_analysis(data, **kwargs)
    elif modality == Modality.ASL:
        return _run_asl_analysis(data, **kwargs)
    elif modality == Modality.IVIM:
        return _run_ivim_analysis(data, **kwargs)
    else:
        msg = f"Unsupported modality: {modality}"
        raise DataValidationError(msg)


def _run_dce_analysis(
    data: "NDArray[np.floating[Any]]", **kwargs: Any
) -> PipelineResult:
    """Run DCE-MRI analysis."""
    from osipy.pipeline.dce_pipeline import DCEPipeline, DCEPipelineConfig

    # Extract configuration options
    model = kwargs.pop("model", "extended_tofts")
    aif_source = kwargs.pop("aif_source", "population")

    config = DCEPipelineConfig(
        model=model,
        aif_source=aif_source,
    )

    pipeline = DCEPipeline(config)
    result = pipeline.run(data, **kwargs)

    return PipelineResult(
        modality=Modality.DCE,
        parameter_maps=result.fit_result.parameter_maps,
        quality_mask=result.fit_result.quality_mask,
        metadata={
            "model": result.fit_result.model_name,
            "fitting_stats": result.fit_result.fitting_stats,
        },
    )


def _run_dsc_analysis(
    data: "NDArray[np.floating[Any]]", **kwargs: Any
) -> PipelineResult:
    """Run DSC-MRI analysis."""
    from osipy.pipeline.dsc_pipeline import DSCPipeline, DSCPipelineConfig

    te = kwargs.pop("te", 30.0)
    apply_leakage = kwargs.pop("apply_leakage_correction", True)

    config = DSCPipelineConfig(
        te=te,
        apply_leakage_correction=apply_leakage,
    )

    pipeline = DSCPipeline(config)
    result = pipeline.run(data, **kwargs)

    # Extract parameter maps
    param_maps = {
        "CBV": result.perfusion_maps.cbv,
        "CBF": result.perfusion_maps.cbf,
        "MTT": result.perfusion_maps.mtt,
    }
    if result.perfusion_maps.ttp is not None:
        param_maps["TTP"] = result.perfusion_maps.ttp

    return PipelineResult(
        modality=Modality.DSC,
        parameter_maps=param_maps,
        quality_mask=result.perfusion_maps.quality_mask
        if result.perfusion_maps.quality_mask is not None
        else np.ones(result.perfusion_maps.cbv.data.shape, dtype=bool),
        metadata={},
    )


def _run_asl_analysis(
    data: "NDArray[np.floating[Any]]", **kwargs: Any
) -> PipelineResult:
    """Run ASL analysis."""
    from osipy.asl import LabelingScheme
    from osipy.pipeline.asl_pipeline import ASLPipeline, ASLPipelineConfig

    labeling = kwargs.pop("labeling_scheme", LabelingScheme.PCASL)
    pld = kwargs.pop("pld", 1800.0)

    config = ASLPipelineConfig(
        labeling_scheme=labeling,
        pld=pld,
    )

    pipeline = ASLPipeline(config)

    # Handle different input formats
    if "control_data" in kwargs:
        result = pipeline.run(data, **kwargs)
    else:
        # Assume interleaved data
        m0_data = kwargs.pop("m0_data", 1.0)
        result = pipeline.run_from_alternating(data, m0_data, **kwargs)

    return PipelineResult(
        modality=Modality.ASL,
        parameter_maps={"CBF": result.cbf_result.cbf_map},
        quality_mask=result.cbf_result.quality_mask,
        metadata={},
    )


def _run_ivim_analysis(
    data: "NDArray[np.floating[Any]]", **kwargs: Any
) -> PipelineResult:
    """Run IVIM analysis."""
    from osipy.pipeline.ivim_pipeline import IVIMPipeline, IVIMPipelineConfig

    config = IVIMPipelineConfig()
    pipeline = IVIMPipeline(config)
    result = pipeline.run(data, **kwargs)

    return PipelineResult(
        modality=Modality.IVIM,
        parameter_maps={
            "D": result.fit_result.d_map,
            "D*": result.fit_result.d_star_map,
            "f": result.fit_result.f_map,
        },
        quality_mask=result.fit_result.quality_mask,
        metadata=result.fit_result.fitting_stats,
    )
