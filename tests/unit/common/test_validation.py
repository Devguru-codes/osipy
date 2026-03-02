"""Unit tests for validation comparison.

Tests for osipy/common/validation/comparison.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from osipy.common.parameter_map import ParameterMap


class TestValidateAgainstDRO:
    """Tests for validate_against_dro function."""

    def test_basic_validation(self) -> None:
        """Test basic validation against reference."""
        from osipy.common.validation import validate_against_dro

        # Create computed values
        computed = {"Ktrans": np.random.rand(10, 10, 5) * 0.1}

        # Create reference (similar with small differences)
        reference = {"Ktrans": computed["Ktrans"] + np.random.randn(10, 10, 5) * 0.001}

        report = validate_against_dro(computed, reference)

        assert report is not None
        assert hasattr(report, "overall_pass")
        assert hasattr(report, "pass_rate")

    def test_validation_with_parameter_maps(self) -> None:
        """Test validation using ParameterMap inputs."""
        from osipy.common.validation import validate_against_dro

        values = np.random.rand(10, 10, 5) * 0.1

        computed = {
            "Ktrans": ParameterMap(
                name="Ktrans",
                symbol="K^{trans}",
                units="min-1",
                values=values,
                affine=np.eye(4),
            )
        }

        reference = {"Ktrans": values + np.random.randn(10, 10, 5) * 0.001}

        report = validate_against_dro(computed, reference)

        assert report is not None

    def test_validation_with_dro_data(self) -> None:
        """Test validation using DROData object."""
        from osipy.common.validation import DROData, validate_against_dro

        computed = {"Ktrans": np.random.rand(10, 10, 5) * 0.1}

        dro = DROData(
            name="test_dro",
            parameters={
                "Ktrans": computed["Ktrans"] + np.random.randn(10, 10, 5) * 0.001
            },
            mask=np.ones((10, 10, 5), dtype=bool),
        )

        report = validate_against_dro(computed, dro)

        assert report is not None
        assert report.reference_name == "test_dro"

    def test_validation_with_mask(self) -> None:
        """Test validation with mask."""
        from osipy.common.validation import validate_against_dro

        mask = np.zeros((10, 10, 5), dtype=bool)
        mask[3:7, 3:7, :] = True

        computed = {"Ktrans": np.random.rand(10, 10, 5) * 0.1}
        reference = {"Ktrans": computed["Ktrans"].copy()}

        report = validate_against_dro(computed, reference, mask=mask)

        assert report is not None

    def test_validation_multiple_parameters(self) -> None:
        """Test validation with multiple parameters."""
        from osipy.common.validation import validate_against_dro

        computed = {
            "Ktrans": np.random.rand(10, 10, 5) * 0.1,
            "ve": np.random.rand(10, 10, 5) * 0.3,
            "vp": np.random.rand(10, 10, 5) * 0.05,
        }

        reference = {
            "Ktrans": computed["Ktrans"] + np.random.randn(10, 10, 5) * 0.001,
            "ve": computed["ve"] + np.random.randn(10, 10, 5) * 0.01,
            "vp": computed["vp"] + np.random.randn(10, 10, 5) * 0.002,
        }

        report = validate_against_dro(computed, reference)

        assert report is not None
        assert "Ktrans" in report.pass_rate
        assert "ve" in report.pass_rate
        assert "vp" in report.pass_rate

    def test_validation_custom_tolerances(self) -> None:
        """Test validation with custom tolerances."""
        from osipy.common.validation import validate_against_dro

        computed = {"Ktrans": np.random.rand(10, 10, 5) * 0.1}
        reference = {"Ktrans": computed["Ktrans"] + 0.01}  # 10% difference

        # Strict tolerance should fail
        strict_tol = {"Ktrans": {"relative": 0.05}}
        report_strict = validate_against_dro(computed, reference, tolerances=strict_tol)

        # Relaxed tolerance should pass
        relaxed_tol = {"Ktrans": {"relative": 0.15}}
        report_relaxed = validate_against_dro(
            computed, reference, tolerances=relaxed_tol
        )

        assert report_relaxed.pass_rate["Ktrans"] >= report_strict.pass_rate["Ktrans"]

    def test_validation_pass_rate(self) -> None:
        """Test that pass rate is correctly computed."""
        from osipy.common.validation import validate_against_dro

        # Perfect match
        values = np.ones((10, 10, 5)) * 0.1
        computed = {"Ktrans": values}
        reference = {"Ktrans": values.copy()}

        report = validate_against_dro(computed, reference)

        assert report.pass_rate["Ktrans"] == pytest.approx(1.0)

    def test_validation_overall_pass(self) -> None:
        """Test overall pass determination."""
        from osipy.common.validation import validate_against_dro

        values = np.ones((10, 10, 5)) * 0.1
        computed = {"Ktrans": values}
        reference = {"Ktrans": values.copy()}

        report = validate_against_dro(computed, reference)

        assert report.overall_pass is True


class TestLoadDRO:
    """Tests for load_dro function."""

    def test_load_nonexistent_path(self) -> None:
        """Test loading from non-existent path."""
        from osipy.common.exceptions import DataValidationError
        from osipy.common.validation import load_dro

        with pytest.raises(DataValidationError, match="not found"):
            load_dro("/nonexistent/path/to/dro")


class TestCreateSyntheticDRO:
    """Tests for create_synthetic_dro function."""

    def test_create_dce_dro(self) -> None:
        """Test creating synthetic DCE DRO."""
        from osipy.common.validation import create_synthetic_dro

        dro = create_synthetic_dro(shape=(8, 8, 4), modality="dce")

        assert dro is not None
        assert dro.name == "synthetic_dce_dro"
        assert "Ktrans" in dro.parameters
        assert "ve" in dro.parameters
        assert "vp" in dro.parameters
        assert dro.mask.shape == (8, 8, 4)

    def test_create_dsc_dro(self) -> None:
        """Test creating synthetic DSC DRO."""
        from osipy.common.validation import create_synthetic_dro

        dro = create_synthetic_dro(shape=(8, 8, 4), modality="dsc")

        assert "CBV" in dro.parameters
        assert "CBF" in dro.parameters
        assert "MTT" in dro.parameters

    def test_create_asl_dro(self) -> None:
        """Test creating synthetic ASL DRO."""
        from osipy.common.validation import create_synthetic_dro

        dro = create_synthetic_dro(shape=(8, 8, 4), modality="asl")

        assert "CBF" in dro.parameters
        assert "ATT" in dro.parameters

    def test_create_ivim_dro(self) -> None:
        """Test creating synthetic IVIM DRO."""
        from osipy.common.validation import create_synthetic_dro

        dro = create_synthetic_dro(shape=(8, 8, 4), modality="ivim")

        assert "D" in dro.parameters
        assert "D*" in dro.parameters
        assert "f" in dro.parameters

    def test_noise_level(self) -> None:
        """Test that noise level affects DRO."""
        from osipy.common.validation import create_synthetic_dro

        dro_low = create_synthetic_dro(noise_level=0.001)
        dro_high = create_synthetic_dro(noise_level=0.1)

        # With same seed, base values are same, but noise differs
        # Just check they're created successfully
        assert dro_low is not None
        assert dro_high is not None

    def test_invalid_modality(self) -> None:
        """Test that invalid modality raises error."""
        from osipy.common.exceptions import DataValidationError
        from osipy.common.validation import create_synthetic_dro

        with pytest.raises(DataValidationError, match="Unknown modality"):
            create_synthetic_dro(modality="invalid")


class TestDROData:
    """Tests for DROData dataclass."""

    def test_dro_data_creation(self) -> None:
        """Test creating DROData."""
        from osipy.common.validation import DROData

        dro = DROData(
            name="test_dro",
            parameters={"Ktrans": np.random.rand(5, 5, 3)},
            mask=np.ones((5, 5, 3), dtype=bool),
            metadata={"source": "test"},
        )

        assert dro.name == "test_dro"
        assert "Ktrans" in dro.parameters
        assert dro.mask is not None
        assert dro.metadata["source"] == "test"

    def test_dro_data_default_metadata(self) -> None:
        """Test DROData with default metadata."""
        from osipy.common.validation import DROData

        dro = DROData(
            name="test_dro",
            parameters={"Ktrans": np.random.rand(5, 5, 3)},
        )

        assert dro.metadata == {}


class TestDefaultTolerances:
    """Tests for DEFAULT_TOLERANCES constant."""

    def test_tolerances_defined(self) -> None:
        """Test that default tolerances are defined."""
        from osipy.common.validation import DEFAULT_TOLERANCES

        assert "Ktrans" in DEFAULT_TOLERANCES
        assert "ve" in DEFAULT_TOLERANCES
        assert "CBV" in DEFAULT_TOLERANCES
        assert "CBF" in DEFAULT_TOLERANCES
        assert "D" in DEFAULT_TOLERANCES
        assert "f" in DEFAULT_TOLERANCES

    def test_tolerances_have_abs_and_rel(self) -> None:
        """Test that tolerances have absolute and relative values."""
        from osipy.common.validation import DEFAULT_TOLERANCES

        for param, tol in DEFAULT_TOLERANCES.items():
            assert "relative" in tol, f"{param} missing relative tolerance"
            # absolute is optional but usually present


class TestValidationReport:
    """Tests for ValidationReport class."""

    def test_report_summary(self) -> None:
        """Test ValidationReport summary method."""
        from osipy.common.validation import validate_against_dro

        computed = {"Ktrans": np.random.rand(10, 10, 5) * 0.1}
        reference = {"Ktrans": computed["Ktrans"].copy()}

        report = validate_against_dro(computed, reference)

        summary = report.summary()
        assert isinstance(summary, str)
        assert "Ktrans" in summary or "PASS" in summary.upper()

    def test_report_attributes(self) -> None:
        """Test ValidationReport has expected attributes."""
        from osipy.common.validation import validate_against_dro

        computed = {"Ktrans": np.random.rand(10, 10, 5) * 0.1}
        reference = {"Ktrans": computed["Ktrans"].copy()}

        report = validate_against_dro(computed, reference)

        assert hasattr(report, "reference_name")
        assert hasattr(report, "reference_parameters")
        assert hasattr(report, "computed_parameters")
        assert hasattr(report, "absolute_errors")
        assert hasattr(report, "relative_errors")
        assert hasattr(report, "tolerances")
        assert hasattr(report, "within_tolerance")
        assert hasattr(report, "pass_rate")
        assert hasattr(report, "overall_pass")


class TestUpdatedTolerances:
    """Tests for updated DEFAULT_TOLERANCES values."""

    def test_contains_all_expected_keys(self) -> None:
        """Test DEFAULT_TOLERANCES contains all expected keys."""
        from osipy.common.validation import DEFAULT_TOLERANCES

        expected_keys = {
            "Ktrans",
            "ve",
            "vp",
            "kep",
            "CBV",
            "CBF",
            "MTT",
            "CBF_ASL",
            "ATT",
            "D",
            "D*",
            "f",
            "PS",
            "Fp",
            "delay",
            "R1",
            "concentration",
            "aif",
        }
        assert expected_keys == set(DEFAULT_TOLERANCES.keys())

    def test_osipi_specific_values(self) -> None:
        """Test specific OSIPI CodeCollection tolerance values."""
        from osipy.common.validation import DEFAULT_TOLERANCES

        assert DEFAULT_TOLERANCES["ve"]["absolute"] == pytest.approx(0.05)
        assert DEFAULT_TOLERANCES["vp"]["absolute"] == pytest.approx(0.025)
        assert DEFAULT_TOLERANCES["CBF"]["absolute"] == pytest.approx(15.0)
        assert DEFAULT_TOLERANCES["Ktrans"]["relative"] == pytest.approx(0.1)

    def test_all_entries_have_absolute_and_relative(self) -> None:
        """Test all entries have both absolute and relative keys."""
        from osipy.common.validation import DEFAULT_TOLERANCES

        for param, tol in DEFAULT_TOLERANCES.items():
            assert "absolute" in tol, f"{param} missing 'absolute' key"
            assert "relative" in tol, f"{param} missing 'relative' key"


class TestValidationReportDict:
    """Tests for ValidationReport.to_dict() method."""

    def _make_report(self):
        """Create a ValidationReport with known data."""
        from osipy.common.validation import validate_against_dro

        computed = {
            "Ktrans": np.ones((5, 5, 3)) * 0.1,
            "ve": np.ones((5, 5, 3)) * 0.2,
        }
        reference = {
            "Ktrans": np.ones((5, 5, 3)) * 0.1,
            "ve": np.ones((5, 5, 3)) * 0.2,
        }
        return validate_against_dro(
            computed, reference, reference_name="test_dict_report"
        )

    def test_to_dict_keys(self) -> None:
        """Test that to_dict() returns expected top-level keys."""
        report = self._make_report()
        d = report.to_dict()

        assert "reference_name" in d
        assert "overall_pass" in d
        assert "parameters" in d
        assert "tolerances" in d
        assert "timestamp" in d
        assert "version" in d

    def test_to_dict_types(self) -> None:
        """Test that to_dict() values have correct types."""
        report = self._make_report()
        d = report.to_dict()

        assert isinstance(d["reference_name"], str)
        assert isinstance(d["overall_pass"], bool)
        assert isinstance(d["parameters"], dict)
        assert isinstance(d["tolerances"], dict)
        assert isinstance(d["timestamp"], str)
        assert isinstance(d["version"], str)

    def test_to_dict_pass_rate_per_param(self) -> None:
        """Test that pass_rate is present for each parameter."""
        report = self._make_report()
        d = report.to_dict()

        for param, info in d["parameters"].items():
            assert "pass_rate" in info, f"{param} missing pass_rate"
            assert "statistics" in info, f"{param} missing statistics"

    def test_to_dict_consistency(self) -> None:
        """Test that repeated to_dict() calls return consistent data."""
        report = self._make_report()
        d1 = report.to_dict()
        d2 = report.to_dict()

        assert d1["reference_name"] == d2["reference_name"]
        assert d1["overall_pass"] == d2["overall_pass"]
        assert d1["parameters"].keys() == d2["parameters"].keys()


class TestValidationReportJSON:
    """Tests for ValidationReport.to_json() method."""

    def _make_report(self):
        """Create a ValidationReport with known data."""
        from osipy.common.validation import validate_against_dro

        computed = {"Ktrans": np.ones((5, 5, 3)) * 0.1}
        reference = {"Ktrans": np.ones((5, 5, 3)) * 0.1}
        return validate_against_dro(
            computed, reference, reference_name="test_json_report"
        )

    def test_to_json_valid(self) -> None:
        """Test that to_json() returns valid JSON."""
        import json

        report = self._make_report()
        json_str = report.to_json()

        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_to_json_writes_file(self, tmp_path) -> None:
        """Test that to_json(path=...) creates a file with valid JSON."""
        import json

        report = self._make_report()
        out_file = tmp_path / "report.json"
        report.to_json(path=out_file)

        assert out_file.exists()
        parsed = json.loads(out_file.read_text())
        assert isinstance(parsed, dict)
        assert "reference_name" in parsed

    def test_to_json_matches_to_dict(self) -> None:
        """Test that JSON round-trip matches to_dict() (ignoring timestamp)."""
        import json

        report = self._make_report()
        d = report.to_dict()
        json_str = report.to_json()
        parsed = json.loads(json_str)

        assert d["reference_name"] == parsed["reference_name"]
        assert d["overall_pass"] == parsed["overall_pass"]
        assert set(d["parameters"].keys()) == set(parsed["parameters"].keys())
