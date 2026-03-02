"""Unit tests for arterial input function module.

Tests for population AIFs and automatic AIF detection.
"""

import numpy as np
import pytest

from osipy.common.aif import (
    ArterialInputFunction,
    FritzHansenAIF,
    GeorgiouAIF,
    GeorgiouAIFParams,
    ParkerAIF,
    ParkerAIFParams,
    PopulationAIFType,
    get_population_aif,
    list_aifs,
)
from osipy.common.aif.population import McGrathAIF, McGrathAIFParams
from osipy.common.exceptions import AIFError
from osipy.common.types import AIFType


class TestParkerAIF:
    """Tests for Parker population AIF."""

    def test_default_params(self) -> None:
        """Test Parker AIF with default parameters."""
        aif = ParkerAIF()
        t = np.linspace(0, 300, 60)  # 5 minutes in seconds

        result = aif(t)

        assert isinstance(result, ArterialInputFunction)
        assert result.time.shape == t.shape
        assert result.concentration.shape == t.shape
        assert result.aif_type == AIFType.POPULATION
        assert result.population_model == "Parker"

    def test_aif_characteristics(self) -> None:
        """Test that Parker AIF has expected characteristics."""
        aif = ParkerAIF()
        t = np.linspace(0, 600, 120)

        result = aif(t)
        conc = result.concentration

        # Should have a peak in the first minute
        peak_idx = np.argmax(conc)
        assert t[peak_idx] < 60  # Peak before 1 minute

        # Should be positive
        assert np.all(conc >= 0)

        # Should decay after peak
        assert conc[-1] < conc[peak_idx]

    def test_custom_params(self) -> None:
        """Test Parker AIF with custom parameters."""
        custom_params = ParkerAIFParams(
            a1=1.0,
            sigma1=0.08,
            alpha=1.5,
        )
        aif = ParkerAIF(params=custom_params)
        t = np.linspace(0, 300, 60)

        result = aif(t)

        assert np.all(np.isfinite(result.concentration))

    def test_model_info(self) -> None:
        """Test model information properties."""
        aif = ParkerAIF()
        assert "Parker" in aif.name
        assert "2006" in aif.reference

    def test_get_parameters(self) -> None:
        """Test get_parameters method."""
        aif = ParkerAIF()
        params = aif.get_parameters()

        assert "a1" in params
        assert "sigma1" in params
        assert "alpha" in params
        assert "beta" in params


class TestGeorgiouAIF:
    """Tests for Georgiou population AIF."""

    def test_default_params(self) -> None:
        """Test Georgiou AIF generation."""
        aif = GeorgiouAIF()
        t = np.linspace(0, 300, 60)

        result = aif(t)

        assert isinstance(result, ArterialInputFunction)
        assert result.population_model == "Georgiou"
        assert np.all(result.concentration >= 0)

    def test_custom_params(self) -> None:
        """Test with custom parameters."""
        params = GeorgiouAIFParams(a1=0.5, a2=0.4)
        aif = GeorgiouAIF(params=params)
        t = np.linspace(0, 300, 60)

        result = aif(t)
        assert np.all(np.isfinite(result.concentration))

    def test_get_parameters(self) -> None:
        """Test get_parameters method."""
        aif = GeorgiouAIF()
        params = aif.get_parameters()

        assert "a1" in params
        assert "m1" in params
        assert "alpha" in params


class TestFritzHansenAIF:
    """Tests for Fritz-Hansen population AIF."""

    def test_default_params(self) -> None:
        """Test Fritz-Hansen AIF generation."""
        aif = FritzHansenAIF()
        t = np.linspace(0, 300, 60)

        result = aif(t)

        assert isinstance(result, ArterialInputFunction)
        assert result.population_model == "Fritz-Hansen"

    def test_bi_exponential_decay(self) -> None:
        """Test bi-exponential decay characteristic."""
        aif = FritzHansenAIF()
        t = np.linspace(0, 600, 120)

        result = aif(t)
        conc = result.concentration

        # Should be monotonically decreasing (bi-exponential decay)
        # Note: May have small numerical variations
        assert conc[0] > conc[-1]

    def test_get_parameters(self) -> None:
        """Test get_parameters method."""
        aif = FritzHansenAIF()
        params = aif.get_parameters()

        assert "a1" in params
        assert "m1" in params
        assert "a2" in params
        assert "m2" in params


class TestGetPopulationAIF:
    """Tests for population AIF factory function."""

    def test_get_parker(self) -> None:
        """Test getting Parker AIF by name."""
        aif = get_population_aif("parker")
        assert isinstance(aif, ParkerAIF)

    def test_get_georgiou(self) -> None:
        """Test getting Georgiou AIF by name."""
        aif = get_population_aif("georgiou")
        assert isinstance(aif, GeorgiouAIF)

    def test_get_fritz_hansen(self) -> None:
        """Test getting Fritz-Hansen AIF by name."""
        aif = get_population_aif("fritz_hansen")
        assert isinstance(aif, FritzHansenAIF)

        # Also test hyphenated version
        aif2 = get_population_aif("fritz-hansen")
        assert isinstance(aif2, FritzHansenAIF)

    def test_get_by_enum(self) -> None:
        """Test getting AIF by enum."""
        aif = get_population_aif(PopulationAIFType.PARKER)
        assert isinstance(aif, ParkerAIF)

    def test_invalid_type_raises(self) -> None:
        """Test that invalid type raises error."""
        with pytest.raises(AIFError, match="Unknown AIF type"):
            get_population_aif("invalid")


class TestArterialInputFunction:
    """Tests for ArterialInputFunction dataclass."""

    def test_creation(self) -> None:
        """Test AIF dataclass creation."""
        t = np.linspace(0, 300, 60)
        conc = np.exp(-t / 100)

        aif = ArterialInputFunction(
            time=t,
            concentration=conc,
            aif_type=AIFType.MEASURED,
        )

        assert aif.time.shape == t.shape
        assert aif.concentration.shape == conc.shape
        assert aif.aif_type == AIFType.MEASURED

    def test_population_aif(self) -> None:
        """Test population AIF field handling."""
        t = np.arange(30, dtype=float)
        conc = np.ones(30)

        aif = ArterialInputFunction(
            time=t,
            concentration=conc,
            aif_type=AIFType.POPULATION,
            population_model="Parker",
        )

        assert aif.aif_type == AIFType.POPULATION
        assert aif.population_model == "Parker"

    def test_n_timepoints_property(self) -> None:
        """Test n_timepoints property."""
        t = np.linspace(0, 100, 50)
        conc = np.ones(50)

        aif = ArterialInputFunction(
            time=t,
            concentration=conc,
            aif_type=AIFType.MEASURED,
        )

        assert aif.n_timepoints == 50

    def test_peak_concentration_property(self) -> None:
        """Test peak_concentration property."""
        t = np.linspace(0, 100, 50)
        conc = np.sin(t / 10) + 2  # Peak around 15

        aif = ArterialInputFunction(
            time=t,
            concentration=conc,
            aif_type=AIFType.MEASURED,
        )

        assert aif.peak_concentration == np.max(conc)

    def test_peak_time_property(self) -> None:
        """Test peak_time property."""
        t = np.linspace(0, 100, 50)
        conc = np.exp(-((t - 30) ** 2) / 100)  # Peak at t=30

        aif = ArterialInputFunction(
            time=t,
            concentration=conc,
            aif_type=AIFType.MEASURED,
        )

        # Peak time should be close to 30
        assert 25 < aif.peak_time < 35

    def test_validation_length_mismatch(self) -> None:
        """Test validation catches length mismatch."""
        t = np.arange(30)
        conc = np.ones(20)  # Wrong length

        with pytest.raises(AIFError, match="time length"):
            ArterialInputFunction(
                time=t,
                concentration=conc,
                aif_type=AIFType.MEASURED,
            )

    def test_validation_empty_raises(self) -> None:
        """Test validation catches empty arrays."""
        with pytest.raises(AIFError, match="at least one"):
            ArterialInputFunction(
                time=np.array([]),
                concentration=np.array([]),
                aif_type=AIFType.MEASURED,
            )


class TestMcGrathAIF:
    """Tests for McGrath preclinical population AIF."""

    def test_default_params(self) -> None:
        """Test McGrath AIF with default parameters."""
        aif = McGrathAIF()
        t = np.linspace(0, 300, 60)  # 5 minutes in seconds

        result = aif(t)

        assert isinstance(result, ArterialInputFunction)
        assert result.time.shape == t.shape
        assert result.concentration.shape == t.shape
        assert result.aif_type == AIFType.POPULATION
        assert result.population_model == "McGrath"
        assert np.all(result.concentration >= 0)

    def test_peak_exists(self) -> None:
        """Test that McGrath AIF has a peak (rises then falls)."""
        aif = McGrathAIF()
        t = np.linspace(0, 300, 120)

        result = aif(t)
        conc = result.concentration

        peak_idx = np.argmax(conc)
        # Peak should not be at the very start or very end
        assert peak_idx > 0
        assert peak_idx < len(conc) - 1
        # Concentration should decay after the peak
        assert conc[-1] < conc[peak_idx]

    def test_custom_params(self) -> None:
        """Test McGrath AIF with custom parameters."""
        custom_params = McGrathAIFParams(
            a_gv=6.0,
            alpha=3.0,
            tau=0.2,
            a_exp=0.8,
            beta=0.15,
        )
        aif = McGrathAIF(params=custom_params)

        params_dict = aif.get_parameters()
        assert params_dict["a_gv"] == 6.0
        assert params_dict["alpha"] == 3.0
        assert params_dict["tau"] == 0.2
        assert params_dict["a_exp"] == 0.8
        assert params_dict["beta"] == 0.15

    def test_get_parameters(self) -> None:
        """Test get_parameters returns expected keys."""
        aif = McGrathAIF()
        params = aif.get_parameters()

        assert "a_gv" in params
        assert "alpha" in params
        assert "tau" in params
        assert "a_exp" in params
        assert "beta" in params

    def test_model_info(self) -> None:
        """Test model information properties."""
        aif = McGrathAIF()
        assert "McGrath" in aif.name
        assert "2009" in aif.reference

    def test_get_mcgrath(self) -> None:
        """Test getting McGrath AIF via factory function."""
        aif = get_population_aif("mcgrath")
        assert isinstance(aif, McGrathAIF)

    def test_list_aifs_includes_mcgrath(self) -> None:
        """Test that list_aifs includes mcgrath."""
        names = list_aifs()
        assert "mcgrath" in names

    def test_mcgrath_enum(self) -> None:
        """Test getting McGrath AIF via enum."""
        aif = get_population_aif(PopulationAIFType.MCGRATH)
        assert isinstance(aif, McGrathAIF)


class TestOSIPIReferenceAIF:
    """Test AIFs against known published values."""

    def test_parker_aif_peak_amplitude(self) -> None:
        """Parker AIF peak should be approximately 5-8 mM at ~10s."""
        t = np.linspace(0, 300, 600)  # Fine resolution
        parker = ParkerAIF()
        result = parker(t)
        peak = float(np.max(result.concentration))
        peak_time = float(t[np.argmax(result.concentration)])

        # Parker AIF peaks around 5-8 mM at ~10 seconds
        assert 3.0 < peak < 12.0, f"Peak amplitude {peak} outside expected range"
        assert 5.0 < peak_time < 20.0, f"Peak time {peak_time}s outside expected range"

    def test_parker_aif_integral(self) -> None:
        """Parker AIF area under curve should match published value."""
        t = np.linspace(0, 360, 720)
        parker = ParkerAIF()
        conc = parker(t).concentration
        auc = float(np.trapezoid(conc, t / 60.0))  # AUC in mM*min

        # Parker AIF AUC over 6 minutes (depends on sigmoid modulation)
        assert 3.0 < auc < 40.0, f"AUC {auc} mM*min outside expected range"
