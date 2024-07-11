from data.observational.goodness_of_fit import GoodnessOfFit
import numpy as np
from unittest.mock import patch
import io
import sys
import pytest


class TestGoodnessOfFit:
    """Class to test the GoodnessOfFit class."""

    def setup_method(self):
        """Setup method to initialize the ObsQuasarData class."""
        self.base_dir = "data\observational\goodness_of_fit.py"
        self.obj = GoodnessOfFit(self.base_dir)
        self.obj.fit_curve()

        # Mocking data
        self.obj.log_L_1350_norm = np.array([1.0, 2.0, 3.0])
        self.obj.log_L_3000_norm = np.array([1.5, 2.5, 3.5])
        self.obj.log_τ = np.array([0.1, 0.2, 0.3])
        self.obj.log_τ_3000 = np.array([0.15, 0.25, 0.35])
        self.obj.sigma_tau_1350_log = np.array([0.01, 0.02, 0.03])
        self.obj.sigma_tau_3000_log = np.array([0.015, 0.025, 0.035])

        # Mocking fitted parameters
        self.obj.beta_1350 = 1.0
        self.obj.gamma_1350 = 0.5
        self.obj.beta_sym_1350 = 1.1
        self.obj.gamma_sym_1350 = 0.55
        self.obj.beta_3000 = 1.2
        self.obj.gamma_3000 = 0.6
        self.obj.beta_sym_3000 = 1.3
        self.obj.gamma_sym_3000 = 0.65

    def test_symmetrized_error(self):
        lower_error = 0.1
        upper_error = 0.2
        expected_result = 0.1373773448

        result = self.obj.symmetrized_error(lower_error, upper_error)
        assert result == pytest.approx(expected_result, rel=1e-9)

    def test_power_law_model(self):
        x = np.array([1.0, 2.0, 3.0])
        beta = 0.5
        gamma = 2.0
        expected_result = np.array([2.5, 4.5, 6.5])

        result = self.obj.power_law_model(x, beta, gamma)
        np.testing.assert_allclose(result, expected_result, rtol=1e-9)

    @patch("data.observational.goodness_of_fit.curve_fit")
    def test_fit_curve(self, mock_curve_fit):
        mock_curve_fit.side_effect = [
            (np.array([1.0, 2.0]), np.array([[0.1, 0], [0, 0.2]])),  # C IV asymmetrical
            (np.array([1.1, 2.1]), np.array([[0.11, 0], [0, 0.21]])),  # C IV symmetrical
            (np.array([1.2, 2.2]), np.array([[0.12, 0], [0, 0.22]])),  # Mg II asymmetrical
            (np.array([1.3, 2.3]), np.array([[0.13, 0], [0, 0.23]])),  # Mg II symmetrical
        ]

        # Call the method
        self.obj.fit_curve()

        # Assert values for C IV (asymmetrical)
        assert self.obj.beta_1350 == 1.0
        assert self.obj.gamma_1350 == 2.0
        assert self.obj.beta_std_1350 == pytest.approx(np.sqrt(0.1), rel=1e-9)
        assert self.obj.gamma_std_1350 == pytest.approx(np.sqrt(0.2), rel=1e-9)

        # Assert values for C IV (symmetrical)
        assert self.obj.beta_sym_1350 == 1.1
        assert self.obj.gamma_sym_1350 == 2.1
        assert self.obj.beta_sym_std_1350 == pytest.approx(np.sqrt(0.11), rel=1e-9)
        assert self.obj.gamma_sym_std_1350 == pytest.approx(np.sqrt(0.21), rel=1e-9)

        # Assert values for Mg II (asymmetrical)
        assert self.obj.beta_3000 == 1.2
        assert self.obj.gamma_3000 == 2.2
        assert self.obj.beta_std_3000 == pytest.approx(np.sqrt(0.12), rel=1e-9)
        assert self.obj.gamma_std_3000 == pytest.approx(np.sqrt(0.22), rel=1e-9)

        # Assert values for Mg II (symmetrical)
        assert self.obj.beta_sym_3000 == 1.3
        assert self.obj.gamma_sym_3000 == 2.3
        assert self.obj.beta_sym_std_3000 == pytest.approx(np.sqrt(0.13), rel=1e-9)
        assert self.obj.gamma_sym_std_3000 == pytest.approx(np.sqrt(0.23), rel=1e-9)

    def _mock_fit_curve_side_effect(self):
        # Mock data to test these lines specifically in "calculate_goodness_of_fit"
        # if self.beta_1350 is None or self.beta_3000 is None:
        # self.fit_curve()
        self.obj.beta_3000 = 1.2
        self.obj.gamma_3000 = 0.6

    @patch.object(GoodnessOfFit, "fit_curve")
    def test_calculate_goodness_of_fit(self, mock_fit_curve):
        self.obj.beta_1350 = 1.0
        self.obj.beta_3000 = None

        mock_fit_curve.side_effect = self._mock_fit_curve_side_effect

        result = self.obj.calculate_goodness_of_fit()

        mock_fit_curve.assert_called_once()

        # Expected values
        y_pred_1350 = self.obj.power_law_model(self.obj.log_L_1350_norm, self.obj.beta_1350, self.obj.gamma_1350)
        residuals_1350 = self.obj.log_τ - y_pred_1350
        expected_intrinsic_scatter_1350 = np.std(residuals_1350)

        y_pred_sym_1350 = self.obj.power_law_model(self.obj.log_L_1350_norm, self.obj.beta_sym_1350, self.obj.gamma_sym_1350)
        residuals_sym_1350 = self.obj.log_τ - y_pred_sym_1350
        expected_intrinsic_scatter_sym_1350 = np.std(residuals_sym_1350)

        y_pred_3000 = self.obj.power_law_model(self.obj.log_L_3000_norm, self.obj.beta_3000, self.obj.gamma_3000)
        residuals_3000 = self.obj.log_τ_3000 - y_pred_3000
        expected_intrinsic_scatter_3000 = np.std(residuals_3000)

        y_pred_sym_3000 = self.obj.power_law_model(self.obj.log_L_3000_norm, self.obj.beta_sym_3000, self.obj.gamma_sym_3000)
        residuals_sym_3000 = self.obj.log_τ_3000 - y_pred_sym_3000
        expected_intrinsic_scatter_sym_3000 = np.std(residuals_sym_3000)

        expected_result = {
            "C IV (asymmetrical)": {"intrinsic_scatter": expected_intrinsic_scatter_1350},
            "C IV (symmetrical)": {"intrinsic_scatter": expected_intrinsic_scatter_sym_1350},
            "Mg II (asymmetrical)": {"intrinsic_scatter": expected_intrinsic_scatter_3000},
            "Mg II (symmetrical)": {"intrinsic_scatter": expected_intrinsic_scatter_sym_3000},
        }

        # Assert that the result matches the expected values
        assert result["C IV (asymmetrical)"]["intrinsic_scatter"] == pytest.approx(
            expected_result["C IV (asymmetrical)"]["intrinsic_scatter"], rel=1e-9
        )
        assert result["C IV (symmetrical)"]["intrinsic_scatter"] == pytest.approx(
            expected_result["C IV (symmetrical)"]["intrinsic_scatter"], rel=1e-9
        )
        assert result["Mg II (asymmetrical)"]["intrinsic_scatter"] == pytest.approx(
            expected_result["Mg II (asymmetrical)"]["intrinsic_scatter"], rel=1e-9
        )
        assert result["Mg II (symmetrical)"]["intrinsic_scatter"] == pytest.approx(
            expected_result["Mg II (symmetrical)"]["intrinsic_scatter"], rel=1e-9
        )

    @patch.object(GoodnessOfFit, "fit_curve")
    @patch.object(GoodnessOfFit, "calculate_goodness_of_fit")
    def test_print_results(self, mock_calculate_goodness_of_fit, mock_fit_curve):
        mock_fit_curve.side_effect = self._mock_fit_curve_side_effect

        mock_calculate_goodness_of_fit.return_value = {
            "C IV (asymmetrical)": {"intrinsic_scatter": 0.1},
            "C IV (symmetrical)": {"intrinsic_scatter": 0.2},
            "Mg II (asymmetrical)": {"intrinsic_scatter": 0.3},
            "Mg II (symmetrical)": {"intrinsic_scatter": 0.4},
        }

        expected_output = (
            "\nEstimated parameters and goodness of fit for quasar dataset (C IV, asymmetrical):\n"
            "beta = 1.00 +/- 0.07\n"
            "gamma = 0.50 +/- 0.03\n"
            "Intrinsic scatter = 0.10\n"
            "\nEstimated parameters and goodness of fit for quasar dataset (C IV, symmetrical):\n"
            "beta = 1.10 +/- 0.08\n"
            "gamma = 0.55 +/- 0.04\n"
            "Intrinsic scatter = 0.20\n"
            "\nEstimated parameters and goodness of fit for quasar dataset (Mg II, asymmetrical):\n"
            "beta = 1.20 +/- 0.05\n"
            "gamma = 0.60 +/- 0.04\n"
            "Intrinsic scatter = 0.30\n"
            "\nEstimated parameters and goodness of fit for quasar dataset (Mg II, symmetrical):\n"
            "beta = 1.30 +/- 0.06\n"
            "gamma = 0.65 +/- 0.05\n"
            "Intrinsic scatter = 0.40\n"
        )

        captured_output = io.StringIO()
        sys.stdout = captured_output

        self.obj.print_results()

        sys.stdout = sys.__stdout__

        assert captured_output.getvalue() == expected_output


# Running the tests with pytest
if __name__ == "__main__":
    pytest.main()
