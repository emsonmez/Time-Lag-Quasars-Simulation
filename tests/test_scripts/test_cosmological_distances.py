from scripts.cosmological_distances import LuminosityDistanceCalculator
import numpy as np
import pytest


class TestLuminosityDistanceCalculator:
    """Class to test the LuminosityDistanceCalculator class."""

    def setup_method(self):
        """Setup method to initialize the LuminosityDistanceCalculator class."""
        self.calculator = LuminosityDistanceCalculator()
        self.Om = 0.3
        self.H0 = 70
        self.z_values = np.array([0.5, 5.0])
        self.Ok_values_non_zero = [0.1, -0.1]
        self.Ok_zero = 0.0
        self.w_x = -0.5

    def test_integrand_lcdm(self):
        manually_calculated_integrands_non_zero = [
            np.array([0.0105387302, 0.0017197979]),
            np.array([0.0113382256, 0.0018142875]),
        ]

        for Ok, expected_values in zip(self.Ok_values_non_zero, manually_calculated_integrands_non_zero):
            integrand_values = np.array([self.calculator.integrand_lcdm(z, self.Om, Ok, self.H0) for z in self.z_values])
            np.testing.assert_almost_equal(
                integrand_values,
                expected_values,
                decimal=8,
                err_msg=f"Lambda-CDM integrand test failed for Ok={Ok}.",
            )

        manually_calculated_integrand_zero = np.array([0.0109165817, 0.0017651488])
        integrand_values_zero = np.array([self.calculator.integrand_lcdm(z, self.Om, self.Ok_zero, self.H0) for z in self.z_values])
        np.testing.assert_almost_equal(
            integrand_values_zero,
            manually_calculated_integrand_zero,
            decimal=8,
            err_msg="Lambda-CDM integrand test failed for Ok=0.",
        )

    def test_integrand_xcdm(self):
        manually_calculated_integrands_non_zero = [
            np.array([0.0093393218, 0.0016257068]),
            np.array([0.0095086209, 0.0016725027]),
        ]

        for Ok, expected_values in zip(self.Ok_values_non_zero, manually_calculated_integrands_non_zero):
            integrand_values = np.array([self.calculator.integrand_xcdm(z, self.Om, Ok, self.H0, self.w_x) for z in self.z_values])
            np.testing.assert_almost_equal(
                integrand_values,
                expected_values,
                decimal=8,
                err_msg=f"X-CDM integrand test failed for Ok={Ok}.",
            )

        manually_calculated_integrand_zero = np.array([0.0094228309, 0.0016486069])
        integrand_values_zero = np.array([self.calculator.integrand_xcdm(z, self.Om, self.Ok_zero, self.H0, self.w_x) for z in self.z_values])
        np.testing.assert_almost_equal(
            integrand_values_zero,
            manually_calculated_integrand_zero,
            decimal=8,
            err_msg="X-CDM integrand test failed for Ok=0.",
        )

    def test_d_L(self):
        manually_calculated_d_L_lcdm = np.array([8.741533562e27, 1.439537185e29])
        manually_calculated_d_L_xcdm = np.array([8.023355001e27, 1.270613285e29])  # Replace with your actual values for XCDM

        d_L_lcdm = self.calculator.d_L(self.z_values, self.Om, self.Ok_zero, model="lcdm")
        np.testing.assert_almost_equal(
            d_L_lcdm,
            manually_calculated_d_L_lcdm,
            decimal=4,
            err_msg="Luminosity distance calculation failed for LCDM model with Ok=0.",
        )

        d_L_xcdm = self.calculator.d_L(self.z_values, self.Om, self.Ok_zero, model="xcdm", w_x=self.w_x)
        np.testing.assert_almost_equal(
            d_L_xcdm,
            manually_calculated_d_L_xcdm,
            decimal=8,
            err_msg="Luminosity distance calculation failed for XCDM model with Ok=0.",
        )

        Ok_values = [0.1, -0.1]
        manually_calculated_d_L_lcdm_non_zero = [
            np.array([]),  # Replace with actual values for Ok = 0.1
            np.array([]),  # Replace with actual values for Ok = -0.1
        ]
        manually_calculated_d_L_xcdm_non_zero = [
            np.array([]),  # Replace with actual values for Ok = 0.1
            np.array([]),  # Replace with actual values for Ok = -0.1
        ]

        for Ok, expected_values_lcdm, expected_values_xcdm in zip(
            Ok_values,
            manually_calculated_d_L_lcdm_non_zero,
            manually_calculated_d_L_xcdm_non_zero,
        ):
            d_L_lcdm_non_zero = self.calculator.d_L(self.z_values, self.Om, Ok, model="lcdm")
            np.testing.assert_almost_equal(
                d_L_lcdm_non_zero,
                expected_values_lcdm,
                decimal=8,
                err_msg=f"Luminosity distance calculation failed for LCDM model with Ok={Ok}.",
            )

            d_L_xcdm_non_zero = self.calculator.d_L(self.z_values, self.Om, Ok, model="xcdm", w_x=self.w_x)
            np.testing.assert_almost_equal(
                d_L_xcdm_non_zero,
                expected_values_xcdm,
                decimal=8,
                err_msg=f"Luminosity distance calculation failed for XCDM model with Ok={Ok}.",
            )


# Running the tests with pytest
if __name__ == "__main__":
    pytest.main()
