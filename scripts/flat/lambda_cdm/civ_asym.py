from data.observational.goodness_of_fit import GoodnessOfFit
from data.observational.reading_data import ObsQuasarData
from scripts.cosmological_distances import LuminosityDistanceCalculator
import numpy as np
from scipy.optimize import differential_evolution


class CIVasymMCMC(object):
    """A class to perform MCMC analysis for a flat Lambda-CDM model assuming
    asymmetrical errors.

    :param beta_1350: Estimated beta parameter from goodness of fit.
    :type beta_1350: float
    :param beta_std_1350: Standard deviation of beta parameter.
    :type beta_std_1350: float
    :param gamma_1350: Estimated gamma parameter from goodness of fit.
    :type gamma_1350: float
    :param gamma_std_1350: Standard deviation of gamma parameter.
    :type gamma_std_1350: float
    :param intrinsic_scatter: Estimated intrinsic scatter from goodness of fit.
    :type intrinsic_scatter: float
    :param z: Redshift values.
    :type z: np.ndarray
    :param log_L_1350_norm: Normalized log luminosity for 1350A.
    :type log_L_1350_norm: np.ndarray :param log_τ: Log time delay for 1350A. :type
        log_τ: np.ndarray
    :param sigma_tau_1350_log_asym: Asymmetrical errors in time delay measurements.
    :type sigma_tau_1350_log_asym: np.ndarray :param σ_F: Errors in flux measurements.
        :type σ_F: np.ndarray
    :param log_F_1350_norm: Normalized log flux for 1350A.
    :type log_F_1350_norm: np.ndarray
    :param dist_calc: Instance of LuminosityDistanceCalculator.
    :type dist_calc: LuminosityDistanceCalculator
    """

    def __init__(self):
        """Initializes the CIVasymMCMC class with the goodness of fit parameters and
        observational data."""

        # Initialize the ObsQuasarData class and read data
        data_file = "data/observational/reading_data.py"
        self.obs_data = ObsQuasarData(data_file)
        self.obs_data.load_data()
        self.obs_data.process_civ_data()

        # Initialize the GoodnessOfFit class and fit the curve
        base_dir = "data/observational/goodness_of_fit.py"
        gof = GoodnessOfFit(base_dir)
        gof.fit_curve()

        self.beta_1350 = gof.beta_1350
        self.beta_std_1350 = gof.beta_std_1350
        self.gamma_1350 = gof.gamma_1350
        self.gamma_std_1350 = gof.gamma_std_1350
        self.intrinsic_scatter_1350 = gof.calculate_goodness_of_fit()["C IV (asymmetrical)"]["intrinsic_scatter"]
        self.z = self.obs_data.z
        self.log_L_1350_norm = gof.log_L_1350_norm
        self.log_τ = self.obs_data.log_τ
        self.sigma_tau_1350_log_asym = gof.sigma_tau_1350_log_asym
        self.σ_F = self.obs_data.σ_F
        self.log_F_1350_norm = self.obs_data.log_F_1350 - np.log10(1e44)
        self.dist_calc = LuminosityDistanceCalculator(H0=70, c=299792.458, conversion_factor=3.08567758e24)

    def log_likelihood(self, params):
        """Calculates the log likelihood for the MCMC analysis.

        :param params: A list of estimated parameters from the Levenberg-Marquardt
            algorithm (lm) + free cosmological parameters.
        :type params: list
        :return: Log likelihood value.
        :rtype: float
        """
        self.beta_1350, self.gamma_1350, self.intrinsic_scatter_1350, Om = params

        # Calculate the luminosity distance for the given redshifts
        d_L_values = self.dist_calc.d_L(self.z, Om=Om, Ok=0, model="lcdm")

        # Calculate the theoretical log time delay
        ln_τ_th = self.beta_1350 + (self.gamma_1350 * self.log_F_1350_norm) + (self.gamma_1350 * np.log10(4 * np.pi))

        mask = d_L_values != 0
        ln_τ_th[mask] += 2 * self.gamma_1350 * np.log10(d_L_values[mask])

        ln_τ_th = np.nan_to_num(ln_τ_th)  # Handle any remaining NaN or inf values

        # Calculate the total variance s^2, including measurement errors of time delay, flux, and intrinsic scatter
        s_squared = (self.sigma_tau_1350_log_asym**2) + (self.gamma_1350 * self.σ_F) ** 2 + (self.intrinsic_scatter_1350**2)

        # Calculate the log likelihood
        term1 = np.sum(np.square(self.log_τ - ln_τ_th) / s_squared, axis=0)  # First term of the likelihood function (right)
        term2 = np.sum(np.log(2 * np.pi * s_squared), axis=0)  # Second term of the likelihood function (left)
        ln_LF = -0.5 * (term1 + term2)

        return ln_LF

    def neg_log_likelihood(self, params):
        """Calculates the negative log likelihood for optimization.

        :param params: A list of estimated parameters from the Levenberg-Marquardt
            algorithm (lm) + free cosmological parameters.
        :type params: list
        :return: Negative log likelihood value.
        :rtype: float
        """
        return -self.log_likelihood(params)

    def optimize_params(self):
        """Optimizes the parameters to determine the maximum likelihood estimates.

        :return: Optimized parameters and chi-squared value.
        :rtype: tuple
        """
        # Run the optimization to find the maximum likelihood estimate
        bounds_guess = [(0, 2), (0, 1), (0, 1), (0, 1)]
        result = differential_evolution(self.neg_log_likelihood, bounds_guess)
        params_max_likelihood = result.x
        LF_max = -result.fun  # The negative log-likelihood is minimized, so flip the sign

        chi_squared = -2 * LF_max
        print("Chi-squared value (-2ln(LF_max)): {:.2f}".format(chi_squared))

        beta_fit, gamma_fit, intrinsic_scatter_fit, Om_fit = params_max_likelihood
        print("Maximum likelihood estimates:")
        print("β = {0:.3f}".format(beta_fit))
        print("γ = {0:.3f}".format(gamma_fit))
        print("σ_int = {0:.3f}".format(intrinsic_scatter_fit))
        print("Ω_M = {0:.3f}".format(Om_fit))

        return params_max_likelihood, chi_squared


# Example code to test log_likelihood function
if __name__ == "__main__":
    beta_13 = 0.98
    gamma_13 = 0.41
    intrinsic_scatter_13 = 0.27
    Om_13 = 0.3
    params = [beta_13, gamma_13, intrinsic_scatter_13, Om_13]

    mcmc_instance = CIVasymMCMC()
    log_likelihood_value = mcmc_instance.log_likelihood(params)
    print(log_likelihood_value)

    params_max_likelihood, chi_squared = mcmc_instance.optimize_params()
    print(f"Optimized parameters: {params_max_likelihood}")
    print(f"Chi-squared value: {chi_squared}")
