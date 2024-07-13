from scripts.cosmological_distances import LuminosityDistanceCalculator
from data.observational.reading_data import ObsQuasarData
import numpy as np
from scipy.optimize import curve_fit


class GoodnessOfFit(object):
    """A class to perform goodness-of-fit analysis for quasar observational data.

    :param base_dir: Base directory of the project.
    :type base_dir: str
    """

    def __init__(self, base_dir):
        """Initializes the GoodnessOfFit class with the base directory.

        :param base_dir: Base directory for dynamic path construction.
        :type base_dir: str
        """
        self.data_loader = ObsQuasarData(base_dir)
        self.data_loader.load_data()
        self.data_loader.process_civ_data()
        self.data_loader.process_mgii_data()

        self.dist_calc = LuminosityDistanceCalculator(H0=70, c=299792.458, conversion_factor=3.08567758e24)

        self.log_L_1350_norm = self.data_loader.log_L_1350 - np.log10(1e44)
        self.log_τ = self.data_loader.log_τ
        self.log_τ_3000 = self.data_loader.log_τ_3000

        z_3000 = self.data_loader.z_3000
        dL_3000_cm = self.dist_calc.d_L(z_3000, Om=0.3, Ok=0, model="lcdm")

        # Manually calculating monochromatic luminosity for MgII quasars assuming flat lcdm
        self.log_L_3000 = self.data_loader.log_F_3000 + np.log10(4 * np.pi) + 2 * np.log10(dL_3000_cm)
        self.log_L_3000_norm = self.log_L_3000 - np.log10(1e44)
        self.log_err_L_3000 = np.sqrt(np.square(np.array(self.data_loader.σ_F3000, dtype=float)))

        # Calculate symmetrized errors for both C IV and Mg II
        self.sigma_tau_1350 = self.symmetrized_error(self.data_loader.σ_Lower, self.data_loader.σ_Upper)
        self.sigma_tau_3000 = self.symmetrized_error(self.data_loader.σ_Lower3000, self.data_loader.σ_Upper3000)
        self.sigma_tau_1350_log = self.sigma_tau_1350 / (np.log(10) * self.data_loader.τ)
        self.sigma_tau_3000_log = self.sigma_tau_3000 / (np.log(10) * self.data_loader.τ_3000)

        # Calculate the asymmetric time-delay errors for C IV and Mg II
        self.fit_curve()
        self.calculate_asymmetric_errors()

        self.beta_1350 = None
        self.beta_3000 = None
        self.gamma_1350 = None
        self.gamma_3000 = None
        self.beta_sym_1350 = None
        self.gamma_sym_1350 = None
        self.beta_sym_3000 = None
        self.gamma_sym_3000 = None

    def symmetrized_error(self, lower_error, upper_error):
        """Calculate symmetrized error for asymmetric errors."""
        return 0.5 * ((2 * lower_error * upper_error) / (upper_error + lower_error) + np.sqrt(lower_error * upper_error))

    def power_law_model(self, x, beta, gamma):
        """Defines the power-law model function.

        :param x: Independent variable (normalized log luminosity).
        :type x: np.ndarray
        :param beta: Intercept parameter.
        :type beta: float
        :param gamma: Slope parameter.
        :type gamma: float
        :return: Predicted values using the power-law model.
        :rtype: np.ndarray
        """
        return beta + gamma * x

    def fit_curve(self):
        """Performs curve fitting using the power-law model and stores the results
        internally."""
        # Fit C IV data (asymmetrical)
        popt_1350, pcov_1350 = curve_fit(self.power_law_model, self.log_L_1350_norm, self.log_τ)
        self.beta_1350, self.gamma_1350 = popt_1350
        self.beta_std_1350 = np.sqrt(pcov_1350[0, 0])
        self.gamma_std_1350 = np.sqrt(pcov_1350[1, 1])

        # Fit C IV data (symmetrical)
        popt_sym_1350, pcov_sym_1350 = curve_fit(self.power_law_model, self.log_L_1350_norm, self.log_τ, sigma=self.sigma_tau_1350_log)
        self.beta_sym_1350, self.gamma_sym_1350 = popt_sym_1350
        self.beta_sym_std_1350 = np.sqrt(pcov_sym_1350[0, 0])
        self.gamma_sym_std_1350 = np.sqrt(pcov_sym_1350[1, 1])

        # Fit Mg II data (asymmetrical)
        popt_3000, pcov_3000 = curve_fit(self.power_law_model, self.log_L_3000_norm, self.log_τ_3000)
        self.beta_3000, self.gamma_3000 = popt_3000
        self.beta_std_3000 = np.sqrt(pcov_3000[0, 0])
        self.gamma_std_3000 = np.sqrt(pcov_3000[1, 1])

        # Fit Mg II data (symmetrical)
        popt_sym_3000, pcov_sym_3000 = curve_fit(self.power_law_model, self.log_L_3000_norm, self.log_τ_3000, sigma=self.sigma_tau_3000_log)
        self.beta_sym_3000, self.gamma_sym_3000 = popt_sym_3000
        self.beta_sym_std_3000 = np.sqrt(pcov_sym_3000[0, 0])
        self.gamma_sym_std_3000 = np.sqrt(pcov_sym_3000[1, 1])

    def calculate_asymmetric_errors(self):
        """Calculates the asymmetric time-delay errors for C IV and Mg II and converts
        them to log(τ) errors."""
        y_pred_1350 = self.power_law_model(self.log_L_1350_norm, self.beta_1350, self.gamma_1350)
        y_pred_3000 = self.power_law_model(self.log_L_3000_norm, self.beta_3000, self.gamma_3000)

        # Calculate the asymmetric time-delay errors for C IV and Mg II
        self.sigma_tau_1350_asym = np.where(self.log_τ < y_pred_1350, self.data_loader.σ_Lower, self.data_loader.σ_Upper)
        self.sigma_tau_3000_asym = np.where(self.log_τ_3000 < y_pred_3000, self.data_loader.σ_Lower3000, self.data_loader.σ_Upper3000)
        self.sigma_tau_1350_log_asym = self.sigma_tau_1350_asym / (np.log(10) * self.data_loader.τ)
        self.sigma_tau_3000_log_asum = self.sigma_tau_3000_asym / (np.log(10) * self.data_loader.τ_3000)

    def calculate_goodness_of_fit(self):
        """
        Calculates goodness of fit metrics: intrinsic scatter and degrees of freedom.

        :return: Intrinsic scatter and degrees of freedom for both C IV and Mg II data.
        :rtype: dict
        """

        if self.beta_1350 is None or self.beta_3000 is None:
            self.fit_curve()

        # Calculate predicted values and residuals for C IV data (asymmetrical)
        y_pred_1350 = self.power_law_model(self.log_L_1350_norm, self.beta_1350, self.gamma_1350)
        residuals_1350 = self.log_τ - y_pred_1350
        intrinsic_scatter_1350 = np.std(residuals_1350)

        # Calculate predicted values and residuals for C IV data (symmetrical)
        y_pred_sym_1350 = self.power_law_model(self.log_L_1350_norm, self.beta_sym_1350, self.gamma_sym_1350)
        residuals_sym_1350 = self.log_τ - y_pred_sym_1350
        intrinsic_scatter_sym_1350 = np.std(residuals_sym_1350)

        # Calculate predicted values and residuals for Mg II data (asymmetrical)
        y_pred_3000 = self.power_law_model(self.log_L_3000_norm, self.beta_3000, self.gamma_3000)
        residuals_3000 = self.log_τ_3000 - y_pred_3000
        intrinsic_scatter_3000 = np.std(residuals_3000)

        # Calculate predicted values and residuals for Mg II data (symmetrical)
        y_pred_sym_3000 = self.power_law_model(self.log_L_3000_norm, self.beta_sym_3000, self.gamma_sym_3000)
        residuals_sym_3000 = self.log_τ_3000 - y_pred_sym_3000
        intrinsic_scatter_sym_3000 = np.std(residuals_sym_3000)

        return {
            "C IV (asymmetrical)": {"intrinsic_scatter": intrinsic_scatter_1350},
            "C IV (symmetrical)": {"intrinsic_scatter": intrinsic_scatter_sym_1350},
            "Mg II (asymmetrical)": {"intrinsic_scatter": intrinsic_scatter_3000},
            "Mg II (symmetrical)": {"intrinsic_scatter": intrinsic_scatter_sym_3000},
        }

    def print_results(self):
        """Prints the estimated parameters and goodness of fit metrics."""
        if self.beta_1350 is None or self.beta_3000 is None:
            self.fit_curve()

        goodness_of_fit = self.calculate_goodness_of_fit()

        print("\nEstimated parameters and goodness of fit for quasar dataset (C IV, asymmetrical):")
        print("beta = {:.2f} +/- {:.2f}".format(self.beta_1350, self.beta_std_1350))
        print("gamma = {:.2f} +/- {:.2f}".format(self.gamma_1350, self.gamma_std_1350))
        print("Intrinsic scatter = {:.2f}".format(goodness_of_fit["C IV (asymmetrical)"]["intrinsic_scatter"]))

        print("\nEstimated parameters and goodness of fit for quasar dataset (C IV, symmetrical):")
        print("beta = {:.2f} +/- {:.2f}".format(self.beta_sym_1350, self.beta_sym_std_1350))
        print("gamma = {:.2f} +/- {:.2f}".format(self.gamma_sym_1350, self.gamma_sym_std_1350))
        print("Intrinsic scatter = {:.2f}".format(goodness_of_fit["C IV (symmetrical)"]["intrinsic_scatter"]))

        print("\nEstimated parameters and goodness of fit for quasar dataset (Mg II, asymmetrical):")
        print("beta = {:.2f} +/- {:.2f}".format(self.beta_3000, self.beta_std_3000))
        print("gamma = {:.2f} +/- {:.2f}".format(self.gamma_3000, self.gamma_std_3000))
        print("Intrinsic scatter = {:.2f}".format(goodness_of_fit["Mg II (asymmetrical)"]["intrinsic_scatter"]))

        print("\nEstimated parameters and goodness of fit for quasar dataset (Mg II, symmetrical):")
        print("beta = {:.2f} +/- {:.2f}".format(self.beta_sym_3000, self.beta_sym_std_3000))
        print("gamma = {:.2f} +/- {:.2f}".format(self.gamma_sym_3000, self.gamma_sym_std_3000))
        print("Intrinsic scatter = {:.2f}".format(goodness_of_fit["Mg II (symmetrical)"]["intrinsic_scatter"]))
