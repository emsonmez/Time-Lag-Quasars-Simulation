from reading_data import ObsQuasarData
import numpy as np
from scipy.optimize import curve_fit
from scipy import integrate

class GoodnessOfFit(object):
    """
    A class to perform goodness-of-fit analysis for quasar observational data.

    :param base_dir: Base directory of the project.
    :type base_dir: str
    """

    def __init__(self, base_dir):
        """
        Initializes the GoodnessOfFit class with the base directory.

        :param base_dir: Base directory of the project.
        :type base_dir: str
        """
        self.data_loader = ObsQuasarData(base_dir)
        self.data_loader.load_data()
        self.data_loader.process_civ_data()
        self.data_loader.process_mgii_data()

        self.log_L_1350_norm = self.data_loader.log_L_1350 - np.log10(1e44)
        self.log_𝜏 = self.data_loader.log_𝜏
        self.log_𝜏_3000 = self.data_loader.log_𝜏_3000 

        c = 299792.458  # speed of light; km/s
        Om = 0.3
        H0 = 70

        def integrand(z, H0, Om):
            Ol = 1 - Om
            return 1 / (H0 * np.sqrt(Om * (1 + z) ** 3 + Ol))

        integrand_vec = np.vectorize(integrand)

        def d_L(z, Om, H0):
            d_c, _ = integrate.quad(integrand_vec, 0, z, args=(H0, Om))
            return (1 + z) * d_c * c

        z_3000 = self.data_loader.z_3000  
        dL_3000_Mpc = np.array([d_L(z, Om, H0) for z in z_3000])

        conversion_factor = 3.08567758e24  # 1 Mpc = 3.08567758 × 10^24 cm
        dL_3000_cm = dL_3000_Mpc * conversion_factor

        self.log_L_3000 = self.data_loader.log_F_3000 + np.log10(4 * np.pi) + 2 * np.log10(dL_3000_cm)
        self.log_L_3000_norm = self.log_L_3000 - np.log10(1e44)
        self.log_err_L_3000 = np.sqrt(np.square(np.array(self.data_loader.σ_F3000, dtype=float)))

        self.beta_1350 = None
        self.gamma_1350 = None
        self.beta_std_1350 = None
        self.gamma_std_1350 = None
        self.beta_3000 = None
        self.gamma_3000 = None
        self.beta_std_3000 = None
        self.gamma_std_3000 = None

    def power_law_model(self, x, beta, gamma):
        """
        Defines the power-law model function.

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

    def power_law_model(self, x, beta, gamma):
        """
        Defines the power-law model function.

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
        """
        Performs curve fitting using the power-law model and stores the results internally.
        """
        # Fit C IV data
        popt_1350, pcov_1350 = curve_fit(self.power_law_model, self.log_L_1350_norm, self.log_τ)
        self.beta_1350, self.gamma_1350 = popt_1350
        self.beta_std_1350 = np.sqrt(pcov_1350[0, 0])
        self.gamma_std_1350 = np.sqrt(pcov_1350[1, 1])

        # Fit Mg II data
        popt_3000, pcov_3000 = curve_fit(self.power_law_model, self.log_L_3000_norm, self.log_𝜏_3000)
        self.beta_3000, self.gamma_3000 = popt_3000
        self.beta_std_3000 = np.sqrt(pcov_3000[0, 0])
        self.gamma_std_3000 = np.sqrt(pcov_3000[1, 1])

    def calculate_goodness_of_fit(self):
        """
        Calculates goodness of fit metrics: intrinsic scatter and degrees of freedom.

        :return: Intrinsic scatter and degrees of freedom for both C IV and Mg II data.
        :rtype: dict
        """
        # Fit the curves if not already done
        if self.beta_1350 is None or self.beta_3000 is None:
            self.fit_curve()

        # Calculate predicted values and residuals for C IV data
        y_pred_1350 = self.power_law_model(self.log_L_1350_norm, self.beta_1350, self.gamma_1350)
        residuals_1350 = self.log_τ - y_pred_1350
        intrinsic_scatter_1350 = np.std(residuals_1350)
        N_1350 = len(self.log_L_1350_norm)
        degrees_of_freedom_1350 = N_1350 - 2

        # Calculate predicted values and residuals for Mg II data
        y_pred_3000 = self.power_law_model(self.log_L_3000_norm, self.beta_3000, self.gamma_3000)
        residuals_3000 = self.log_𝜏_3000 - y_pred_3000
        intrinsic_scatter_3000 = np.std(residuals_3000)
        N_3000 = len(self.log_L_3000_norm)
        degrees_of_freedom_3000 = N_3000 - 2

        return {
            "C IV": {
                "intrinsic_scatter": intrinsic_scatter_1350,
                "degrees_of_freedom": degrees_of_freedom_1350
            },
            "Mg II": {
                "intrinsic_scatter": intrinsic_scatter_3000,
                "degrees_of_freedom": degrees_of_freedom_3000
            }
        }

    def print_results(self):
        """
        Prints the estimated parameters and goodness of fit metrics.
        """
        if self.beta_1350 is None or self.beta_3000 is None:
            self.fit_curve()

        goodness_of_fit = self.calculate_goodness_of_fit()

        print("Estimated parameters and goodness of fit for quasar dataset (C IV):")
        print("beta = {:.2f} +/- {:.2f}".format(self.beta_1350, self.beta_std_1350))
        print("gamma = {:.2f} +/- {:.2f}".format(self.gamma_1350, self.gamma_std_1350))
        print("Intrinsic scatter = {:.2f}".format(goodness_of_fit["C IV"]["intrinsic_scatter"]))
        print("Degrees of freedom = {}".format(goodness_of_fit["C IV"]["degrees_of_freedom"]))

        print("\nEstimated parameters and goodness of fit for quasar dataset (Mg II):")
        print("beta = {:.2f} +/- {:.2f}".format(self.beta_3000, self.beta_std_3000))
        print("gamma = {:.2f} +/- {:.2f}".format(self.gamma_3000, self.gamma_std_3000))
        print("Intrinsic scatter = {:.2f}".format(goodness_of_fit["Mg II"]["intrinsic_scatter"]))
        print("Degrees of freedom = {}".format(goodness_of_fit["Mg II"]["degrees_of_freedom"]))


# Example usage:
if __name__ == "__main__":
    base_dir = "/path/to/your/project"
    analysis = GoodnessOfFit(base_dir)
    analysis.print_results()


# Example usage:
if __name__ == "__main__":
    base_dir = "/path/to/your/project"
    analysis = GoodnessOfFit(base_dir)
    analysis.print_results()