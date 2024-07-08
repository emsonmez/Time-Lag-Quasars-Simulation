import numpy as np

class LuminosityDistanceCalculator(object):
    """
    A class to perform all necessary cosmological distances formulas, taking into account all models.
    """
    
    def __init__(self, H0: float = 70, c: float = 299792.458, conversion_factor: float = 3.08567758e24):
       """
        Initializes the LuminosityDistanceCalculator class with cosmological constants.

        :param H0: Hubble constant (current expansion rate of the universe) :unit: km/s/Mpc
        :type H0: float
        :param c: Speed of light :unit: km/s
        :type c: float
        :param conversion_factor: Conversion rate bewteen Megaparsecs (Mpc) and centimeters (cm) :unit: cm/Mpc
        """
       self.H0 = H0
       self.c = c
       self.conversion_factor = conversion_factor

    @staticmethod
    def integrand_lcdm(z, Om, Ok, H0):
       """
        Integrand function for Lambda-CDM model.

        :param z: Redshift
        :type z: float
        :param Om: Matter density parameter
        :type Om: float
        :param Ok: Curvature density parameter
        :type Ok: float
        :param H0: Hubble constant
        :type H0: float
        :return: Integrand value
        :rtype: float
        """
       Ol = 1 - Om - Ok
       if Ok != 0:
            return 1 / (H0 * np.sqrt(Om * (1 + z) ** 3 + Ok * (1 + z) ** 2 + Ol))
       else:
            return 1 / (H0 * np.sqrt(Om * (1 + z) ** 3 + Ol))
       
    @staticmethod
    def integrand_xcdm(z, Om, Ok, H0, w_x):
        """
        Integrand function for X-CDM model.

        :param z: Redshift
        :type z: float
        :param Om: Matter density parameter
        :type Om: float
        :param Ok: Curvature density parameter
        :type Ok: float
        :param H0: Hubble constant
        :type H0: float
        :param w_x: Equation of state parameter for dark energy
        :type w_x: float
        :return: Integrand value
        :rtype: float
        """
        Ox = 1 - Om - Ok
        if Ok != 0:
            return 1 / (H0 * np.sqrt(Om * (1 + z) ** 3 + Ok * (1 + z) ** 2 + Ox * (1 + z) ** (3 * (1 + w_x))))
        else:
            return 1 / (H0 * np.sqrt(Om * (1 + z) ** 3 + Ox * (1 + z) ** (3 * (1 + w_x))))
    
    def d_L(self, z, Om, Ok, model='lcdm', w_x=-1):
        """
        Calculate the luminosity distance for given redshift values and cosmological parameters.

        :param z: Array of redshift values
        :type z: np.ndarray
        :param Om: Matter density parameter
        :type Om: float
        :param Ok: Curvature density parameter
        :type Ok: float
        :param model: Cosmological model ('lcdm' or 'xcdm')
        :type model: str
        :param w_x: Equation of state parameter for dark energy (only used for xcdm)
        :type w_x: float
        :return: Array of luminosity distances in cm
        :rtype: np.ndarray
        """
        num_points = len(z)
        integral_values = np.zeros(num_points)

        if model == 'lcdm':
            integrand_func = self.integrand_lcdm
            args = (Om, Ok, self.H0)
        elif model == 'xcdm':
            integrand_func = self.integrand_xcdm
            args = (Om, Ok, self.H0, w_x)
        else:
            raise ValueError("Model must be 'lcdm' or 'xcdm'")

        for i in range(num_points):
            redshift = z[i]
            integrand_values = np.array([integrand_func(x, *args) for x in np.linspace(0, redshift, 1000)])
            integral_values[i] = np.trapz(integrand_values, np.linspace(0, redshift, 1000))

        if Ok > 0:
            return (self.c * (1 + z) / (self.H0 * np.sqrt(Ok))) * np.sinh(self.H0 * np.sqrt(Ok) * integral_values) * self.conversion_factor
        elif Ok < 0:
            return (self.c * (1 + z) / (self.H0 * np.sqrt(abs(Ok)))) * np.sin(self.H0 * np.sqrt(abs(Ok)) * integral_values) * self.conversion_factor
        else:
            return (1 + z) * self.c * integral_values * self.conversion_factor