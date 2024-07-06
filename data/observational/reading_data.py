import os
import numpy as np
import re

class ObsQuasarData:
    """
    A class to read and process quasar observational data.

    :param base_dir: Base directory of the project.
    :type base_dir: str
    """
    def __init__(self, base_dir):
        """
        Initializes the ObsQuasarData class with the base directory.

        :param base_dir: Base directory of the project.
        :type base_dir: str
        """
        self.base_dir = base_dir
        self.civ_data = None
        self.mgii_data = None
        self.civ_file_path = self.get_file_path("data/observational/CIV_Cao_et_al_2022.txt")
        self.mgii_file_path = self.get_file_path("data/observational/MgII_Khadka_et_al_2021.txt")

    def get_file_path(self, relative_path):
        """
        Constructs the dynamic path to the data file.

        :param relative_path: Relative path to the data file.
        :type relative_path: str
        :return: Absolute path to the data file.
        :rtype: str
        """
        base_path = os.path.dirname(__file__)
        return base_path.replace(base_path, relative_path)
    
    def load_data(self):
        """Loads the C IV and Mg II data from the specified files."""
        self.civ_data = self.load_file(self.civ_file_path, expected_columns=9)
        self.mgii_data = self.load_file(self.mgii_file_path, expected_columns=7)

    def load_file(self, file_path, expected_columns):
        """
        Loads data from a specified file, excluding lines with incorrect number of columns.

        :param file_path: Path to the data file.
        :type file_path: str
        :param expected_columns: Expected number of columns in the data file.
        :type expected_columns: int
        :return: Processed data.
        :rtype: list of lists
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        lines = lines[1:]  # Omitting the title names in the .txt file
        data = []

        for line in lines:
            raw_values = re.split(r'\s+', line.strip())
            values = [val for val in raw_values if val]  # Filter out any empty strings

            if len(values) == expected_columns:
                data.append(values)  # Append all columns, including the first column
            else:
                print(f"Ignoring line with incorrect number of values: {line.strip()}")
        
        if not data:
            print(f"No valid data found in the file: {file_path}")
            exit()

        return data
    
    def process_data(self):
        """Processes the loaded C IV and Mg II data."""
        self.process_civ_data()
        self.process_mgii_data()

    def process_civ_data(self):
        """Processes the C IV data."""
        Object, z, log_F_1350, σ_F, log_L_1350, σ_L, 𝜏, σ_Lower, σ_Upper = zip(*self.civ_data)

        self.Object = np.array(Object)
        self.z = np.array(z, dtype=float)
        self.log_F_1350 = self.convert_to_float_array(log_F_1350)
        self.σ_F = np.array(σ_F, dtype=float)
        self.log_L_1350 = self.convert_to_float_array(log_L_1350)
        self.σ_L = np.array(σ_L, dtype=float)
        self.𝜏 = np.array(𝜏, dtype=np.float64)
        self.σ_Lower = np.array(σ_Lower, dtype=np.float64)
        self.σ_Upper = np.array(σ_Upper, dtype=np.float64)
        self.log_𝜏 = np.log10(np.array(self.𝜏, dtype=np.float64))
        self.log_σ_Lower = np.log10(np.abs(np.array(self.σ_Lower, dtype=np.float64)))
        self.log_σ_Upper = np.log10(np.abs(np.array(self.σ_Upper, dtype=np.float64)))

    def process_mgii_data(self):
        """Processes the Mg II data."""
        Object_3000, z_3000, log_F_3000, σ_F3000, 𝜏_3000, σ_Lower3000, σ_Upper3000 = zip(*self.mgii_data)

        self.Object_3000 = np.array(Object_3000)
        self.z_3000 = np.array(z_3000, dtype=float)
        self.log_F_3000 = self.convert_to_float_array(log_F_3000)
        self.σ_F3000 = np.array(σ_F3000, dtype=float)
        self.𝜏_3000 = np.array(𝜏_3000, dtype=np.float64)
        self.σ_Lower3000 = np.array(σ_Lower3000, dtype=np.float64)
        self.σ_Upper3000 = np.array(σ_Upper3000, dtype=np.float64)
        self.log_𝜏_3000 = np.log10(np.array(self.𝜏_3000, dtype=np.float64))
        self.log_σ_Lower_3000 = np.log10(np.abs(np.array(self.σ_Lower3000, dtype=np.float64)))
        self.log_σ_Upper_3000 = np.log10(np.abs(np.array(self.σ_Upper3000, dtype=np.float64)))

    def convert_to_float_array(self, data):
        """
        Converts a list of string values to a numpy array of floats, handling 'None' values.

        :param data: List of string values.
        :type data: list of str
        :return: Numpy array of floats.
        :rtype: np.ndarray
        """
        return np.vectorize(self.convert_float)(data)

    def convert_float(self, value):
        """
        Converts a string value to float, handling 'None' values.

        :param value: String value.
        :type value: str
        :return: Float value.
        :rtype: float
        """
        if value == 'None':
            return np.nan
        else:
            return float(value.replace('−', '-'))