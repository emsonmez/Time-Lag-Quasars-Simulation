from data.observational.reading_data import ObsQuasarData
import numpy as np
import os
from unittest.mock import patch
from io import StringIO
import pytest

class TestObsQuasarData:
    """Class to test the ObsQuasarData class."""

    def setup_method(self):
        """Setup method to initialize the ObsQuasarData class."""
        self.base_dir = os.path.dirname(__file__)
        self.quasar_data = ObsQuasarData(self.base_dir)
        self.file_path_civ = self.quasar_data.get_file_path("data/observational/CIV_Cao_et_al_2022.txt")
        self.file_path_mgii = self.quasar_data.get_file_path("data/observational/MgII_Khadka_et_al_2021.txt")

        # Initialize mock data
        self.civ_data_mock = [
            ['NGC4395', '0.001064', '−11.4848', '0.0272', '39.9112', '0.0272', '0.040', '0.018', '0.024']
        ]
        self.mgii_data_mock = [
            ['018', '0.848', '−13.1412', '0.0009', '125', '7.0', '6.8']
        ]

    def test_get_file_path(self):
        relative_path_civ = "data/observational/CIV_Cao_et_al_2022.txt"
        relative_path_mgii = "data/observational/MgII_Khadka_et_al_2021.txt"

        expected_path_civ = relative_path_civ
        expected_path_mgii = relative_path_mgii

        assert self.quasar_data.get_file_path(relative_path_civ) == expected_path_civ, \
            f"Expected path '{expected_path_civ}' but got '{self.quasar_data.get_file_path(relative_path_civ)}'"
        
        assert self.quasar_data.get_file_path(relative_path_mgii) == expected_path_mgii, \
            f"Expected path '{expected_path_mgii}' but got '{self.quasar_data.get_file_path(relative_path_mgii)}'"

    def test_load_data(self, mocker):
        mocker.patch.object(self.quasar_data, 'load_file', side_effect=[
            self.civ_data_mock,
            self.mgii_data_mock
        ])

        self.quasar_data.load_data()

        assert self.quasar_data.civ_data == self.civ_data_mock, \
            f"Expected civ_data to be {self.civ_data_mock}, but got {self.quasar_data.civ_data}"

        assert self.quasar_data.mgii_data == self.mgii_data_mock, \
            f"Expected mgii_data to be {self.mgii_data_mock}, but got {self.quasar_data.mgii_data}"
    
    def test_load_file(self):
        civ_data_content = [
            "Object z log_F_1350 σ_F log_L_1350 σ_L 𝜏 σ_Lower σ_Upper\n",
            "NGC4395 0.001064 −11.4848 0.0272 39.9112 0.0272 0.040 0.018 0.024\n",
            "NGC3783 0.00973 −9.7341 0.0918 43.5899 0.0918 3.80 0.9\n"  # Incorrect number of columns
        ]
        mgii_data_content = [
            "Object z log_F_3000 σ_F3000 𝜏_3000 σ_Lower3000 σ_Upper3000\n",
            "018 0.848 −13.1412 0.0009 125 7.0 6.8\n",
            "J214355 2.607 −11.7786 0.0485 46.9624 0.0485 136 90 100\n"  # Incorrect number of columns
        ]

        # Mocking the open function to return mocked file content
        with patch('builtins.open') as mock_open:
            mock_open.side_effect = [StringIO(''.join(civ_data_content)), StringIO(''.join(mgii_data_content))]
            data_civ = self.quasar_data.load_file(self.file_path_civ, expected_columns=9)
            assert len(data_civ) == 1 
            assert all(len(row) == 9 for row in data_civ)

            data_mgii = self.quasar_data.load_file(self.file_path_mgii, expected_columns=7)
            assert len(data_mgii) == 1 
            assert all(len(row) == 7 for row in data_mgii)

        # Verify that the print statements are captured
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            with pytest.raises(SystemExit):
                self.quasar_data.load_file(r"tests\test_data\test_observational\nonexistent_file.txt", expected_columns=9)
                assert "No valid data found in the file" in mock_stdout.getvalue()
    
    def test_process_data_and_related_functions(self, mocker):
        """
        Test function for process_data, process_civ_data, process_mgii_data, and convert_to_float_array methods.

        This test function specifically tests the following methods:
        - process_data: Overall processing of C IV and Mg II data.
        - process_civ_data: Processing specifically for C IV data.
        - process_mgii_data: Processing specifically for Mg II data.
        - convert_to_float_array: Conversion utility for converting string data to float arrays.

        Each method is tested in isolation to ensure correct functionality.
        """

        mocker.patch.object(self.quasar_data, 'civ_data', self.civ_data_mock)
        mocker.patch.object(self.quasar_data, 'mgii_data', self.mgii_data_mock)

        self.quasar_data.process_data()

        # Assertions for processed civ_data attributes
        assert (self.quasar_data.Object == np.array(['NGC4395'])).all(), \
            f"Expected Object for civ_data to be ['NGC4395'], but got {self.quasar_data.Object}"
        
        assert (self.quasar_data.z == np.array([0.001064])).all(), \
            f"Expected z for civ_data to be [0.001064], but got {self.quasar_data.z}"

        assert (self.quasar_data.log_F_1350 == np.array([-11.4848])).all(), \
            f"Expected log_F_1350 for civ_data to be [-11.4848], but got {self.quasar_data.log_F_1350}"

        assert (self.quasar_data.σ_F == np.array([0.0272])).all(), \
            f"Expected σ_F for civ_data to be [0.0272], but got {self.quasar_data.σ_F}"

        assert (self.quasar_data.log_L_1350 == np.array([39.9112])).all(), \
            f"Expected log_L_1350 for civ_data to be [39.9112], but got {self.quasar_data.log_L_1350}"

        assert (self.quasar_data.σ_L == np.array([0.0272])).all(), \
            f"Expected σ_L for civ_data to be [0.0272], but got {self.quasar_data.σ_L}"

        assert (self.quasar_data.𝜏 == np.array([0.040])).all(), \
            f"Expected 𝜏 for civ_data to be [0.040], but got {self.quasar_data.𝜏}"

        assert (self.quasar_data.σ_Lower == np.array([0.018])).all(), \
            f"Expected σ_Lower for civ_data to be [0.018], but got {self.quasar_data.σ_Lower}"

        assert (self.quasar_data.σ_Upper == np.array([0.024])).all(), \
            f"Expected σ_Upper for civ_data to be [0.024], but got {self.quasar_data.σ_Upper}"

        assert (self.quasar_data.log_𝜏 == np.log10(np.array([0.040], dtype=np.float64))).all(), \
            f"Expected log_𝜏 for civ_data to be [log10(0.040)], but got {self.quasar_data.log_𝜏}"

        assert (self.quasar_data.log_σ_Lower == np.log10(np.abs(np.array([0.018], dtype=np.float64)))).all(), \
            f"Expected log_σ_Lower for civ_data to be [log10(0.018)], but got {self.quasar_data.log_σ_Lower}"

        assert (self.quasar_data.log_σ_Upper == np.log10(np.abs(np.array([0.024], dtype=np.float64)))).all(), \
            f"Expected log_σ_Upper for civ_data to be [log10(0.024)], but got {self.quasar_data.log_σ_Upper}"

        self.quasar_data.Object = None
        self.quasar_data.z = None
        self.quasar_data.log_F_1350 = None
        self.quasar_data.σ_F = None
        self.quasar_data.log_L_1350 = None
        self.quasar_data.σ_L = None
        self.quasar_data.𝜏 = None
        self.quasar_data.σ_Lower = None
        self.quasar_data.σ_Upper = None
        self.quasar_data.log_𝜏 = None
        self.quasar_data.log_σ_Lower = None
        self.quasar_data.log_σ_Upper = None

        self.quasar_data.process_data()

        # Assertions for processed mgii_data attributes
        assert (self.quasar_data.Object_3000 == np.array(['018'])).all(), \
            f"Expected Object_3000 for mgii_data to be ['018'], but got {self.quasar_data.Object_3000}"
        
        assert (self.quasar_data.z_3000 == np.array([0.848])).all(), \
            f"Expected z_3000 for mgii_data to be [0.848], but got {self.quasar_data.z_3000}"

        assert (self.quasar_data.log_F_3000 == np.array([-13.1412])).all(), \
            f"Expected log_F_3000 for mgii_data to be [-13.1412], but got {self.quasar_data.log_F_3000}"

        assert (self.quasar_data.σ_F3000 == np.array([0.0009])).all(), \
            f"Expected σ_F3000 for mgii_data to be [0.0009], but got {self.quasar_data.σ_F3000}"

        assert (self.quasar_data.𝜏_3000 == np.array([125])).all(), \
            f"Expected 𝜏_3000 for mgii_data to be [125], but got {self.quasar_data.𝜏_3000}"

        assert (self.quasar_data.σ_Lower3000 == np.array([7.0])).all(), \
            f"Expected σ_Lower3000 for mgii_data to be [7.0], but got {self.quasar_data.σ_Lower3000}"

        assert (self.quasar_data.σ_Upper3000 == np.array([6.8])).all(), \
            f"Expected σ_Upper3000 for mgii_data to be [6.8], but got {self.quasar_data.σ_Upper3000}"

        assert (self.quasar_data.log_𝜏_3000 == np.log10(np.array([125], dtype=np.float64))).all(), \
            f"Expected log_𝜏_3000 for mgii_data to be [log10(125)], but got {self.quasar_data.log_𝜏_3000}"

        assert (self.quasar_data.log_σ_Lower_3000 == np.log10(np.abs(np.array([7.0], dtype=np.float64)))).all(), \
            f"Expected log_σ_Lower_3000 for mgii_data to be [log10(7.0)], but got {self.quasar_data.log_σ_Lower_3000}"

        assert (self.quasar_data.log_σ_Upper_3000 == np.log10(np.abs(np.array([6.8], dtype=np.float64)))).all(), \
            f"Expected log_σ_Upper_3000 for mgii_data to be [log10(6.8)], but got {self.quasar_data.log_σ_Upper_3000}"
    
    def test_convert_float(self):
        result = self.quasar_data.convert_float('None')
        assert np.isnan(result), f"Expected NaN for 'None' input, but got {result}"

        result = self.quasar_data.convert_float('3.31')
        assert result == 3.31, f"Expected float value 3.31, but got {result}"

        result = self.quasar_data.convert_float('−1.45')
        assert result == -1.45, f"Expected float value −1.45, but got {result}"

        with pytest.raises(ValueError):
            self.quasar_data.convert_float('invalid')