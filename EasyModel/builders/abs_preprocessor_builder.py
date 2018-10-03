import abc
import json
from pandas import read_csv
from sklearn.model_selection import train_test_split
from os import getcwd


class AbsPreprocessorBuilder(metaclass=abc.ABCMeta):
    """The abstract class to inherit for creating your preprocessor builder.

    Attributes:
        _X: the core of your dataset
        _y: the target variable of your dataset
        _preprocessors: the dictionary that stores all the loaded preprocessors
        _preprocessors_info: the dictionary that stores the info of the preprocessors loaded from parameters_json
    """

    def __init__(self, parameters_json, data):
        """Constructor of PreprocessorBuilder

        Loads the CSV data. Loads the preprocessor info from parameter_json into _preprocessors_info.

        parameter_json format example: { "drop_columns" : {"parameters": {}, "columns" : ["User ID"]},
                                         "target_variable" : {"parameters": {}, "columns" : ["Purchased"]} }

        Args:
            parameters_json: the json file that contains your preprocessor setup
            data: your data in csv format

        Returns:
        """

        self._X = read_csv(data)
        self._y = []
        self._preprocessors = {}
        with open(getcwd() + "\\" + parameters_json, 'r') as f:
            self._preprocessors_info = json.load(f)

    @abc.abstractmethod
    def compute_preprocessors(self):
        """Abstract function to be implemented for building your preprocessors"""

        pass

    def train_test_split(self, **parameters):
        """Function that uses sklearn.model_selection.train_test_split

        Args:
            parameters: sklearn.model_selection.train_test_split's parameters

        Returns:
            returns X_train, X_test, y_train, y_test
        """

        return train_test_split(self._X, self._y, **locals()["parameters"])

    @property
    def get_data(self):
        return self._X, self._y