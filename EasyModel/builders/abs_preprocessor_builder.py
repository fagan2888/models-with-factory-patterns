import abc
import json
from pandas import read_csv
from sklearn.model_selection import train_test_split
from os import getcwd


class AbsPreprocessorBuilder(metaclass=abc.ABCMeta):

    def __init__(self, parameters_json, data):
        self._X = read_csv(data)
        self._y = []
        self._preprocessors = {}
        with open(getcwd() + "\\" + parameters_json, 'r') as f:
            self._preprocessors_info = json.load(f)

    @abc.abstractmethod
    def compute_preprocessors(self):
        pass

    def train_test_split(self, **parameters):
        return train_test_split(self._X, self._y, **locals()["parameters"])

    @property
    def get_data(self):
        return self._X, self._y