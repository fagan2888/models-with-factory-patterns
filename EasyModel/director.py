import abc
import json
from os import getcwd


class Director(metaclass=abc.ABCMeta):

    def __init__(self, model_json, gridsearchcv_json):
        self._X_train = []
        self._y_train = []
        self._X_test = []
        self._y_test = []
        self._models = {}
        with open(getcwd() + "/" + model_json, 'r') as f:
            self._models_info = json.load(f)

        with open(getcwd() + "/" + gridsearchcv_json, 'r') as f:
            self._gridsearchcv_info = json.load(f)

    @abc.abstractmethod
    def build_models(self):
        pass

    def set_data(self, x_train, x_test, y_train, y_test):
        self._X_train = x_train
        self._y_train = y_train
        self._X_test = x_test
        self._y_test = y_test

    def fit_all(self):
        for model in self._models:
            print('-' * 20 + model + '-' * 20)
            print(self._models[model].fit(self._X_train, self._y_train))

    def run(self, x_train, x_test, y_train, y_test):
        self.build_models()
        self.set_data(x_train, x_test, y_train, y_test)
        self.fit_all()

