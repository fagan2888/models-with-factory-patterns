import abc
import json
from os import getcwd


class AbsModelBuilder(metaclass=abc.ABCMeta):
    """The abstract class to inherit for creating your model builder.

    Attributes:
        _X_train: the X_train of your dataset
        _y_train: the y_train of your dataset
        _X_test: the X_test of your dataset
        _y_test: the y_test of your dataset
        _models: the dictionary that stores your models
        _models_info: the dictionary that stores the models info loaded from your model_json
        _gridsearchcv_info: the dictionary that stores the grid search info loaded from your gridsearchcv_json
    """

    def __init__(self, model_json, gridsearchcv_json):
        """Constructor for ModeiBuilder class

        Loads the information from both json files.

        model_json format example: { "LogisticRegression" : {"penalty": [ "l1", "l2"]},
                                    "RandomForestClassifier" : {"max_depth": [2,3,4]} }

        gridsearchcv_json format example: { "LogisticRegression" : {"cv": 5, "scoring" : "roc_auc"},
                                            "RandomForestClassifier" : {"cv": 5, "scoring" : "roc_auc"} }

        For more info about gridsearchcv, please visit:
        http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

        Args:
            model_json: the json file that contains the info of the models you wish to load and use.
            gridsearchcv_json: the json file that contains the info of the grid search you will be using.

        Returns:
            None
        """

        self._X_train = []
        self._y_train = []
        self._X_test = []
        self._y_test = []
        self._models = {}
        with open(getcwd() + "\\" + model_json, 'r') as f:
            self._models_info = json.load(f)

        with open(getcwd() + "\\" + gridsearchcv_json, 'r') as f:
            self._gridsearchcv_info = json.load(f)

    @abc.abstractmethod
    def build_models(self):
        """Abstract function to be implemented for building your models"""

        pass

    def set_data(self, x_train, x_test, y_train, y_test):
        """Sets the X_train, X_test, y_train, y_test used for the modelling."""

        self._X_train = x_train
        self._y_train = y_train
        self._X_test = x_test
        self._y_test = y_test

    def fit_all(self):
        """Runs all the fit functions of each individual model."""

        for model in self._models:
            print('-' * 20 + model + '-' * 20)
            print(self._models[model].fit(self._X_train, self._y_train))

    def run(self, x_train, x_test, y_train, y_test):
        """Builds the models, sets the data(x_train, x_test, y_train, y_test), then fits the models."""

        self.build_models()
        self.set_data(x_train, x_test, y_train, y_test)
        self.fit_all()

    def score_reports(self):
        """Prints all the scores of your models. The scorer methods can be set in your gridsearchcv.json."""

        for model in self._models:
            print(model + ": " + str(self._models[model].best_score_))