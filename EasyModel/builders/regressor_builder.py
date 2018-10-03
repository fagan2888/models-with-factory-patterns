from .abs_model_builder import AbsModelBuilder
from EasyModel.factories.loader import load_regressor
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class EasyRegressors(AbsModelBuilder):
    """The class which contains all your regressors loaded from your json file.

    This class provides model evaluations such as plot_results(y_true against y_pred) and feature_inspect. It also
    provide residual checks for regression.

    Attributes:
        _X_train: the X_train of your dataset
        _y_train: the y_train of your dataset
        _X_test: the X_test of your dataset
        _y_test: the y_test of your dataset
        _models: the dictionary that stores your models
        _models_info: the dictionary that stores the models info loaded from your model_json
        _gridsearchcv_info: the dictionary that stores the grid search info loaded from your gridsearchcv_json

    Refer to the parent class for these attribute instantiations.
    """

    def build_models(self):
        """Load and build regressors. Loaded regressors can be found in _models."""

        print("building models")
        for entry in self._models_info:
            model = load_regressor(entry)
            model = model.create_model(self._models_info[entry], self._gridsearchcv_info[entry])
            self._models[entry] = model

    def plot_results(self):
        """Plot y_true against y_pred for each model."""

        for model in self._models:
            data = pd.DataFrame()
            data['y_true'] = self._y_test.iloc[:, 0]
            data['y_pred'] = self._models[model].predict(self._X_test)
            sns.lmplot(x='y_true', y='y_pred', data=data)
            ax = plt.gca()
            ax.set_title(model)

    def feature_inspect(self, max_feature=10):
        """Prints the feature coefficients or importances in each models

        Args:
            max_feature: the number of features to be displayed

        Returns:
            None
        """

        for model in self._models:
            # for models that support feature coefficients
            if hasattr(self._models[model].best_estimator_, 'coef_'):
                # Doing this because LinearRegression's coef is an array within an array,
                if len(self._models[model].best_estimator_.coef_) == 1:
                    coef = self._models[model].best_estimator_.coef_[0]
                else:
                    coef = self._models[model].best_estimator_.coef_

                coefs_vars = pd.DataFrame({
                    'coef': coef,
                    'variable': self._X_train.columns,
                    'abscoef': np.abs(coef)
                })
                coefs_vars.sort_values('abscoef', ascending=False, inplace=True)
                print('-' * 20 + model + '-' * 20)
                print(coefs_vars.head(max_feature))

            # for models that support feature_importances
            elif hasattr(self._models[model].best_estimator_, 'feature_importances_'):
                fi = pd.DataFrame({'feature': self._X_train.columns,
                                   'importance': self._models[model].best_estimator_.feature_importances_})

                fi.sort_values('importance', ascending=False, inplace=True)
                print('-' * 20 + model + '-' * 20)
                print(fi.head(max_feature))

            else:
                print('-' * 20 + model + '-' * 20)
                print(model + " does not support feature coefficient or importance")

    def run(self, x_train, x_test, y_train, y_test, residual_diagnostic=False, significance=0.05,
            residual_tests_json=r'EasyModel\builders\residual_diagnostic\residual_tests.json'):
        """Compute the "run" function from parent class, followed by the residual checks (optional).

        model_json format example: { "shapiro" : {}, "acorr_ljungbox": {"lags": 1} }

        Args:
            x_train: X_train of your dataset
            x_test: X_test of your dataset
            y_train: y_train of your dataset
            y_test: y_test of your dataset
            residual_diagnostic: condition for performing residual tests. Default is False
            significance: significance level used in null/alternate hypothesis. Default is 0.05.
            residual_tests_json: the json file that contains your residual tests setup

        Returns:
            None
        """

        super().run(x_train, x_test, y_train, y_test)

        if residual_diagnostic is True:
            from EasyModel.builders.residual_diagnostic.residual_test_builder import ResidualTestBuilder

            for model in self._models:
                residual = self._y_test.values - self._models[model].predict(self._X_test).reshape(-1, 1)
                print('-' * 20 + model + '-' * 20)

                residual_tests = ResidualTestBuilder(residual_tests_json)
                residual_tests.compute_tests(residual, significance, x=self._X_test)
