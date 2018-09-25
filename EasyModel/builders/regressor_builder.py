from .abs_model_builder import AbsModelBuilder
from EasyModel.factories.loader import load_regressor
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class EasyRegressors(AbsModelBuilder):

    def build_models(self):
        print("building models")
        for entry in self._models_info:
            model = load_regressor(entry)
            model = model.create_model(self._models_info[entry], self._gridsearchcv_info[entry])
            self._models[entry] = model

    def plot_results(self):
        for model in self._models:
            data = pd.DataFrame()
            data['price'] = self._y_test.iloc[:, 0]
            data['y_pred'] = self._models[model].predict(self._X_test)
            sns.lmplot(x='price', y='y_pred', data=data)
            ax = plt.gca()
            ax.set_title(model)

    def feature_inspect(self, max_feature=10):
        for model in self._models:
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

        super().run(x_train, x_test, y_train, y_test)

        if residual_diagnostic is True:
            from EasyModel.builders.residual_diagnostic.residual_test_builder import ResidualTestBuilder

            for model in self._models:
                residual = self._y_test.values - self._models[model].predict(self._X_test).reshape(-1, 1)
                print('-' * 20 + model + '-' * 20)

                residual_tests = ResidualTestBuilder(residual_tests_json)
                residual_tests.compute_tests(residual, significance, x=self._X_test)
