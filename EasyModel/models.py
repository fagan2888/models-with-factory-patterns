from EasyModel.director import Director
from sklearn.metrics import classification_report, precision_recall_curve
from EasyModel.factories.loader import load_classifier
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt


class EasyClassifiers(Director):

    def build_models(self):
        print("building models")
        for entry in self._models_info:
            model = load_classifier(entry)
            model = model.create_model(self._models_info[entry], self._gridsearchcv_info[entry])
            self._models[entry] = model

    def classification_reports(self):
        for model in self._models:
            print('-'*20  + model + '-'*20)
            print(classification_report(self._y_test, self._models[model].predict(self._X_test)))

    def score_reports(self):
        for model in self._models:
            print(model + ": " + str(self._models[model].best_score_))

    def feature_inspect(self, max_feature=10):
        for model in self._models:
            if hasattr(self._models[model].best_estimator_, 'feature_importances_'):
                fi = pd.DataFrame({'feature': self._X_train.columns,
                                   'importance': self._models[model].best_estimator_.feature_importances_})

                fi.sort_values('importance', ascending=False, inplace=True)
                print('-' * 20 + model + '-' * 20)
                print(fi.head(max_feature))

            elif hasattr(self._models[model].best_estimator_, 'coef_'):
                coefs_vars = pd.DataFrame({
                    'coef': self._models[model].best_estimator_.coef_[0],
                    'variable': self._X_train.columns,
                    'abscoef': np.abs(self._models[model].best_estimator_.coef_[0])
                })
                coefs_vars.sort_values('abscoef', ascending=False, inplace=True)
                print('-' * 20 + model + '-' * 20)
                print(coefs_vars.head(max_feature))

            else:
                print('-' * 20 + model + '-' * 20)
                print(model + " does not support feature coefficient or importance")

    def decision_threshold(self, target_col = 1, categories = [0, 1]):

        def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, model_name):
            """
            Modified from:
            Hands-On Machine learning with Scikit-Learn
            and TensorFlow; p.89
            """
            plt.figure(figsize=(6, 6))
            plt.title("Precision and Recall Scores of " + model_name)
            plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
            plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
            plt.ylabel("Score")
            plt.xlabel("Decision Threshold")
            plt.legend(loc='best')

        y_test_roc = label_binarize(self._y_test, classes=categories)
        for model in self._models:
            if hasattr(self._models[model], 'predict_proba'):
                y_scores = self._models[model].predict_proba(self._X_test)[:, target_col]
                p, r, thresholds = precision_recall_curve(y_test_roc, y_scores)
                plot_precision_recall_vs_threshold(p, r, thresholds, model)
