from .abs_model_builder import AbsModelBuilder
from sklearn.metrics import classification_report, precision_recall_curve
from EasyModel.factories.loader import load_classifier
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import eli5
from eli5.sklearn import PermutationImportance
from IPython.display import display
from pdpbox import pdp, get_dataset, info_plots


class EasyClassifiers(AbsModelBuilder):

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

    def decision_threshold(self, target_col=1, categories=[0, 1]):

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
            else:
                print(model + " does not support 'predict_proba' function")

    def permutation_importance(self):
        for model in self._models:
            if model != "XGBClassifier":
                perm = PermutationImportance(self._models[model], random_state=1).fit(self._X_test, self._y_test)
                print('-' * 20 + model + '-' * 20)
                display(eli5.show_weights(perm, feature_names=self._X_test.columns.tolist()))
            else:
                print(model + " is incompatible with eli5's PermutationImportance")

    def partial_dependence_plot(self, feature):
        for model in self._models:
            # Create the data that we will plot
            pdp_goals = pdp.pdp_isolate(model=self._models[model], dataset=self._X_test,
                                        model_features=self._X_test.columns.tolist(), feature=feature)
            # plot it
            pdp.pdp_plot(pdp_goals, model + "'s " + feature)
            plt.show()