from .abs_model_builder import AbsModelBuilder
from sklearn.metrics import classification_report, precision_recall_curve
from EasyModel.factories.loader import load_classifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from IPython.display import display


class EasyClassifiers(AbsModelBuilder):
    """The class which contains all your classifiers loaded from your json file.

    This class provides model evaluations such as classification_reports, feature_inspect, decision_threshold,
    permutation_importance, partial_dependence_plot, and shap_analysis.

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
        """Load and build classifiers. Loaded classifiers can be found in _models."""

        print("building models")
        for entry in self._models_info:
            model = load_classifier(entry)
            model = model.create_model(self._models_info[entry], self._gridsearchcv_info[entry])
            self._models[entry] = model

    def classification_reports(self):
        """Prints the classification reports for each model using sklearn.metrics.classification_report."""

        for model in self._models:
            print('-'*20  + model + '-'*20)
            print(classification_report(self._y_test, self._models[model].predict(self._X_test)))

    def feature_inspect(self, max_feature=10):
        """Prints the feature coefficients or importances in each models

        Args:
            max_feature: the number of features to be displayed
        """

        for model in self._models:
            # for models that support feature_importances
            if hasattr(self._models[model].best_estimator_, 'feature_importances_'):
                fi = pd.DataFrame({'feature': self._X_train.columns,
                                   'importance': self._models[model].best_estimator_.feature_importances_})

                fi.sort_values('importance', ascending=False, inplace=True)
                print('-' * 20 + model + '-' * 20)
                print(fi.head(max_feature))

            # for models that support feature coefficients
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

    def decision_threshold(self, target_category_index=1, categories=[0, 1]):
        """Evaluates the models' thresholds based on their precisions and recalls.

        Args:
            target_category_index: the index of your target category
            categories: the categories of your target column

        """

        def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, model_name):
            """Plot precision and recall against threshold

            Modified from: Hands-On Machine learning with Scikit-Learn and TensorFlow; p.89

            Args:
                precisions: precisions derived from sklearn.metrics.precision_recall_curve
                recalls: recalls derived from sklearn.metrics.precision_recall_curve
                thresholds: thresholds derived from sklearn.metrics.precision_recall_curve
                model_name: the classifier name of this plot
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
                y_scores = self._models[model].predict_proba(self._X_test)[:, target_category_index]
                p, r, thresholds = precision_recall_curve(y_test_roc, y_scores)
                plot_precision_recall_vs_threshold(p, r, thresholds, model)
            else:
                print(model + " does not support 'predict_proba' function")

    def permutation_importance(self):
        """Computes the permutation importance of each model.

        For more info, please visit https://eli5.readthedocs.io/en/latest/ or
                                    https://www.kaggle.com/dansbecker/permutation-importance
        """

        import eli5
        from eli5.sklearn import PermutationImportance

        for model in self._models:
            # does not work with XGBClassifier currently.
            if model != "XGBClassifier":
                perm = PermutationImportance(self._models[model], random_state=1).fit(self._X_test, self._y_test)
                print('-' * 20 + model + '-' * 20)
                display(eli5.show_weights(perm, feature_names=self._X_test.columns.tolist()))
            else:
                print(model + " is incompatible with eli5's PermutationImportance")

    def partial_dependence_plot(self, feature):
        """Plots the partial dependence plot of each model.

        For more info, please visit https://pdpbox.readthedocs.io/en/latest/ or
                                    https://www.kaggle.com/dansbecker/partial-plots

        Args:
            feature: the name of the feature you wish to evaluate.
        """

        from pdpbox import pdp, get_dataset, info_plots

        for model in self._models:
            # Create the data that we will plot
            pdp_goals = pdp.pdp_isolate(model=self._models[model], dataset=self._X_test,
                                        model_features=self._X_test.columns.tolist(), feature=feature)
            # plot it
            pdp.pdp_plot(pdp_goals, model + "'s " + feature)
            plt.show()

    def shap_analysis(self, model, feature_col=1, nsamples=100):
        """Evaluates the dataset using shap values

        For more info, please visit https://www.kaggle.com/dansbecker/shap-values or
                                    https://github.com/slundberg/shap
        Args:
            model: name of the model to be used. (currently only supports logistic regression and random forest).
            feature_col:
            nsamples: used in KernelExplainer. Default is 100.
        """

        import shap
        shap.initjs()

        if model in ["LogisticRegression"]:
            explainer = shap.KernelExplainer(self._models[model].predict_proba, self._X_train)
            shap_values = explainer.shap_values(self._X_test , nsamples=nsamples)

        elif model in ["RandomForestClassifier"]:
            explainer = shap.TreeExplainer(self._models[model].best_estimator_)
            shap_values = explainer.shap_values(self._X_test)
        else:
            print(model + " does not work in shap_analysis currently.")
            return

        display(shap.force_plot(explainer.expected_value[feature_col], shap_values[feature_col], self._X_test))
        display(shap.summary_plot(shap_values, self._X_test))
