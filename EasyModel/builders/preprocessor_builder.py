from .abs_preprocessor_builder import AbsPreprocessorBuilder
from EasyModel.factories.loader import load_preprocessor


class EasyPreprocessors(AbsPreprocessorBuilder):
    """The class which contains all your preprocessors loaded from your json file.

    This class provides model evaluations such as classification_reports, feature_inspect, decision_threshold,
    permutation_importance, partial_dependence_plot, and shap_analysis.

    Attributes:
        _X: the core of your dataset
        _y: the target variable of your dataset
        _preprocessors: the dictionary that stores all the loaded preprocessors
        _preprocessors_info: the dictionary that stores the info of the preprocessors loaded from parameters_json

    Refer to the parent class for these attribute instantiations.
    """

    def compute_preprocessors(self):
        """Compute each preprocessors"""

        print("starting preprocessors")
        for entry in self._preprocessors_info:
            # "target_variable"  requires a different treatment as it involves splitting to x and y.
            if "target_variable" == entry:
                print("computing target_variable")
                self._y = self._X[self._preprocessors_info["target_variable"]["columns"]]
                self._X.drop([self._preprocessors_info["target_variable"]["columns"][0]], axis=1, inplace=True)

            else:
                preprocessor = load_preprocessor(entry, self._preprocessors_info[entry]["parameters"])
                self._X = preprocessor.compute(self._X, self._preprocessors_info[entry]["columns"])
