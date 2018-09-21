from .abs_preprocessor_builder import AbsPreprocessorBuilder
from EasyModel.factories.loader import load_preprocessor


class EasyPreprocessors(AbsPreprocessorBuilder):

    def compute_preprocessors(self):

        print("starting preprocessors")
        for entry in self._preprocessors_info:

            if "target_variable" == entry:
                print("computing target_variable")
                self._y = self._X[self._preprocessors_info["target_variable"]["columns"]]
                self._X.drop([self._preprocessors_info["target_variable"]["columns"]], axis=1, inplace=True)

            else:
                preprocessor = load_preprocessor(entry, self._preprocessors_info[entry]["parameters"])
                self._X = preprocessor.compute(self._X, self._preprocessors_info[entry]["columns"])
