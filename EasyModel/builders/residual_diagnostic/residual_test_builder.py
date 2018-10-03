import json
from os import getcwd
from .loader import load_residual_test


class ResidualTestBuilder(object):
    """The builder to load all the residual tests using the input json file.

    Attributes:
        _tests_info: the information of each test loaded from the json
    """

    def __init__(self, parameters_json):
        """Constructor for ResidualTestBuilder

        Loads the information from parameters_json into _test_info
        parameters_json format example: { "shapiro" : {}, "acorr_ljungbox": {"lags": 1} }

        Args:
            parameters_json: the json file which contains the residual test you wish to load and use.

        Returns:
            None
        """

        with open(getcwd() + "\\" + parameters_json, 'r') as f:
            self._tests_info = json.load(f)

    def compute_tests(self, residual, significance, x=None):
        """The function to load and compute all the residual tests

        Args:
            residual: the residual data derived from your dataset
            significance: this value should be 0.05 in general. Used for determining the hypothesis.
            x: the X_test of your dataset, only applicable to certain test such as het_goldfeldquandt

        Returns:
            None
        """
        for entry in self._tests_info:
            test = load_residual_test(entry)

            # The "entry" argument passed is just the name of residual test (meant for labeling).
            # The "self._tests_info[entry]" argument is the extra parameters for the computing function
            test.compute(residual, significance, entry, self._tests_info[entry], x=x)
