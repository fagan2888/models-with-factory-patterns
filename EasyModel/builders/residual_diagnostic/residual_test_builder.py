import json
from os import getcwd
from .loader import load_residual_test


class ResidualTestBuilder(object):

    def __init__(self, parameters_json):
        self._tests = {}
        with open(getcwd() + "\\" + parameters_json, 'r') as f:
            self._tests_info = json.load(f)

    def compute_tests(self, residual, significance, x=None):
        for entry in self._tests_info:
            test = load_residual_test(entry)
            test.compute(residual, significance, entry, self._tests_info[entry], x=x)