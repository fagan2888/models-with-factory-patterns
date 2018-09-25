from importlib import import_module
from inspect import getmembers, isabstract, isclass
from .abs_residual_test_factory import AbsResidualTestFactory


def load_residual_test(factory_name):
    try:
        factory_module = import_module('.' + factory_name, 'EasyModel.builders.residual_diagnostic.factories')
    except ImportError:
        print("Failed to load test!")

    classes = getmembers(factory_module,
                         lambda m: isclass(m) and not isabstract(m))

    for name, _class in classes:
        if issubclass(_class, AbsResidualTestFactory):
            return _class()