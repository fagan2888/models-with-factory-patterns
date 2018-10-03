from importlib import import_module
from inspect import getmembers, isabstract, isclass
from .abs_residual_test_factory import AbsResidualTestFactory


def load_residual_test(factory_name):
    """The function to load the residual class from your factories folder.

    The folder is at EasyModel.builders.residual_diagnostic.factories

    Args:
        factory_name: string name of the residual test class you wish to load. factory_name and class file must be
        identical.

    Returns:
        returns the loaded class

    Raises:
        ImportError: An error when you fail to load the class from the factories folder
    """

    try:
        factory_module = import_module('.' + factory_name, 'EasyModel.builders.residual_diagnostic.factories')
    except ImportError:
        print("Failed to load test!")

    classes = getmembers(factory_module, lambda m: isclass(m) and not isabstract(m))

    for name, _class in classes:
        if issubclass(_class, AbsResidualTestFactory):
            return _class()