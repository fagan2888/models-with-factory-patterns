from importlib import import_module
from inspect import getmembers, isabstract, isclass
from .abs_factory import AbsFactory


def load_classifier(factory_name):
    try:
        factory_module = import_module('.' + factory_name, 'EasyModel.factories.classifiers')
    except ImportError:
        print("Failed to load classifiers!")

    classes = getmembers(factory_module,
                         lambda m: isclass(m) and not isabstract(m))

    for name, _class in classes:
        if issubclass(_class, AbsFactory):
            return _class()


def load_regressor(factory_name):
    try:
        factory_module = import_module('.' + factory_name, 'EasyModel.factories.regressors')
    except ImportError:
        print("Failed to load regressors!")

    classes = getmembers(factory_module,
                         lambda m: isclass(m) and not isabstract(m))

    for name, _class in classes:
        if issubclass(_class, AbsFactory):
            return _class()


def load_neural_network(factory_name):
    try:
        factory_module = import_module('.' + factory_name, 'EasyModel.factories.neuralnetworks')
    except ImportError:
        print("Failed to load neural networks!")

    classes = getmembers(factory_module,
                         lambda m: isclass(m) and not isabstract(m))

    for name, _class in classes:
        if issubclass(_class, AbsFactory):
            return _class()