from importlib import import_module
from inspect import getmembers, isabstract, isclass
from .abs_model_factory import AbsModelFactory
from .abs_preprocessor_factory import AbsPreprocessorFactory


def load_classifier(factory_name):
    try:
        factory_module = import_module('.' + factory_name, 'EasyModel.factories.classifiers')
    except ImportError:
        print("Failed to load classifiers!")

    classes = getmembers(factory_module,
                         lambda m: isclass(m) and not isabstract(m))

    for name, _class in classes:
        if issubclass(_class, AbsModelFactory):
            return _class()


def load_regressor(factory_name):
    try:
        factory_module = import_module('.' + factory_name, 'EasyModel.factories.regressors')
    except ImportError:
        print("Failed to load regressors!")

    classes = getmembers(factory_module,
                         lambda m: isclass(m) and not isabstract(m))

    for name, _class in classes:
        if issubclass(_class, AbsModelFactory):
            return _class()


def load_neural_network(factory_name):
    try:
        factory_module = import_module('.' + factory_name, 'EasyModel.factories.neuralnetworks')
    except ImportError:
        print("Failed to load neural networks!")

    classes = getmembers(factory_module,
                         lambda m: isclass(m) and not isabstract(m))

    for name, _class in classes:
        if issubclass(_class, AbsModelFactory):
            return _class()


def load_preprocessor(factory_name, parameters):
    try:
        factory_module = import_module('.' + factory_name, 'EasyModel.factories.preprocessors')
    except ImportError:
        print("Failed to load preprocessors!")

    classes = getmembers(factory_module,
                         lambda m: isclass(m) and not isabstract(m))

    for name, _class in classes:
        if issubclass(_class, AbsPreprocessorFactory):
            return _class(parameters)