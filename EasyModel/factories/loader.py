from importlib import import_module
from inspect import getmembers, isabstract, isclass
from .abs_model_factory import AbsModelFactory
from .abs_preprocessor_factory import AbsPreprocessorFactory


def load_classifier(factory_name):
    """The function to load the classifier class from your factories folder.

    The folder is at EasyModel.factories.classifiers

    Args:
        factory_name: string name of the classifier class you wish to load. factory_name and class file must be
        identical.

    Returns:
        returns the loaded class

    Raises:
        ImportError: An error when you fail to load the class from the factories folder
    """

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
    """The function to load the regressor class from your factories folder.

    The folder is at EasyModel.factories.regressors

    Args:
        factory_name: string name of the regressor class you wish to load. factory_name and class file must be
        identical.

    Returns:
        returns the loaded class

    Raises:
        ImportError: An error when you fail to load the class from the factories folder
    """

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
    """The function to load the neuralnetworks class from your factories folder.

    The folder is at EasyModel.factories.neuralnetworks

    Args:
        factory_name: string name of the neuralnetworks class you wish to load. factory_name and class file must be
        identical.

    Returns:
        returns the loaded class

    Raises:
        ImportError: An error when you fail to load the class from the factories folder
    """

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
    """The function to load the preprocessors class from your factories folder.

    The folder is at EasyModel.factories.preprocessors

    Args:
        factory_name: string name of the preprocessors class you wish to load. factory_name and class file must be
        identical.

    Returns:
        returns the loaded class

    Raises:
        ImportError: An error when you fail to load the class from the factories folder
    """

    try:
        factory_module = import_module('.' + factory_name, 'EasyModel.factories.preprocessors')
    except ImportError:
        print("Failed to load preprocessors!")

    classes = getmembers(factory_module,
                         lambda m: isclass(m) and not isabstract(m))

    for name, _class in classes:
        if issubclass(_class, AbsPreprocessorFactory):
            return _class(parameters)