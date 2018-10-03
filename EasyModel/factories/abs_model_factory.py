import abc


class AbsModelFactory(metaclass=abc.ABCMeta):
    """the abstract class to inherit for creating the models in the factories folder"""

    """The function to create the model. To be implemented in the child class.

    Args:
        tuned_parameters: parameters to be tuned on the model
        grid_search_parameters: parameters to be used on GridSearchCV

    Returns:
        returns the created model(in GridSearchCV type)
    """
    @abc.abstractmethod
    def create_model(self, tuned_parameters, grid_search_parameters):
        pass

