import abc


class AbsModelFactory(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def create_model(self, tuned_parameters, grid_search_parameters):
        pass

