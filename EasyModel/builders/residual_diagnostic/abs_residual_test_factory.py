import abc


class AbsResidualTestFactory(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def compute(self, residual, significance, test_name, parameters, x):
        pass