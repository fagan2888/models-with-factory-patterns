import abc


class AbsPreprocessorFactory(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def compute(self, data, columns):
        pass
