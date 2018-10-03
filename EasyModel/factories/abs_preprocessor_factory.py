import abc


class AbsPreprocessorFactory(metaclass=abc.ABCMeta):
    """the abstract class to inherit for creating the preprocessor in the factories folder"""

    @abc.abstractmethod
    def compute(self, data, columns):
        """The main body of the preprocessor. To be implemented in the child class

         Args:
             data: your dataset
             columns: list of columns affected

         Returns:
             returns the preprocessed dataset
         """

        pass
