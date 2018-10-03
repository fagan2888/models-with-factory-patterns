import abc


class AbsResidualTestFactory(metaclass=abc.ABCMeta):
    """ The abstract class to inherit for writing residual test in the residual_diagnostic/factories folder."""

    @abc.abstractmethod
    def compute(self, residual, significance, test_name, parameters, x):
        """The function to compute the residual check

        Args:
            residual: the residual data derived from your dataset
            significance: this value should be 0.05 in general. Used for determining the hypothesis.
            test_name: the name of the test. This is just a label that doesn't not alter computations.
            parameters: extra parameters for the external function you will be using
            x: this is meant for test like het_goldfeldquandt, which is the X_test of your dataset.

        Returns:
            None
        """
        pass
