from scipy.stats import shapiro
from EasyModel.builders.residual_diagnostic.abs_residual_test_factory import AbsResidualTestFactory


class ShapiroFactory(AbsResidualTestFactory):
    """Check for normal distribution in the residuals.

    For more information, please visit https://www.listendata.com/2018/01/linear-regression-in-python.html

    Null Hypothesis: The residuals are normally distributed.
    Alternative Hypothesis: The residuals are not normally distributed.

    For regression, you would want the residuals to be normally distributed.
    """

    def compute(self, residual, significance, test_name, parameters, x=None):
        """The function to compute the autocorrelation check

        Please review scipy.stats.shapiro for more information.

        Args:
            residual: the residual data derived from your dataset
            significance: this value should be 0.05 in general. Used for determining the hypothesis.
            test_name: the name of the test. This is just a label that doesn't not alter computations.
            parameters: extra parameters for the function "shapiro" from scipy
            x: this is meant for another check. You can safely ignore here.

        Returns:
            None
        """
        p_value = shapiro(residual, **parameters)[1]
        if p_value > significance:
            print(test_name + ': Good. The residuals are normally distributed. p value: ' + str(p_value))
        else:
            print(test_name + ': Bad. The residuals are not normally distributed. p value: ' + str(p_value))
