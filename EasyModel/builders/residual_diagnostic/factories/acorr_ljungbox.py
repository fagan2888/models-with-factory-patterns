from statsmodels.stats.diagnostic import acorr_ljungbox
from EasyModel.builders.residual_diagnostic.abs_residual_test_factory import AbsResidualTestFactory


class Acorr_ljungboxFactory(AbsResidualTestFactory):
    """Check for autocorrelation in the residuals.

    For more information, please visit https://www.listendata.com/2018/01/linear-regression-in-python.html

    Null Hypothesis: Autocorrelation is absent.
    Alternative Hypothesis: Autocorrelation is present.

    For regression, you would want autocorrelation to be absent.
    """

    def compute(self, residual, significance, test_name, parameters, x=None):
        """The function to compute the autocorrelation check

        Please review statsmodels.stats.diagnostic.acorr_ljungbox for more information.

        Args:
            residual: the residual data derived from your dataset
            significance: this value should be 0.05 in general. Used for determining the hypothesis.
            test_name: the name of the test. This is just a label that doesn't not alter computations.
            parameters: extra parameters for the function "acorr_ljungbox" from statsmodel
            x: this is meant for another check. You can safely ignore here.

        Returns:
            None
        """
        p_value = acorr_ljungbox(residual, **parameters)[1]
        if p_value > significance:
            print(test_name + ': Good. Autocorrelation is absent in residuals. p value: ' + str(p_value[0]))
        else:
            print(test_name + ': Bad. Autocorrelation is present in residuals. p value: ' + str(p_value[0]))
