from statsmodels.stats.api import het_goldfeldquandt
from EasyModel.builders.residual_diagnostic.abs_residual_test_factory import AbsResidualTestFactory


class Het_goldfeldquandtFactory(AbsResidualTestFactory):
    """Check for constant variance in the residuals (heteroscedasticity).

    For more information, please visit https://www.listendata.com/2018/01/linear-regression-in-python.html

    Null Hypothesis: Error terms are homoscedastic.
    Alternative Hypothesis: Error terms are heteroscedastic.

    For regression, you would want your residuals to be homoscedastic.
    """

    def compute(self, residual, significance, test_name, parameters, x=None):
        """The function to compute the constant variance check

        Please review statsmodels.stats.api.het_goldfeldquandtfor more information.

        Args:
            residual: the residual data derived from your dataset
            significance: this value should be 0.05 in general. Used for determining the hypothesis.
            test_name: the name of the test. This is just a label that doesn't not alter computations.
            parameters: extra parameters for the function "het_goldfeldquandt" from statsmodel
            x: the X_test of your dataset

        Returns:
            None
        """

        p_value = het_goldfeldquandt(residual, x, **parameters)[1]
        if p_value > significance:
            print(test_name + ': Good. The residuals have constant variance. (homoscedastic) p value: ' + str(p_value))
        else:
            print(test_name + ': Bad. The residuals do not have constant variance. (heteroscedastic) p value: ' + str(p_value))
