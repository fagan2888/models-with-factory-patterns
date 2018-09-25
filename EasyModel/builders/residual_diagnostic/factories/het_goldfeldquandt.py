from statsmodels.stats.api import het_goldfeldquandt
from EasyModel.builders.residual_diagnostic.abs_residual_test_factory import AbsResidualTestFactory


class Het_goldfeldquandtFactory(AbsResidualTestFactory):

    def compute(self, residual, significance, test_name, parameters, x=None):
        p_value = het_goldfeldquandt(residual, x, **parameters)[1]
        if p_value > significance:
            print(test_name + ': The residuals have constant variance. (homoscedastic) p value: ' + str(p_value))
        else:
            print(test_name + ': The residuals do not have constant variance. (heteroscedastic) p value: ' + str(p_value))
