from statsmodels.stats.diagnostic import acorr_ljungbox
from EasyModel.builders.residual_diagnostic.abs_residual_test_factory import AbsResidualTestFactory


class Acorr_ljungboxFactory(AbsResidualTestFactory):

    def compute(self, residual, significance, test_name, parameters, x=None):
        p_value = acorr_ljungbox(residual, **parameters)[1]
        if p_value > significance:
            print(test_name + ': Autocorrelation is absent in residuals. p value: ' + str(p_value[0]))
        else:
            print(test_name + ': Autocorrelation is present in residuals. p value: ' + str(p_value[0]))
