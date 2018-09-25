from scipy.stats import shapiro
from EasyModel.builders.residual_diagnostic.abs_residual_test_factory import AbsResidualTestFactory


class ShapiroFactory(AbsResidualTestFactory):

    def compute(self, residual, significance, test_name, parameters, x=None):
        p_value = shapiro(residual, **parameters)[1]
        if p_value > significance:
            print(test_name + ': The residuals are normally distributed. p value: ' + str(p_value))
        else:
            print(test_name + ': The residuals are not normally distributed. p value: ' + str(p_value))
