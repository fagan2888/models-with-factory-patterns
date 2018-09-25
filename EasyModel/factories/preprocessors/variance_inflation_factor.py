from EasyModel.factories.abs_preprocessor_factory import AbsPreprocessorFactory
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
import operator

class variance_inflation_factor(AbsPreprocessorFactory):

    def __init__(self, parameters):
        self.parameters = parameters

    def compute(self, data, columns):
        print("computing variance_inflation_factor")
        results = {}
        for i in range(data.shape[1]):
            results[data.columns[i]] = vif(data.values, i)

        print(sorted(results.items(), key=operator.itemgetter(1)))
        return data
