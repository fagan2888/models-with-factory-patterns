from EasyModel.factories.abs_preprocessor_factory import AbsPreprocessorFactory
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
import operator


class variance_inflation_factor(AbsPreprocessorFactory):
    """The factory class that creates the preprocessor.

    Attributes:
        parameters: parameters(if any) for the functions in use.
    """

    def __init__(self, parameters):
        self.parameters = parameters

    def compute(self, data, columns):
        """Checks for multicolinearity of the dataset by using variance inflation factor.

        Args:
            data: your dataset
            columns: not in use; exist due to structural consistency
        Returns:
        """

        print("computing variance_inflation_factor")
        results = {}
        for i in range(data.shape[1]):
            results[data.columns[i]] = vif(data.values, i)

        print(sorted(results.items(), key=operator.itemgetter(1)))

        # returns dataset due to structural consistency
        return data
