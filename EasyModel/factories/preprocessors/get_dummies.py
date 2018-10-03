from EasyModel.factories.abs_preprocessor_factory import AbsPreprocessorFactory
from pandas import get_dummies as gd
from pandas import concat


class get_dummies(AbsPreprocessorFactory):
    """The factory class that creates the preprocessor.

    Attributes:
        parameters: parameters(if any) for the get_dummies function by pandas.
    """

    def __init__(self, parameters):
        self.parameters = parameters

    def compute(self, data, columns):
        """Converts the columns(categorical data) into dummy variables

        Args:
            data: your dataset
            columns: list of columns to convert into dummy variables

        Returns:
            returns the preprocessed dataset
        """

        print("computing get_dummies")

        for col in columns:
            dummies = gd(data[col], **self.parameters)
            data = concat([data, dummies], axis=1, sort=False, copy=False)
            data.drop([col], axis=1, inplace=True)

        return data
