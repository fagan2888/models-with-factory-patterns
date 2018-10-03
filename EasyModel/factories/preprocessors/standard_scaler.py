from EasyModel.factories.abs_preprocessor_factory import AbsPreprocessorFactory
from sklearn.preprocessing import StandardScaler


class standard_scaler(AbsPreprocessorFactory):
    """The factory class that creates the preprocessor.

    Attributes:
        parameters: parameters(if any) for the StandardScaler in use.
        scaler: stores the sklearn.preprocessing.StandardScaler
    """

    def __init__(self, parameters):
        self.scaler = StandardScaler(**parameters)

    def compute(self, data, columns):
        """Standardize columns by removing the mean and scaling to unit variance

        Args:
            data: your dataset
            columns: list of columns to convert into dummy variables

        Returns:
            returns the preprocessed dataset
        """

        print("computing standard_scaler")
        data[columns] = self.scaler.fit_transform(data[columns])
        return data

