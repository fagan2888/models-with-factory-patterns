from EasyModel.factories.abs_preprocessor_factory import AbsPreprocessorFactory


class drop_columns(AbsPreprocessorFactory):
    """The factory class that creates the preprocessor.

    Attributes:
        parameters: parameters(if any) for the drop function by pandas data frame
    """

    def __init__(self, parameters):
        self.parameters = parameters

    def compute(self, data, columns):
        """Drops the column(s) from the dataset

        Args:
            data: your dataset
            columns: list of columns to drop

        Returns:
            returns the preprocessed dataset
        """

        print("computing drop_columns")
        data.drop(columns, **self.parameters, inplace=True, axis=1)
        return data
