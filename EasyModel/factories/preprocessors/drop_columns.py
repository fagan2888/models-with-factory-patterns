from EasyModel.factories.abs_preprocessor_factory import AbsPreprocessorFactory


class drop_columns(AbsPreprocessorFactory):

    def __init__(self, parameters):
        self.parameters = parameters

    def compute(self, data, columns):
        print("computing drop_columns")
        data.drop(columns, **self.parameters, inplace=True, axis=1)
        return data
