from EasyModel.factories.abs_preprocessor_factory import AbsPreprocessorFactory
from pandas import get_dummies as gd
from pandas import concat


class get_dummies(AbsPreprocessorFactory):

    def __init__(self, parameters):
        self.parameters = parameters

    def compute(self, data, columns):
        print("computing get_dummies")

        for col in columns:
            dummies = gd(data[col], **self.parameters)
            data = concat([data, dummies], axis=1, sort=False, copy=False)
            data.drop([col], axis=1, inplace=True)

        return data
