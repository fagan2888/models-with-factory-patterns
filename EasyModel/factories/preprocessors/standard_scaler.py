from EasyModel.factories.abs_preprocessor_factory import AbsPreprocessorFactory
from sklearn.preprocessing import StandardScaler


class standard_scaler(AbsPreprocessorFactory):

    def __init__(self, parameters):
        self.scaler = StandardScaler(**parameters)

    def compute(self, data, columns):
        print("computing standard_scaler")
        data[columns] = self.scaler.fit_transform(data[columns])
        return data

