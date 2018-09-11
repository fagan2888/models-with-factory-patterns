import abc


class AbsBuilder(metaclass=abc.ABCMeta):

    def __init__(self, model):
        self._model = model

    def fit(self, X_train, y_train):
        return self._model.fit(X_train, y_train)

    def predict(self, X_test, y_test):
        return self._model(X_test, y_test)

    @abc.abstractmethod
    def pre_proc(self, commands):
        pass

    @abc.abstractmethod
    def result(self):
        pass
