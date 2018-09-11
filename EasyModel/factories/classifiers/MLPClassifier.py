from EasyModel.factories.abs_factory import AbsFactory
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


class MLPClassifierFactory(AbsFactory):

    def create_model(self, tuned_parameters, grid_search_parameters):
        self._clf = clf = GridSearchCV(MLPClassifier(), tuned_parameters, **grid_search_parameters)
        return clf
