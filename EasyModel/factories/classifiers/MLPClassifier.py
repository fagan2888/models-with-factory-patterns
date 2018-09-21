from EasyModel.factories.abs_model_factory import AbsModelFactory
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


class MLPClassifierFactory(AbsModelFactory):

    def create_model(self, tuned_parameters, grid_search_parameters):
        self._clf = clf = GridSearchCV(MLPClassifier(), tuned_parameters, **grid_search_parameters)
        return clf
