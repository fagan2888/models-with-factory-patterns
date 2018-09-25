from EasyModel.factories.abs_model_factory import AbsModelFactory
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import GridSearchCV


class BayesianRidgeFactory(AbsModelFactory):

    def create_model(self, tuned_parameters, grid_search_parameters):
        self._clf = clf = GridSearchCV(BayesianRidge(), tuned_parameters, **grid_search_parameters)
        return clf
