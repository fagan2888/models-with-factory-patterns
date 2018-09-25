from EasyModel.factories.abs_model_factory import AbsModelFactory
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV


class SGDRegressorFactory(AbsModelFactory):

    def create_model(self, tuned_parameters, grid_search_parameters):
        self._clf = clf = GridSearchCV(SGDRegressor(), tuned_parameters, **grid_search_parameters)
        return clf
