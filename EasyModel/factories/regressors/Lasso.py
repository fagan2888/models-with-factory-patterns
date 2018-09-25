from EasyModel.factories.abs_model_factory import AbsModelFactory
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV


class LassoFactory(AbsModelFactory):

    def create_model(self, tuned_parameters, grid_search_parameters):
        self._clf = clf = GridSearchCV(Lasso(), tuned_parameters, **grid_search_parameters)
        return clf
