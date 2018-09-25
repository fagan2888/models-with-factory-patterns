from EasyModel.factories.abs_model_factory import AbsModelFactory
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


class RidgeFactory(AbsModelFactory):

    def create_model(self, tuned_parameters, grid_search_parameters):
        self._clf = clf = GridSearchCV(Ridge(), tuned_parameters, **grid_search_parameters)
        return clf
