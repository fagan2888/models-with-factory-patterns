from EasyModel.factories.abs_model_factory import AbsModelFactory
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV


class KNeighborsRegressorFactory(AbsModelFactory):

    def create_model(self, tuned_parameters, grid_search_parameters):
        self._clf = clf = GridSearchCV(KNeighborsRegressor(), tuned_parameters, **grid_search_parameters)
        return clf
