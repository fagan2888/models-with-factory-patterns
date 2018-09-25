from EasyModel.factories.abs_model_factory import AbsModelFactory
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


class RandomForestRegressorFactory(AbsModelFactory):

    def create_model(self, tuned_parameters, grid_search_parameters):
        self._clf = clf = GridSearchCV(RandomForestRegressor(), tuned_parameters, **grid_search_parameters)
        return clf
