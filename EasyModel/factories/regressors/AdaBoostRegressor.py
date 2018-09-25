from EasyModel.factories.abs_model_factory import AbsModelFactory
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV


class DecisionTreeRegressorFactory(AbsModelFactory):

    def create_model(self, tuned_parameters, grid_search_parameters):

        self._clf = clf = GridSearchCV(AdaBoostRegressor(), tuned_parameters, **grid_search_parameters)
        return clf
