from EasyModel.factories.abs_factory import AbsFactory
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


class LogisticRegressionFactory(AbsFactory):

    def create_model(self, tuned_parameters, grid_search_parameters):
        self._clf = clf = GridSearchCV(LogisticRegression(), tuned_parameters, **grid_search_parameters)
        return clf
