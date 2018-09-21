from EasyModel.factories.abs_model_factory import AbsModelFactory
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV


class AdaBoostClassifierFactory(AbsModelFactory):

    def create_model(self, tuned_parameters, grid_search_parameters):
        self._clf = clf = GridSearchCV(AdaBoostClassifier(), tuned_parameters, **grid_search_parameters)
        return clf
