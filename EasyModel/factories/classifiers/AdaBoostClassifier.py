from EasyModel.factories.abs_factory import AbsFactory
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV


class AdaBoostClassifierFactory(AbsFactory):

    def create_model(self, tuned_parameters, grid_search_parameters):
        self._clf = clf = GridSearchCV(AdaBoostClassifier(), tuned_parameters, **grid_search_parameters)
        return clf
