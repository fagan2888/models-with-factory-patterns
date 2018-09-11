from EasyModel.factories.abs_factory import AbsFactory
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class RandomForestClassifierFactory(AbsFactory):

    def create_model(self, tuned_parameters, grid_search_parameters):
        self._clf = clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, **grid_search_parameters)
        return clf
