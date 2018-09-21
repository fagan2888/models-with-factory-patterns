from EasyModel.factories.abs_model_factory import AbsModelFactory
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class RandomForestClassifierFactory(AbsModelFactory):

    def create_model(self, tuned_parameters, grid_search_parameters):
        self._clf = clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, **grid_search_parameters)
        return clf
