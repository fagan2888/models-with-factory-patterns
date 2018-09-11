from EasyModel.factories.abs_factory import AbsFactory
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


class KNeighborsClassifierFactory(AbsFactory):

    def create_model(self, tuned_parameters, grid_search_parameters):
        self._clf = clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters, **grid_search_parameters)
        return clf
