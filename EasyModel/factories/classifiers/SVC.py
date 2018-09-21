from EasyModel.factories.abs_model_factory import AbsModelFactory
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


class SVCFactory(AbsModelFactory):

    def create_model(self, tuned_parameters, grid_search_parameters):
        self._clf = clf = GridSearchCV(SVC(), tuned_parameters, **grid_search_parameters)
        return clf