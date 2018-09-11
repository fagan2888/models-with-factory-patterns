from EasyModel.factories.abs_factory import AbsFactory
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


class SVCFactory(AbsFactory):

    def create_model(self, tuned_parameters, grid_search_parameters):
        self._clf = clf = GridSearchCV(SVC(), tuned_parameters, **grid_search_parameters)
        return clf