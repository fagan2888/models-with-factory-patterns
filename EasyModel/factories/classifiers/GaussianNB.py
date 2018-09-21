from EasyModel.factories.abs_model_factory import AbsModelFactory
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV


class GaussianNBFactory(AbsModelFactory):

    def create_model(self, tuned_parameters, grid_search_parameters):
        self._clf = clf = GridSearchCV(GaussianNB(), tuned_parameters, **grid_search_parameters)
        return clf
