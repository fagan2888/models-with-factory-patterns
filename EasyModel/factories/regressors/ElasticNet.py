from EasyModel.factories.abs_model_factory import AbsModelFactory
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV


class ElasticNetFactory(AbsModelFactory):

    def create_model(self, tuned_parameters, grid_search_parameters):
        self._clf = clf = GridSearchCV(ElasticNet(), tuned_parameters, **grid_search_parameters)
        return clf
