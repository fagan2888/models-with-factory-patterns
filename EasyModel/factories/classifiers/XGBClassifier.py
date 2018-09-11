from xgboost import XGBClassifier
from EasyModel.factories.abs_factory import AbsFactory
from sklearn.model_selection import GridSearchCV


class XGBClassifierFactory(AbsFactory):

    def create_model(self, tuned_parameters, grid_search_parameters):
        self._clf = clf = GridSearchCV(XGBClassifier(), tuned_parameters, **grid_search_parameters)
        return clf
