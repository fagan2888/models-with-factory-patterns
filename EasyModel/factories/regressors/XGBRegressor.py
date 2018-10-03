from xgboost import XGBRegressor
from EasyModel.factories.abs_model_factory import AbsModelFactory
from sklearn.model_selection import GridSearchCV


class XGBRegressorFactory(AbsModelFactory):
    """The factory class that creates the model with grid search.

    For more info, please visit https://xgboost.readthedocs.io/en/latest/python/python_api.html
                                http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

    Attributes:
        _reg: stores the model xgboost.XGBRegressor
    """

    def create_model(self, tuned_parameters, grid_search_parameters):
        """The function to create the model

        Args:
            tuned_parameters: parameters to be tuned on the model
            grid_search_parameters: parameters to be used on GridSearchCV

        Returns:
            returns the created model(in GridSearchCV type)
        """

        self._reg = reg = GridSearchCV(XGBRegressor(), tuned_parameters, **grid_search_parameters)
        return reg
