from EasyModel.factories.abs_model_factory import AbsModelFactory
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV


class AdaBoostClassifierFactory(AbsModelFactory):
    """The factory class that creates the model with grid search.

    For more info, please visit http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
                                http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

    Attributes:
        _clf: stores the model sklearn.ensemble.AdaBoostClassifier
    """

    def create_model(self, tuned_parameters, grid_search_parameters):
        """The function to create the model

        Args:
            tuned_parameters: parameters to be tuned on the model
            grid_search_parameters: parameters to be used on GridSearchCV

        Returns:
            returns the created model(in GridSearchCV type)
        """

        self._clf = clf = GridSearchCV(AdaBoostClassifier(), tuned_parameters, **grid_search_parameters)
        return clf
