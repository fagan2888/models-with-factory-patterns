from .abs_model_builder import AbsModelBuilder
from EasyModel.factories.loader import load_regressor

class EasyRegressors(AbsModelBuilder):

    def build_models(self):
        print("building models")
        for entry in self._models_info:
            model = load_regressor(entry)
            model = model.create_model(self._models_info[entry], self._gridsearchcv_info[entry])
            self._models[entry] = model