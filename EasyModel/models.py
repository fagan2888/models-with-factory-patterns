from EasyModel.director import Director
from sklearn.metrics import classification_report
from EasyModel.factories.loader import load_classifier


class EasyClassifiers(Director):

    def build_models(self):
        print("building models")
        for entry in self._models_info:
            model = load_classifier(entry)
            model = model.create_model(self._models_info[entry], self._gridsearchcv_info[entry])
            self._models[entry] = model

    def fit_all(self):
        for model in self._models:
            print('-' * 20 + model + '-' * 20)
            print(self._models[model].fit(self._X_train, self._y_train))

    def run(self, x_train, x_test, y_train, y_test):
        self.build_models()
        self.set_data(x_train, x_test, y_train, y_test)
        self.fit_all()

    def classification_reports(self):
        for model in self._models:
            print('-'*20  + model + '-'*20)
            print(classification_report(self._y_test, self._models[model].predict(self._X_test)))

