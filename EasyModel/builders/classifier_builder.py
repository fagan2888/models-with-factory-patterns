from .abs_builder import AbsBuilder


class ClassifierBuilder(AbsBuilder):

    def pre_proc(self, commands):
        print("preprocessing")

    def result(self):
        print("Result")