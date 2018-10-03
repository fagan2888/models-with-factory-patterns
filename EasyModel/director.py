import abc


class Director(metaclass=abc.ABCMeta):
    """This is meant for higher level structure that encapsulates all the models and builders.
    Not in use yet.
    """

    def __init__(self):
        print("Meant for custom project setup")
