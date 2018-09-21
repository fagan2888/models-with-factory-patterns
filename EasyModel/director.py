import abc


class Director(metaclass=abc.ABCMeta):

    def __init__(self):
        print("Meant for custom project setup")
