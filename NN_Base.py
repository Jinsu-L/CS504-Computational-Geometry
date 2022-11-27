from abc import ABC, abstractmethod

class NNBase(ABC):

    @abstractmethod
    def add(self, elements: iter):
        pass

    @abstractmethod
    def build(self, points):
        pass

    @abstractmethod
    def search(self, query):
        pass