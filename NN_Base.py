from abc import ABC, abstractmethod

class NNBase(ABC):

    @abstractmethod
    def add(self, elements: iter):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def search(self, e, k):
        pass