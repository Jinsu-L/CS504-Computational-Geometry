from abc import ABC, abstractmethod
from scipy.spatial import distance


class Point(ABC):
    """
    각 데이터를 위한 포인트 객체..
    """

    def __init__(self, emb):
        self.emb = emb


class SimplePoint(ABC):
    """
    각 데이터를 위한 포인트 객체..
    """

    def __init__(self, emb):
        self.org_emb = emb

    def set_emb(self, emb):
        self.emb = emb


class CifarPoint(Point):

    def __init__(self, emb):
        self.emb = emb
