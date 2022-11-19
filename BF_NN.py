from NN_Base import NNBase
import numpy as np
from scipy.spatial import distance
from functools import partial

class BruteForceNN(NNBase):
    def __init__(self) -> None:
        super().__init__()

        self.points = []
        self.dist = self._calc_dist

    def _calc_dist(self, a, b):
        return distance.euclidean(a, b)

    def build(self, points):
        for p in points:
            self.add(p)

    def add(self, elements: iter):
        self.points.append(elements)

    def search(self, query):

        # build를 해서 nn을 만들어야함.
        assert len(self.points) != 0

        # calc dist
        return self.points[np.argmin(list(map(partial(self._calc_dist, b=query), self.points)))]


if __name__ == '__main__':
    nn = BruteForceNN()
    points = [(1,2,5), (2,2,2),(3,4,5),(5,5,0),(10,1,2)]
    nn.build(points)

    print(nn.search((3,3,2))) # (2, 2, 2)
