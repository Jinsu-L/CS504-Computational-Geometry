from NN_Base import NNBase
import numpy as np
from scipy.spatial import distance
from functools import partial
import pickle
import sys


class BruteForceNN(NNBase):
    def __init__(self) -> None:
        super().__init__()

        self.points = []
        self.dist = self._calc_dist

    def _calc_dist(self, a, b):
        return distance.euclidean(a, b)

    def clear(self):
        self.points = []

    def build(self, points):
        self.points = points
        # for p in points:
        #     self.add(p)

    def add(self, elements: iter):
        self.points.append(elements)

    def search(self, query):
        # build를 해서 nn을 만들어야함.
        assert len(self.points) != 0

        # calc dist
        min_idx = -1
        min_dis = sys.float_info.max
        for idx, point in enumerate(self.points):
            dis = self._calc_dist(point, query)
            if dis < min_dis:
                min_idx = idx
                min_dis = dis

        return min_idx, self.points[min_idx]


if __name__ == '__main__':
    import random

    nn = BruteForceNN()
    with open("Dataset/cifar100-resnet56-avgpool.pickle", "rb") as f:
        dataset = pickle.load(f)

    # 64 dimension points
    points = dataset["embeddings"]['64']
    nn.build(points)

    rand = random.randint(0, len(points))

    print("label: ", dataset["label"][rand])
    ret_idx, ret_vector = nn.search(dataset["embeddings"]['64'][rand])
    print(ret_idx, dataset["label"][ret_idx])
