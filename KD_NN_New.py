from NN_Base import NNBase
from scipy.spatial import distance
import numpy as np
import heapq
import gc


class KDTreeNN(NNBase):
    class Node:
        def __init__(self, value, left, right, depth):
            self.value = value
            self.left = left
            self.right = right
            self.depth = depth

    def __init__(self) -> None:
        super().__init__()

        self.root = None
        self.dim = 0
        self.points = []

    def _calc_dist(self, a, b):
        return distance.euclidean(a, b)

    def clear(self):
        self.root = None
        gc.collect()

    def build(self, points):
        self.dim = len(points[0])

        def make(points, i=0):
            if len(points) == 0:
                return None
            cur_dim = i % self.dim

            points = points[points[:, cur_dim].argsort()]
            # mean_value = np.mean(points[:, cur_dim])
            # mid_index = -1
            # for i, point in enumerate(points[:, cur_dim]):
            #     if point >= mean_value:
            #         mid_index = i
            #         break

            mid_index = len(points) // 2
            node = self.Node(points[mid_index], None, None, i)
            node.left = make(points[0:mid_index], i + 1)
            node.right = make(points[mid_index + 1:], i + 1)
            return node

        self.root = make(points)
        gc.collect()

    def add(self, elements: iter):
        pass

    def search(self, query):
        self.KNN_result = []

        def _search(node, query, k):
            if node is None:
                return

            cur_data = node.value

            distance = self._calc_dist(cur_data, query)

            if len(self.KNN_result) < k:
                self.KNN_result.append((node, distance))
            elif distance < self.KNN_result[0][1]:

                self.KNN_result = self.KNN_result[1:] + [(node, distance)]
            self.KNN_result = sorted(self.KNN_result, key=lambda x: -x[1])
            cuttint_dim = node.depth % self.dim
            if abs(query[cuttint_dim] - cur_data[cuttint_dim]) < self.KNN_result[0][1] or len(self.KNN_result) < k:
                _search(node.left, query, k)
                _search(node.right, query, k)
            elif query[cuttint_dim] < cur_data[cuttint_dim]:
                _search(node.left, query, k)
            else:
                _search(node.right, query, k)

        _search(self.root, query, 1)

        return self.KNN_result
if __name__ == '__main__':
    import random
    import pickle

    nn = KDTreeNN()
    with open("Dataset/cifar100-resnet56-avgpool.pickle", "rb") as f:
        dataset = pickle.load(f)

    # 64 dimension points
    points = dataset["embeddings"]['64']
    nn.build(points)

    rand = random.randint(0, len(points))

    print("label: ", dataset["label"][rand])

    ret = nn.search(dataset["embeddings"]['64'][rand])
    # print(ret_idx, dataset["label"][ret_idx])
    print("dist : ", distance.euclidean(dataset["embeddings"]['64'][rand], ret[0][0].value))
