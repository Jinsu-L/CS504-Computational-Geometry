from NN_Base import NNBase
from scipy.spatial import distance
import heapq
import gc


class KDTreeNN(NNBase):
    class Node:
        pass

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
            if len(points) > 1:
                # 해당 dim 축으로 정렬 O(nlogn)
                # points.sort(key=lambda x: x[i])
                sorted(points, key=lambda x: x[i])

                i = (i + 1) % self.dim
                m = len(points) >> 1

                # Left, Right, Now
                return [make(points[:m], i), make(points[m + 1:], i),
                        points[m]]
            if len(points) == 1:
                return [None, None, points[0]]

        self.root = make(points)
        gc.collect()

    def add(self, elements: iter):
        pass

    def search(self, query):
        heap = []
        return self._search(self.root, query, 1, heap)

    def _search(self, node, query, k, heap, i=0, tiebreaker=1):
        if node is not None:
            # 2번째 element와 거리 계싼
            dist_sq = self._calc_dist(query, node[2])
            dx = node[2][i] - query[i]  # i dim에서 diff

            if len(heap) < k:
                heapq.heappush(heap, (-dist_sq, tiebreaker, node[2]))
            elif dist_sq < -heap[0][0]:
                heapq.heappushpop(heap, (-dist_sq, tiebreaker, node[2]))
            i = (i + 1) % self.dim

            # Goes into the left branch, then the right branch if needed
            for b in (dx < 0, dx >= 0)[:1 + (dx * dx < -heap[0][0])]:
                self._search(node[b], query, k, heap, i, (tiebreaker << 1) | b)
        if tiebreaker == 1:
            # return [(-h[0], h[2]) if return_dist_sq else h[2] for h in sorted(heap)][::-1]
            return [(-h[0], h[2]) for h in sorted(heap)][::-1]


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

    ret = nn.search(dataset["embeddings"]['64'][rand])[0]
    # print(ret_idx, dataset["label"][ret_idx])
    print("dist : ", distance.euclidean(dataset["embeddings"]['64'][rand], ret[1]))