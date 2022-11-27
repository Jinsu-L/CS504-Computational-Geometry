import gc

import numpy as np
import heapq
from NN_Base import NNBase
from scipy.spatial import distance

class BallTreeNN(NNBase):
    class Node:
        """
            BallTree를 위한 Node....
        """

        def __init__(self):
            pass

    def __init__(self):
        pass

    def _calc_dist(self, a, b):
        pass

    def clear(self):
        pass

    def build(self, points):
        pass

    def add(self, elements: iter):
        pass

    def search(self, query):
        pass


class BallTreeNNN(NNBase):
    """
        numpy vector만 사용용
    """

    class Node:
        def __init__(self, center=None, radius=None, points=None, left=None, right=None):
            self.center = center
            self.radius = radius
            self.left = left
            self.right = right
            self.points = points

    def __init__(self):
        self.root = self.Node()
        self.size = 0
        self.KNN_result = [(None, np.inf)]

    def _calc_dist(self, a, b):
        return distance.euclidean(a, b)

    def clear(self):
        pass

    """
        def build_BallTree(self):
        data = np.column_stack((self.values,self.labels))
        return self.build_BallTree_core(data)
    """

    def build(self, points):
        def make(point_list):
            if len(point_list) == 0:
                return None

            if len(point_list) < 3:
                # points[0, :-1] 라벨을 때는 행위
                return self.Node(center=point_list[0], radius=0.0, points=point_list)

            # 첫번째를 맨 뒤로 보냄?, 각 데이터 포인트가 동일하면, 모두 ball 에 넣어서 재귀를 종료
            # 무한하게 깊어져서 overflow 나는 걸 방지
            # data_disloc = np.row_stack((point_list[1:], point_list[0]))
            # if np.sum(data_disloc - point_list) == 0:
            #     # print(len(point_list))
            #     return self.Node(center=point_list[0], radius=1e-8, points=point_list)

            cur_center = np.mean(point_list, axis=0)  # 현재 ball의 중심
            dists_with_center = np.array(
                [self._calc_dist(cur_center, point) for point in point_list])  # 중심으로부터 각 점까지 거리
            max_dist_index = np.argmax(dists_with_center)

            max_dist = dists_with_center[max_dist_index]
            root = self.Node(center=cur_center, radius=max_dist, points=point_list)
            point1 = point_list[max_dist_index]

            # 중심으로 지정된 point에서 가장 먼 point
            dists_with_point1 = np.array([self._calc_dist(point1, point) for point in point_list])
            max_dist_index2 = np.argmax(dists_with_point1)
            point2 = point_list[max_dist_index2]

            dists_with_point2 = np.array([self._calc_dist(point2, point) for point in point_list])
            assign_point1 = dists_with_point1 < dists_with_point2

            root.left = make(point_list[assign_point1])
            root.right = make(point_list[~assign_point1])

            return root

        self.root = make(points)
        gc.collect()

    def add(self, elements: iter):
        pass

    def search(self, query):
        return self.search_k(query, 1)

    def search_k(self, query, k):
        self.KNN_result = [(None, np.inf)]

        # 재귀적으로 찾는데...
        def search_KNN_core(root_ball, query, K):
            if root_ball is None:
                return

            # 가장 마지막 node 일때
            if root_ball.left is None or root_ball.right is None:
                # self.insert(root_ball, query, K)
                for point in root_ball.points:  # cur points는 전부 같은 vector들임.. vector가 중복되는 경우...
                    # repr_point = root_ball.points[0]  # cur points는 전부 같은 vector들임.. vector가 중복되는 경우...
                    distance = self._calc_dist(query, point)
                    if (len(self.KNN_result) < K):
                        self.KNN_result.append((point, distance))
                    elif distance < self.KNN_result[0][1]:
                        self.KNN_result = self.KNN_result[1:] + [(point, distance)]
                    self.KNN_result = sorted(self.KNN_result, key=lambda x: -x[1])

            #  해당 root와 점의 거리 < root의 반지름 + 가장 가까운 점의 거리, 여기가 삼각부등식 부분이구나...
            if abs(self._calc_dist(root_ball.center, query)) <= root_ball.radius + self.KNN_result[0][1]:
                search_KNN_core(root_ball.left, query, K)
                search_KNN_core(root_ball.right, query, K)

        search_KNN_core(self.root, query, k)

        return self.KNN_result


# 결과 테스트 ...
if __name__ == '__main__':
    import random
    import pickle

    nn = BallTreeNNN()
    with open("Dataset/cifar100-resnet56-avgpool.pickle", "rb") as f:
        dataset = pickle.load(f)

    # 64 dimension points
    points = dataset["embeddings"]['64']
    nn.build(points)

    rand = random.randint(0, len(points))

    print("label: ", dataset["label"][rand])
    ret = nn.search(dataset["embeddings"]['64'][rand])
    # print(ret_idx, dataset["label"][ret_idx])
    print("dist : ", distance.euclidean(dataset["embeddings"]['64'][rand], ret[0][0]))
