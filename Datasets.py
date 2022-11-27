import gc

import numpy as np
import pickle
import copy
from abc import ABC, abstractmethod
from Point import SimplePoint, CifarPoint
from collections import defaultdict
from collections import namedtuple
from tqdm import tqdm


class Dataset(ABC):

    @abstractmethod
    def get_dataset(self, dim, size):
        pass


Point = namedtuple("simple", ["emb", "emb64", "emb32", "emb16", "emb8", "emb4", "emb2"])


class RandData(Dataset):

    # def __init__(self, centroid_size=10, a=-10, b=10, v=3, max_size=50000000, max_dim=64):
    def __init__(self, a=-0.5, b=0.5, v=1, max_size=1000000, max_dim=64):
        # self.means = (b - a) * np.random.random_sample(2) + a
        # self.vars = v * np.random.random_sample(2)
        # self.dataset = []
        # # self.dataset = np.random.normal(self.means, self.vars, (max_size, max_dim))
        # n = max_size // 2
        # for i in tqdm(range(2)):
        #     gen_points = np.random.normal(self.means[i], self.vars[i], (n, max_dim))
        #     # for emb in gen_points:
        #     # self.dataset.append(Point(emb, emb[:64], emb[:32], emb[:16], emb[:8], emb[:4], emb[:2]))
        #     self.dataset.append(gen_points)
        # self.dataset = np.concatenate(self.dataset)


        mean = [np.random.random_sample() * 10 for _ in range(max_dim)]
        cov = []
        for _ in range(max_dim):
            cov.append([np.random.random_sample() * np.random.random_sample() for _ in range(max_dim)])
        self.dataset = np.random.multivariate_normal(mean, cov, max_size)

        np.random.shuffle(self.dataset)
        gc.collect()

    def get_dataset(self, dim=64, size=1000000):
        # result = []
        # for i, point in enumerate(self.dataset):
        #     # result.append(point["emb" + str(dim)])
        #     if i >= size:
        #         break

        return self.dataset[:size, :dim]


class Cifar100(Dataset):
    def __init__(self, path):
        self.path = path
        self._load_dataset(self.path)

    def get_dataset(self, dim=64, size=None):
        """
        :param dim:
        :param size:
        :return:
        """
        target_dim = dim
        result = []

        target = self._refine_dataset(self.dataset, dim)
        np.random.shuffle(target)
        if size is not None:
            target = target[:size]

        return target

    def _refine_dataset(self, org_dataset, dim):
        data = []
        # 64 32 ,16, 8, 4, 2
        embedding = org_dataset["embeddings"][str(dim)]

        for emb, label, img in zip(embedding, org_dataset["label"], org_dataset["img"]):
            point_group = dict()
            point_group["embedding"] = CifarPoint(emb)
            point_group["label"] = label
            point_group["img"] = img
            data.append(copy.deepcopy(point_group))

        return data

    def _load_dataset(self, path):
        with open(path, "rb") as f:
            self.dataset = pickle.load(f)  # embeddings:64,label, img


class Ubuntu(Dataset):
    pass
