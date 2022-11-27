import argparse
import pickle
import numpy as np
from BF_NN import BruteForceNN
from KD_NN_New import KDTreeNN
from BT_NN import BallTreeNNN
from Datasets import Cifar100, RandData
from itertools import product
import time
from datetime import timedelta
import gc
import sys
sys.setrecursionlimit(100000)


def run():
    # model load
    model = BruteForceNN()

    # dataset load
    # dataset = Cifar100("./Dataset/cifar100-resnet56-avgpool.pickle")
    dataset = RandData()

    target_dim = [8, 16, 32, 64, 128, 256, 512, 1024]
    target_size = [1000, 10000, 100000, 1000000]  # , 20000000, 30000000, 40000000, 50000000]

    # evaluation
    for dim, size in product(target_dim, target_size):
        target_dataset = dataset.get_dataset(dim=dim, size=size)
        print("start ", size)
        # embeddings = [d["embedding"] for d in target_dataset]
        embeddings = target_dataset
        # model = BruteForceNN()
        model = KDTreeNN()
        # model = BallTreeNNN()
        model.build(embeddings)
        # print(end - start)
        q_idxs = [np.random.randint(0, size) for _ in range(22)]

        # todo: N번 테스트 해서 평균 시간을 사용하도록 할 것..
        results = []
        for target_idx in q_idxs:
            # target_idx = np.random.randint(0, size)
            query = embeddings[target_idx]
            start = time.process_time()
            model.search(query)  # 1-NN
            t = time.process_time() - start
            results.append(t)

        results = sorted(results)#[:-2]
        t = sum(results) / len(results)
        print("kd", dim, t * 1000, sep="\t")

        # model = KDTreeNN()
        model = BallTreeNNN()
        model.build(embeddings)
        # print(end - start)

        # todo: N번 테스트 해서 평균 시간을 사용하도록 할 것..
        results = []
        for target_idx in q_idxs:
            # target_idx = np.random.randint(0, size)
            query = embeddings[target_idx]
            start = time.process_time()
            model.search(query)  # 1-NN
            t = time.process_time() - start
            results.append(t)

        results = sorted(results)#[:-2]
        t = sum(results) / len(results)
        print("ball", dim, t * 1000, sep="\t")

        gc.collect()

        # todo: 나중에 csv로 결과를 떨구도록 코드 붙일 것


if __name__ == "__main__":
    run()
