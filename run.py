import argparse
import pickle
import numpy as np

def run():
    with open("Dataset/cifar100-resnet56-avgpool-64.pickle", "rb") as f:
        dataset = pickle.load(f)

        print(len(dataset["label"]))
        print(dataset["label"][0])
        print(np.shape(dataset["embedding"]))
        print(len(dataset["img"]))

if __name__ == "__main__":
    run()