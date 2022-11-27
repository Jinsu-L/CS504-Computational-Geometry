import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm
import gc
import numpy as np
import pickle

from sklearn.decomposition import PCA

"""
embedding size = 100
chenyaofo/pytorch-cifar-models/cifar100_resnet56의 fc 직전 값을 사용


"""

dataset = torchvision.datasets.CIFAR100("./Dataset/", train=False, transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=1)
# 0 norm img, 1 label

model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet56", pretrained=True)
print(model)

return_nodes = {"avgpool": "avgpool", "layer3": "layer3"}
feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)

# print(feature_extractor(torch.rand(1, 3, 32, 32))['avgpool'].size())
# shape = (1, 3, 32, 32)
# print(feature_extractor(torch.rand(1, 3, 32, 32))['avgpool'].view(shape[0], -1).size())

img_list = []
emb_list = []
label_list = []

i = 0
for img, label in tqdm(dataloader):
    # 이게 아마 64dim
    emb = feature_extractor(img)['avgpool'].view(1, -1).detach().numpy()

    img_list.append(img)
    # print(emb)
    # print(np.shape(emb.detach().numpy()))
    # print(label[0])
    emb_list.append(emb)
    label_list.append(int(label[0]))
    # break
    i += 1

    if i % 1000 == 0:
        gc.collect()

img_list = np.concatenate(img_list, axis=0)
emb_list = np.concatenate(emb_list, axis=0)

body = {
    '64': emb_list,
}

# 2, 4, 8, 16, 32, 64의 차원으로 저장이 필요
for dim in [2, 4, 8, 16, 32]:
    model = PCA(n_components=dim)
    pca_features = model.fit_transform(emb_list)
    body[str(dim)] = pca_features

data = {"embeddings": body,
        "label": label_list,
        "img": img_list
        }

with open("./Dataset/cifar100-resnet56-avgpool.pickle", "wb") as w:
    pickle.dump(data, w, pickle.HIGHEST_PROTOCOL)
