import math
import torch
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
   def __init__(self, dim, class_num):
       super(Net, self).__init__()
       self.class_num = class_num
       self.encoder = nn.Sequential(
           nn.Dropout(p=0.2, inplace=False),
           nn.Linear(dim, 2000, bias=True),
           nn.ReLU(),
           nn.Dropout(p=0.2, inplace=False),
           nn.Linear(2000, 2000, bias=True),
           nn.ReLU(),
           nn.Dropout(p=0.2, inplace=False),
           nn.Linear(2000, 200, bias=True),
           nn.ReLU(),
           nn.Dropout(p=0.2, inplace=False),
           nn.Linear(200, 20, bias=True),
       )
       self.decoder = nn.Sequential(
           nn.Linear(20, 200, bias=True),
           nn.ReLU(),
           nn.Linear(200, 2000, bias=True),
           nn.ReLU(),
           nn.Linear(2000, 2000, bias=True),
           nn.ReLU(),
           nn.Linear(2000, dim, bias=True),
           nn.Sigmoid(),
       )
       self.classifier = nn.Sequential(
           nn.Linear(20, 10, bias=True),
           nn.ReLU(),
           nn.Linear(10, 5, bias=True),
           nn.ReLU(),
           nn.Linear(5, 1, bias=True),
       )
       self.cluster_layer = nn.Linear(10, class_num, bias=False)
       self.cluster_center = torch.rand([class_num, 10], requires_grad=False).cuda()

    def encode(self, x):
        x = self.encoder(x)
        x = F.normalize(x)
        return x

    def decode(self, x):
        return self.decoder(x)
        
    def cluster(self, z):
        return self.cluster_layer(z)

    def init_cluster_layer(self, alpha, cluster_center):
        self.cluster_layer.weight.data = 2 * alpha * cluster_center

    def compute_cluster_center(self, alpha):
        self.cluster_center = 1.0 / (2 * alpha) * self.cluster_layer.weight
        return self.cluster_center

    def normalize_cluster_center(self, alpha):
        self.cluster_layer.weight.data = (
            F.normalize(self.cluster_layer.weight.data, dim=1) * 2.0 * alpha
        )

    def predict(self, z):
        distance = torch.cdist(z, self.cluster_center, p=2)
        prediction = torch.argmin(distance, dim=1)
        return prediction

