import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import torchvision
from model import Net
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
from pald import *
from sknetwork.clustering import modularity


class MyDataset(Dataset):
    def __init__(self,data_path):
        self.data = pd.read_csv(data_path).values

    def __getitem__(self, index):
        return MyDataset.to_tensor(self.data[index])

    def __len__(self):
        return len(self.data)

    @staticmethod
    def to_tensor(data):
        return torch.from_numpy(data)


def save_model(model, optimizer, current_epoch):
    out = os.path.join("./save/checkpoint_{}.tar".format(current_epoch))
    state = {
        "net": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": current_epoch,
    }
    torch.save(state, out)


def inference():
    net.compute_cluster_center(alpha)
    net.eval()
    feature_vector = []
    # labels_vector = []
    pred_vector = []
    with torch.no_grad():
        for step, x in enumerate(data_loader_test):
            x = x.float().cuda()
            with torch.no_grad():
                z = net.encode(x)
                pred = net.predict(z)
            feature_vector.extend(z.detach().cpu().numpy())
            # labels_vector.extend(y.numpy())
            pred_vector.extend(pred.detach().cpu().numpy())
    feature_vector = np.array(feature_vector)
    # labels_vector = np.array(labels_vector)
    pred_vector = np.array(pred_vector)
    return feature_vector, pred_vector#,labels_vector


def visualize_cluster_center():
    with torch.no_grad():
        cluster_center = net.compute_cluster_center(alpha)
        reconstruction = net.decode(cluster_center)
        print(reconstruction.shape)

    plt.figure()
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(
            reconstruction[i]
            .detach()
            .cpu()
            .numpy()
            # .reshape(dataset[0][0].shape[1], dataset[0][0].shape[2]),
            ,cmap="gray",
        )
    plt.savefig("./cluster_center.png")
    plt.close()


if __name__ == "__main__":
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    reload = False
    train_dataset = MyDataset("/home/sunz19/PALD_NN/fitted_data_train.csv")
    test_dataset = MyDataset("/home/sunz19/PALD_NN/fitted_data_test.csv")
    dataset = ConcatDataset([train_dataset, test_dataset])
    # class_num = 10
    # batch_size = 256
    class_num = 10
    batch_size = 20
    data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    data_loader_test = DataLoader(
        test_dataset, batch_size=50, shuffle=False, drop_last=False
    )
    net = Net(dim=8, class_num=class_num).cuda()
    optimizer = torch.optim.Adadelta(net.parameters())
    criterion = nn.MSELoss(reduction="mean")
    start_epoch = 0
    epochs = 3001
    alpha = 0.001
    beta = 0.001
    gamma = 1
    net.normalize_cluster_center(alpha)
    # print(len(data_loader))
    if reload:
        model_fp = os.path.join("./save/checkpoint_3000.tar")
        checkpoint = torch.load(model_fp)
        net.load_state_dict(checkpoint["net"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
    for epoch in range(start_epoch, epochs):
        labellist = []
        trainlist = []
        loss_clu_epoch = loss_rec_epoch = loss_pald_epoch = loss_modular_epoch = 0
        net.train()
        for step, x in enumerate(data_loader):
            trainlist.append(x.detach().cpu().numpy())
            x = x.float().cuda()
            z = net.encode(x)

            if epoch % 2 == 1:
                cluster_batch = net.cluster(z)
            else:
                cluster_batch = net.cluster(z.detach())
            soft_label = F.softmax(cluster_batch.detach(), dim=1)
            hard_label = torch.argmax(soft_label, dim=1)
            # print(soft_label.shape)
            # print(hard_label,"hl")
            hl = hard_label.detach().cpu().numpy()
            modular_loss = 0
            modu = modularity(np.ones((batch_size,batch_size)),hl)
            modular_loss += gamma * (1-modu)
            df = cohesion_matrix(pd.DataFrame(x.detach().cpu().numpy()))
            listuse = []
            for num in np.unique(hl):
                list1 = np.where(hl == num)[0]
                if len(list1) > 1:
                    listuse.append(list(list1))
            listread = []
            for a in listuse:
                for pair in index_pair(a):
                    listread.append(df.iloc[pair[0], pair[1]])
            pald_loss = 0
            for pl in listread:
                pald_loss += beta * 1.0 / (1+pl)
            labellist.append(hl)
            delta = torch.zeros((batch_size, 10), requires_grad=False).cuda()
            for i in range(batch_size):
                delta[i, torch.argmax(soft_label[i, :])] = 1
            # print('delta shape',delta.shape)
            # print(delta)
            # print('cluster_batch shape',cluster_batch.shape)
            # print(cluster_batch)
            loss_clu_batch = 2 * alpha - torch.mul(delta, cluster_batch)
            # print('cluster loss shape',loss_clu_batch.shape)
            loss_clu_batch = 0.01 / alpha * loss_clu_batch.mean()

            x_ = net.decode(z)
            loss_rec = criterion(x, x_)

            loss = loss_rec + loss_clu_batch + pald_loss +modular_loss
            optimizer.zero_grad()
            loss.backward()
            if epoch % 2 == 0:
                net.cluster_layer.weight.grad = (
                    F.normalize(net.cluster_layer.weight.grad, dim=1) * 0.2 * alpha
                )
            else:
                net.cluster_layer.zero_grad()
            optimizer.step()
            net.normalize_cluster_center(alpha)
            loss_clu_epoch += loss_clu_batch.item()
            loss_rec_epoch += loss_rec.item()
            loss_pald_epoch += pald_loss.item()
            loss_modular_epoch += modular_loss.item()
        print(
            f"Epoch [{epoch}/{epochs}]\t Clu Loss: {loss_clu_epoch / len(data_loader)}\t Rec Loss: {loss_rec_epoch / len(data_loader)}\t Pald Loss: {loss_pald_epoch / len(data_loader)}\t Modularaity Loss: {loss_modular_epoch / len(data_loader)}"
        )
        labellist = np.array(labellist).reshape(-1,1).flatten()
        trainlist = np.array(trainlist).reshape(-1,8)
        if epoch % 1000 == 0 and epoch >1:
            with torch.no_grad():
                cluster_center = net.compute_cluster_center(alpha)
                reconstruction = net.decode(cluster_center)
                X_embedded = TSNE(n_components=2).fit_transform(trainlist)
                # print(hard_label.shape,X_embedded.shape)
                assert X_embedded.shape[0] == labellist.shape[0], "warning: figure out Y shape"
                color_types = ["red","blue","orange","green","black","grey","brown","pink","cyan","purple"]
                plt.figure(figsize=(8,6))
                plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=5, c=[color_types[item] for item in labellist])
                plt.tight_layout()
                plt.savefig("/home/sunz19/PALD_NN/figures/trainsample.png",dpi=400)
                # plt.show()
                plt.close()
    labellist = []
    testlist = []
    for step, x in enumerate(data_loader):
        testlist.append(x.detach().cpu().numpy())
        x = x.float().cuda()
        z = net.encode(x)
        labellist.append(net.predict(z).detach().cpu().numpy())
    labellist = np.array(labellist).reshape(-1, 1).flatten()
    testlist = np.array(trainlist).reshape(-1, 8)
    # print(labellist.shape,testlist.shape)
    # print(labellist,testlist)
    X_embedded = TSNE(n_components=2).fit_transform(trainlist)
    assert X_embedded.shape[0] == labellist.shape[0], "warning: figure out Y shape"
    color_types = ["red", "blue", "orange", "green", "black", "grey", "brown", "pink", "cyan", "purple"]
    plt.figure(figsize=(8,6))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=5, c=[color_types[item] for item in labellist])
    plt.tight_layout()
    plt.savefig("/home/sunz19/PALD_NN/figures/testsample.png", dpi=400)
    # plt.show()
    plt.close()