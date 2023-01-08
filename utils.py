import torch
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

from MI.kde import mi_kde


def train(net, optimizer, criterion, data):
    net.train()
    optimizer.zero_grad()
    output = net(data.x, data.adj)
    loss = criterion(output[data.train_mask], data.y[data.train_mask])
    acc = accuracy(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss, acc


def val(net, criterion, data):
    net.eval()
    output = net(data.x, data.adj)
    loss_val = criterion(output[data.val_mask], data.y[data.val_mask])
    acc_val = accuracy(output[data.val_mask], data.y[data.val_mask])
    return loss_val, acc_val


def test(net, criterion, data):
    net.eval()
    output = net(data.x, data.adj)
    loss_test = criterion(output[data.test_mask], data.y[data.test_mask])
    acc_test = accuracy(output[data.test_mask], data.y[data.test_mask])
    return loss_test, acc_test


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def measure_row_diff(net, data):
    net.eval()
    with torch.no_grad():
        h = net(data.x, data.adj)

    h = h  # [data.test_mask]
    h = h.data.cpu().numpy()
    # data_x = data.x[data.test_mask].data.cpu().numpy()
    # data_x = data.x.data.cpu().numpy()
    # print("h shape = ", h.shape)
    # print("data_x shape = ", data_x.shape)
    pair_dist = pairwise_distances(h)
    sum_dist = np.sum(pair_dist)
    n = h.shape[0]
    row_diff = sum_dist / (n * n)

    return row_diff


def measure_col_diff(net, data):
    net.eval()
    with torch.no_grad():
        h = net(data.x, data.adj)

    h = h  # [data.test_mask]
    h = h.data.cpu().numpy()
    print("h shape = ", h.shape)
    h_norm1 = normalize(h, norm='l1', axis=0)

    print("h_norm1 shape = ", h_norm1.shape)
    h_t = h_norm1.transpose()
    print("h_t shape = ", h_t.shape)
    # data_x = data.x[data.test_mask].data.cpu().numpy()
    # data_x = data.x.data.cpu().numpy()

    pair_dist = pairwise_distances(h_t)
    print(pair_dist.shape)
    sum_dist = np.sum(pair_dist)
    d = h_t.shape[0]
    col_diff = sum_dist / (d * d)

    return col_diff


# compute the instance information gain
def compute_iig(net, data):
    net.eval()
    data_x = data.x.data.cpu().numpy()
    with torch.no_grad():
        layers_self = net(data.x, data.adj)
    layer_self = layers_self.data.cpu().numpy()
    iig = mi_kde(layer_self, data_x, var=0.1)
    return iig


# compute group distance ratio
def compute_gdr(net, data, data_name):
    dis_intra, dis_inter = dis_cluster(net, data)
    if data_name == 'CoauthorCS':
        # if the intra-group and inter-group distances are close, we assign them the same values
        # and have the distance ratio of 1.
        distance_gap = dis_inter - dis_intra
        dis_ratio = 1. if distance_gap < 0.35 else dis_inter / dis_intra
    else:
        dis_ratio = dis_inter / dis_intra
    # if both dis_inter and dis_intra are close to zero, the value of dis_ratio is nan
    # in this case, we assign the distance ratio to 1.
    dis_ratio = 1. if np.isnan(dis_ratio) else dis_ratio
    return dis_ratio


def dis_cluster(net, data):
    net.eval()
    with torch.no_grad():
        X = net(data.x, data.adj)
    X_labels = []
    num_classes = int(data.y.max()) + 1
    for i in range(num_classes):
        X_label = X[data.y == i].data.cpu().numpy()
        h_norm = np.sum(np.square(X_label), axis=1, keepdims=True)
        h_norm[h_norm == 0.] = 1e-3
        X_label = X_label / np.sqrt(h_norm)
        X_labels.append(X_label)

    dis_intra = 0.
    for i in range(num_classes):
        x2 = np.sum(np.square(X_labels[i]), axis=1, keepdims=True)
        dists = x2 + x2.T - 2 * np.matmul(X_labels[i], X_labels[i].T)
        dis_intra += np.mean(dists)
    dis_intra /= num_classes

    dis_inter = 0.
    for i in range(num_classes - 1):
        for j in range(i + 1, num_classes):
            x2_i = np.sum(np.square(X_labels[i]), axis=1, keepdims=True)
            x2_j = np.sum(np.square(X_labels[j]), axis=1, keepdims=True)
            dists = x2_i + x2_j.T - 2 * np.matmul(X_labels[i], X_labels[j].T)
            dis_inter += np.mean(dists)
    num_inter = float(num_classes * (num_classes - 1) / 2)
    dis_inter /= num_inter

    # print('dis_intra: ', dis_intra)
    # print('dis_inter: ', dis_inter)
    return dis_intra, dis_inter


def reset_args(args):
    args.hid = 16
    if args.data == 'Citeseer' and args.missing_rate == 0.:
        args.skip_weight = 0.001 if args.nlayer < 6 else 0.005
    elif args.data == 'Citeseer' and args.missing_rate == 100:
        args.skip_weight = 0.01
    elif args.data == 'Pubmed' and args.missing_rate == 0:
        args.skip_weight = 0.005 if args.nlayer < 6 else 0.01
        args.num_groups = 5
    elif args.data == 'Pubmed' and args.missing_rate == 100:
        args.skip_weight = 0.03
        args.num_groups = 5
    elif args.data == 'Cora' and args.missing_rate == 0:
        args.skip_weight = 0.001 if args.nlayer < 6 else 0.01
    elif args.data == 'Cora' and args.missing_rate == 100:
        args.skip_weight = 0.01

    return args
