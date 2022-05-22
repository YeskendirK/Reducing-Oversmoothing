import torch
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize


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
