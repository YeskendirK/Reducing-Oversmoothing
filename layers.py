import torch
import torch.nn as nn
from torch_sparse import spmm  # require the newest torch_sprase
import numpy as np


class GraphAttConv(nn.Module):
    def __init__(self, in_features, out_features, heads, dropout, relations):
        super(GraphAttConv, self).__init__()
        assert out_features % heads == 0
        out_perhead = out_features // heads

        self.graph_atts = nn.ModuleList([GraphAttConvOneHead(
            in_features, out_perhead, dropout=dropout, relations=relations) for _ in range(heads)])

        self.in_features = in_features
        self.out_perhead = out_perhead
        self.heads = heads

    def forward(self, input, adj):
        output = torch.cat([att(input, adj) for att in self.graph_atts], dim=1)
        # notice that original GAT use elu as activation func. 
        return output

    def __repr__(self):
        return self.__class__.__name__ + "({}->[{}x{}])".format(
            self.in_features, self.heads, self.out_perhead)


class GraphAttConvOneHead(nn.Module):
    """
    Sparse version GAT layer, single head
    """

    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, relations={}):
        super(GraphAttConvOneHead, self).__init__()
        self.relations = relations
        self.concat_features = 2 * out_features
        for relation_type in relations.keys():
            if relations[relation_type] is True:
                self.concat_features += out_features
        self.weight = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.zeros(size=(1, self.concat_features)))
        # init 
        nn.init.xavier_normal_(self.weight.data, gain=nn.init.calculate_gain('relu'))  # look at here
        nn.init.xavier_normal_(self.a.data, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, input, adj):
        edge = adj._indices()
        h = torch.mm(input, self.weight)
        # Self-attention on the nodes - Shared attention mechanism
        final_features = [h[edge[0, :], :], h[edge[1, :], :]]
        if self.relations["difference"]:
            new_feature = torch.sub(h[edge[1, :], :], h[edge[0, :], :])
            final_features += [new_feature]
        if self.relations["abs_difference"]:
            new_feature = torch.sub(h[edge[1, :], :], h[edge[0, :], :])
            new_feature = torch.abs(new_feature)
            final_features += [new_feature]
        if self.relations["elem_product"]:
            new_feature = torch.mul(h[edge[0, :], :], h[edge[1, :], :])
            final_features += [new_feature]

        edge_h = torch.cat(final_features, dim=1).t()  # edge_h: 2*D x E
        # do softmax for each row, this need index of each row, and for each row do softmax over it
        alpha = self.leakyrelu(self.a.mm(edge_h).squeeze())  # E
        n = len(input)
        alpha = softmax(alpha, edge[0], n)
        output = spmm(edge, self.dropout(alpha), n, n, h)  # h_prime: N x out
        # output = spmm(edge, self.dropout(alpha), n, n, self.dropout(h)) # h_prime: N x out
        return output


"""
    helpers
"""
from torch_scatter import scatter_max, scatter_add


def softmax(src, index, num_nodes=None):
    """
        sparse softmax
    """
    num_nodes = index.max().item() + 1 if num_nodes is None else num_nodes
    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)
    return out
