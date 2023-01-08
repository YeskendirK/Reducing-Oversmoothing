import torch
import torch.nn as nn
from torch_sparse import spmm  # require the newest torch_sprase
import numpy as np
import torch.nn.functional as F


class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, out_features))

        self.in_features = in_features
        self.out_features = out_features
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        h = torch.mm(input, self.weight)
        output = torch.spmm(adj, h)
        if self.bias is not None:
            return output + self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + "({}->{})".format(
            self.in_features, self.out_features)


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
        # self.weight = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.weight_embedding = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.weight_relation = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.zeros(size=(1, self.concat_features)))
        # init 
        # nn.init.xavier_normal_(self.weight.data, gain=nn.init.calculate_gain('relu'))  # look at here
        nn.init.xavier_normal_(self.weight_embedding.data, gain=nn.init.calculate_gain('relu'))  # look at here
        nn.init.xavier_normal_(self.weight_relation.data, gain=nn.init.calculate_gain('relu'))  # look at here
        nn.init.xavier_normal_(self.a.data, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, input, adj):
        edge = adj._indices()
        # h = torch.mm(input, self.weight)
        h_e = torch.mm(input, self.weight_embedding)
        h_r = torch.mm(input, self.weight_relation)
        # Self-attention on the nodes - Shared attention mechanism
        # final_features = [h[edge[0, :], :], h[edge[1, :], :]]
        final_features = [h_e[edge[0, :], :], h_e[edge[1, :], :]]
        if self.relations["difference"]:
            # new_feature = torch.sub(h[edge[1, :], :], h[edge[0, :], :])
            new_feature = torch.sub(h_r[edge[1, :], :], h_r[edge[0, :], :])
            final_features += [new_feature]
        if self.relations["abs_difference"]:
            # new_feature = torch.sub(h[edge[1, :], :], h[edge[0, :], :])
            new_feature = torch.sub(h_r[edge[1, :], :], h_r[edge[0, :], :])
            new_feature = torch.abs(new_feature)
            final_features += [new_feature]
        if self.relations["elem_product"]:
            # new_feature = torch.mul(h[edge[0, :], :], h[edge[1, :], :])
            new_feature = torch.mul(h_r[edge[0, :], :], h_r[edge[1, :], :])
            final_features += [new_feature]

        edge_h = torch.cat(final_features, dim=1).t()  # edge_h: 2*D x E
        # do softmax for each row, this need index of each row, and for each row do softmax over it
        alpha = self.leakyrelu(self.a.mm(edge_h).squeeze())  # E
        n = len(input)
        alpha = softmax(alpha, edge[0], n)
        # output = spmm(edge, self.dropout(alpha), n, n, h)  # h_prime: N x out
        output = spmm(edge, self.dropout(alpha), n, n, h_e)  # h_prime: N x out
        # output = spmm(edge, self.dropout(alpha), n, n, self.dropout(h)) # h_prime: N x out
        return output


class GroupNorm(nn.Module):
    def __init__(self, dim_hidden, num_groups=1, skip_weight=0.005):
        """
        """
        super(GroupNorm, self).__init__()
        self.dim_hidden = dim_hidden
        self.num_groups = num_groups
        self.skip_weight = skip_weight

        self.bn = torch.nn.BatchNorm1d(dim_hidden * self.num_groups, momentum=0.3)
        self.group_func = torch.nn.Linear(dim_hidden, self.num_groups, bias=True)

    def forward(self, x):
        if self.num_groups == 1:
            x_temp = self.bn(x)
        else:
            score_cluster = F.softmax(self.group_func(x), dim=1)
            x_temp = torch.cat([score_cluster[:, group].unsqueeze(dim=1) * x for group in range(self.num_groups)],
                               dim=1)
            x_temp = self.bn(x_temp).view(-1, self.num_groups, self.dim_hidden).sum(dim=1)
        x = x + x_temp * self.skip_weight
        return x


class PairNorm(nn.Module):
    def __init__(self, mode='PN', scale=1):
        """
            mode:
              'None' : No normalization
              'PN'   : Original version
              'PN-SI'  : Scale-Individually version
              'PN-SCS' : Scale-and-Center-Simultaneously version

            ('SCS'-mode is not in the paper but we found it works well in practice,
              especially for GCN and GAT.)
            PairNorm is typically used after each graph convolution operation.
        """
        assert mode in ['None', 'PN', 'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

        # Scale can be set based on origina data, and also the current feature lengths.
        # We leave the experiments to future. A good pool we used for choosing scale:
        # [0.1, 1, 10, 50, 100]

    def forward(self, x):
        if self.mode == 'None':
            return x

        col_mean = x.mean(dim=0)
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean

        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual

        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean

        return x


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
