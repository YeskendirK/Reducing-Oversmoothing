from layers import *


class SGC(nn.Module):
    # for SGC we use data without normalization
    def __init__(self, nfeat, nhid, nclass, dropout, nlayer=2, norm_mode='None', norm_scale=10, **kwargs):
        super(SGC, self).__init__()
        self.linear = torch.nn.Linear(nfeat, nclass)
        self.norm = PairNorm(norm_mode, norm_scale)
        self.dropout = nn.Dropout(p=dropout)
        self.nlayer = nlayer

    def forward(self, x, adj):
        x = self.norm(x)
        for _ in range(self.nlayer):
            x = adj.mm(x)
            x = self.norm(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,
                 norm_mode='None', norm_scale=1, **kwargs):
        super(GCN, self).__init__()
        self.gc1 = GraphConv(nfeat, nhid)
        self.gc2 = GraphConv(nhid, nclass)

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU(True)
        self.norm = PairNorm(norm_mode, norm_scale)

    def forward(self, x, adj):
        x = self.dropout(x)
        x = self.gc1(x, adj)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.gc2(x, adj)
        return x


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nhead, 
                 norm_mode='None', norm_scale=1, relations={},**kwargs):
        super(GAT, self).__init__()
        alpha_droprate = dropout
        self.gac1 = GraphAttConv(nfeat, nhid, nhead, alpha_droprate, relations)
        self.gac2 = GraphAttConv(nhid, nclass, 1, alpha_droprate, relations)

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ELU(True)
        self.norm = PairNorm(norm_mode, norm_scale)

    def forward(self, x, adj):
        x = self.dropout(x)  # ?
        x = self.gac1(x, adj)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.gac2(x, adj)
        return x


class DeepGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayer=2, residual=0,
                 norm_mode='None', norm_scale=1, **kwargs):
        super(DeepGCN, self).__init__()
        assert nlayer >= 1
        self.hidden_layers = nn.ModuleList([
            GraphConv(nfeat if i == 0 else nhid, nhid)
            for i in range(nlayer - 1)
        ])
        self.out_layer = GraphConv(nfeat if nlayer == 1 else nhid, nclass)

        self.dropout = nn.Dropout(p=dropout)
        self.dropout_rate = dropout
        self.relu = nn.ReLU(True)
        self.norm = PairNorm(norm_mode, norm_scale)
        self.skip = residual

    def forward(self, x, adj):
        x_old = 0
        for i, layer in enumerate(self.hidden_layers):
            x = self.dropout(x)
            x = layer(x, adj)
            x = self.norm(x)
            x = self.relu(x)
            if self.skip > 0 and i % self.skip == 0:
                x = x + x_old
                x_old = x

        x = self.dropout(x)
        x = self.out_layer(x, adj)
        return x


class DeepGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayer=2, residual=0, nhead=1,
                 norm_mode='None', norm_scale=1, relations={}, num_groups=10, skip_weight=0.005, **kwargs):
        super(DeepGAT, self).__init__()
        assert nlayer >= 1
        alpha_droprate = dropout
        self.hidden_layers = nn.ModuleList([
            GraphAttConv(nfeat if i == 0 else nhid, nhid, nhead, alpha_droprate, relations)
            for i in range(nlayer - 1)
        ])
        self.out_layer = GraphAttConv(nfeat if nlayer == 1 else nhid, nclass, 1, alpha_droprate, relations)

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ELU(True)

        self.layers_bn = nn.ModuleList([])
        for i in range(nlayer):
            if i == nlayer-1:
                dim_out = self.out_layer.out_features
            else:
                dim_out = self.hidden_layers[i].out_features

            if norm_mode == 'group':
                self.layers_bn.append(GroupNorm(dim_out, num_groups=num_groups, skip_weight=skip_weight))
            else:
                self.layers_bn.append(PairNorm(norm_mode, norm_scale))



        # if norm_mode == 'group':
        #     self.norm = GroupNorm(nhid, num_groups=num_groups, skip_weight=skip_weight)
        # else:
        #     self.norm = PairNorm(norm_mode, norm_scale)
        self.skip = residual

    def forward(self, x, adj):
        x_old = 0
        for i, layer in enumerate(self.hidden_layers):
            x = self.dropout(x)
            x = layer(x, adj)
            x = self.layers_bn[i](x)
            x = self.relu(x)
            if self.skip > 0 and i % self.skip == 0:
                x = x + x_old
                x_old = x

        x = self.dropout(x)
        x = self.out_layer(x, adj)
        x = self.layers_bn[-1](x)
        return x
