from layers import *


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
        self.skip = residual
        if norm_mode in ['None', 'PN', 'PN-SI', 'PN-SCS']:
            self.norm = PairNorm(norm_mode, norm_scale)
        elif norm_mode == "GN": # norm_mode
            self.norm = GroupNorm(nhid, num_groups=num_groups, skip_weight=skip_weight)
        else:
            Exception(f'Please include correct norm_mode instead of {self.norm}')

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