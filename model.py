import dgl
import torch
import torch.nn as nn


class GraphEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GraphEmbedding, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, input_size)

    def forward(self, x1, x2, x):
        x = x.to('cuda')
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class GraphConvolution():
    def __init__(self, dm, hv):
        pass

    def forward(self,graph, X):
        graph = dgl.add_self_loop(graph)
        A = graph.adjacency_matrix().to_dense()
        degrees = graph.in_degrees().float()
        D = torch.diag(degrees)
        D = D.to('cuda:0')
        A = A.to('cuda:0')
        X = X.to('cuda:0')

        norm0 = torch.pow(D, -0.5)
        norm0[torch.isinf(norm0)] = 0
        Q = torch.mm(norm0, A)
        Q = torch.mm(Q, norm0)
        Q = torch.mm(Q, X)
        Q = torch.relu(Q)
        return Q


def GCN(graph, X, dm, hv):
    gcn = GraphConvolution(dm, hv)
    X1 = gcn.forward(graph, X)
    X2 = gcn.forward(graph, X1)
    return X2


class GAE(nn.Module):
    def __init__(self, in_dim, hidden_dims_v, hidden_dims, views,):
        super(GAE, self).__init__()
        self.views = views
        self.hidden_v = hidden_dims_v
        self.in_dim = in_dim
        self. hidden_dims = hidden_dims
        self.featfusion = FeatureFusion(in_dim=in_dim, size=hidden_dims_v[1])

    def forward(self, graph0, graph1, feature0, feature1, graph):
        h0 = feature0
        h1 = feature1

        h0 = GCN(graph0, h0, self.in_dim[0], self.hidden_v)
        h1 = GCN(graph1, h1, self.in_dim[1], self.hidden_v)
        xh = self.featfusion(h0, h1)
        xh = GCN(graph, xh, xh.shape[1], xh.shape[1])

        return xh, h0, h1


class FeatureFusion(nn.Module):
    def __init__(self, in_dim, size, activation=torch.relu):
        super(FeatureFusion, self).__init__()
        self.w1 = 0.5
        self.w2 = 0.5
        self.fc1 = nn.Linear(in_dim[0], size)
        self.fc2 = nn.Linear(in_dim[1], size)

    def forward(self, z1, z2):
        z1 = self.fc1(z1)
        z2 = self.fc2(z2)
        F_weighted = (self.w1 * z1) + (self.w2 * z2)
        return F_weighted
