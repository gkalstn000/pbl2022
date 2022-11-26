from models.networks.base_network import BaseNetwork
import torch
import torch.nn as nn
class PCAfeatureselect() :
    @staticmethod
    def modify_commandline_options(parser, is_train):
        opt, _ = parser.parse_known_args()
        return parser
    def __init__(self, opt):
        self.opt = opt
        self.n, self.p = opt.batchSize, opt.featur_size
        self.n_components = opt.n_component if opt.n_component > 0 else self.p

        ones = torch.ones(self.n).view([self.n, 1])
        h = ((1 / self.n) * torch.mm(ones, ones.t()))
        self.H = torch.eye(self.n) - h

    def __call__(self, x):
        # x = x.detach()
        X_center = torch.mm(self.H.double().to(x.device), x.double())
        u, s, v = torch.svd(X_center)
        components = v[:self.n_components].t()
        explained_variance = torch.mul(s[:self.n_components], s[:self.n_components]) / (self.n - 1)
        return {'X': x, 'k': self.n_components, 'components': components,
                'explained_variance': explained_variance}

        return x


    def PCA_eig(self, X, k, center=True, scale=False):
        n, p = X.size()
        ones = torch.ones(n).view([n, 1])
        h = ((1 / n) * torch.mm(ones, ones.t())) if center else torch.zeros(n * n).view([n, n])
        H = torch.eye(n) - h
        X_center = torch.mm(H.cuda().double(), X.double())
        covariance = 1 / (n - 1) * torch.mm(X_center.t(), X_center).view(p, p)
        scaling = torch.sqrt(1 / torch.diag(covariance)).double() if scale else torch.ones(p).double()
        scaled_covariance = torch.mm(torch.diag(scaling.cuda()).view(p, p), covariance)
        eigenvalues, eigenvectors = torch.linalg.eig(scaled_covariance)
        components = (eigenvectors[:, :k]).t()
        explained_variance = eigenvalues[:k, 0]
        return {'X': X, 'k': k, 'components': components,
                'explained_variance': explained_variance}