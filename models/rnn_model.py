"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch.nn as nn
import torch
import models.networks as networks
import util.util as util
import torch.nn.functional as F
from models.networks.featureselect import PCAfeatureselect

class RNNModel(torch.nn.Module) :
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super(RNNModel, self).__init__()
        self.opt = opt

        self.estimator, self.feature_selector = self.initialize_networks(opt) # 오타수정
        self.sequential_model = None

        if opt.isTrain:
            self.criterionMSE = nn.MSELoss()

    def forward(self, data, mode):
        static, sequence, label = self.preprocess_input(data)

        if mode == 'estimate' :
            E_losses, estimated = self.compute_estimator_loss(static)
            self.estimated = estimated
            return E_losses, estimated
        elif mode == 'feature_select' :
            self.pca_dict = self.feature_selector(self.estimated)
            self.selected_features = self.pca_dict['components']
        elif mode == 'sequence' :
            return None
    def create_optimizers(self, opt):
        Estimator_params = list(self.estimator.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        estimator_lr = opt.lr

        optimizer_estimator = torch.optim.Adam(Estimator_params, lr=estimator_lr, betas=(beta1, beta2))

        return optimizer_estimator

    def save(self, epoch):
        util.save_network(self.estimator, 'estimator', epoch, self.opt)

    def initialize_networks(self, opt):
        net_estimator = networks.define_estimator(opt)
        net_feature_selector = PCAfeatureselect(opt)
        if not opt.isTrain or opt.continue_train:
            net_estimator = util.load_network(net_estimator, 'estimator', opt.which_epoch, opt)
            # net_feature_selector = util.load_network(net_feature_selector, 'FS', opt.which_epoch, opt)

        return net_estimator, net_feature_selector

    def preprocess_input(self, data):
        data['static_input'] = data['static_input'].float().cuda()
        data['sequence_input'] = data['sequence_input'].float().cuda()
        data['label'] = data['label'].float().cuda()

        return data['static_input'], data['sequence_input'], data['label']

    def compute_estimator_loss(self, static_data):
        E_losses = {}
        self.estimator.train()

        nan_index = static_data.isnan().cuda()
        gaussian_tensor = torch.normal(0, 1, size = static_data.size()).cuda()

        static_data = torch.nan_to_num(static_data, 0) + torch.mul(nan_index, gaussian_tensor) # fill nan to gaussian variable
        estimated = self.estimate_feature(static_data)

        E_losses['Feature_BCE'] = self.criterionMSE(estimated[~nan_index], static_data[~nan_index])

        return E_losses, estimated

    def select_features(self, estimated):
        selected = self.feature_selector(estimated)
        return selected
    def estimate_feature(self, static_data):
        estimated = self.estimator(static_data)
        return estimated
    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0