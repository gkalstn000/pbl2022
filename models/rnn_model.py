"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch.nn as nn
import torch
import models.networks as networks
import util.util as util
import torch.nn.functional as F

class RNNModel(torch.nn.Module) :
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super(RNNModel, self).__init__()
        self.opt = opt

        self.estimtator = self.initialize_networks(opt) # 오타수정
        self.feature_seoector = None
        self.sequential_model = None

        if opt.isTrain:
            self.criterionMSE = nn.MSELoss()

    def forward(self, data, mode):
        static, sequence, label = self.preprocess_input(data)

        if mode == 'estimate' :
            E_losses, estimated = self.compute_estimator_loss(static)
            return E_losses, estimated

    def create_optimizers(self, opt):
        Estimator_params = list(self.estimtator.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        estimator_lr = opt.lr

        optimizer_estimator = torch.optim.Adam(Estimator_params, lr=estimator_lr, betas=(beta1, beta2))

        return optimizer_estimator

    def save(self, epoch):
        util.save_network(self.estimator, 'estimator', epoch, self.opt)

    def initialize_networks(self, opt):
        net_estimator = networks.define_estimator(opt)

        if not opt.isTrain or opt.continue_train:
            net_estimator = util.load_network(net_estimator, 'estimator', opt.which_epoch, opt)

        return net_estimator

    def preprocess_input(self, data):
        data['static_input'] = data['static_input'].float().cuda()
        data['sequence_input'] = data['sequence_input'].float().cuda()
        data['label'] = data['label'].float().cuda()

        return data['static_input'], data['sequence_input'], data['label']

    def compute_estimator_loss(self, static_data):
        E_losses = {}
        self.estimtator.train()

        nan_index = static_data.isnan().cuda()
        gaussian_tensor = torch.normal(0, 1, size = static_data.size()).cuda()

        static_data = torch.nan_to_num(static_data, 0) + torch.mul(nan_index, gaussian_tensor) # fill nan to gaussian variable
        estimated = self.estimate_feature(static_data)

        E_losses['Feature_BCE'] = self.criterionMSE(estimated[~nan_index], static_data[~nan_index])

        return E_losses, estimated
    def estimate_feature(self, static_data):
        estimated = self.estimtator(static_data)
        return estimated
    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0