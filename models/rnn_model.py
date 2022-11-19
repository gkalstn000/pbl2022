"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch.nn as nn
import torch
import models.networks as networks
import util.util as util

class RNNModel(torch.nn.Module) :
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        self.opt = opt

        self.estimtator = self.initialize_networks(opt)

        if opt.isTrain:
            self.criterionBCE = nn.BCELoss()

    def forward(self, data, mode):
        static, sequence, label = self.preprocess_input(data)

        if mode == 'estimate' :
            e_loss, estimated = self.compute_estimator_loss(static)
            return e_loss, estimated

    def create_optimizers(self, opt):
        Estimator_params = list(self.Estimator_params.parameters())

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
        pass

    def compute_estimator_loss(self, static_data):
        e_losses = {}

        return e_losses
    def estimate_feature(self, static_data):

        estimated = self.netG(static_data)

        return estimated
    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0