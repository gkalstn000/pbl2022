"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
from models.networks.base_network import BaseNetwork
import util.util as util


def find_network_using_name(target_network_name, filename):
    target_class_name = target_network_name + filename
    module_name = 'models.networks.' + filename
    network = util.find_class_in_module(target_class_name, module_name)

    assert issubclass(network, BaseNetwork), \
        "Class %s should be a subclass of BaseNetwork" % network

    return network


def modify_commandline_options(parser, is_train):
    opt, _ = parser.parse_known_args()

    net_estimator = find_network_using_name(opt.estimator, 'estimator')
    parser = net_estimator.modify_commandline_options(parser, is_train)

    # net_fs = find_network_using_name(opt.featureSelector, 'featureselect')
    # parser = net_fs.modify_commandline_options(parser, is_train)

    net_sequential = find_network_using_name(opt.sequence_model, 'sequential')
    parser = net_sequential.modify_commandline_options(parser, is_train)

    return parser


def create_network(cls, opt):
    net = cls(opt)
    net.print_network()
    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda()
    net.init_weights(opt.init_type, opt.init_variance)
    return net

def define_estimator(opt) :
    net_estimator = find_network_using_name(opt.estimator, 'estimator')
    return create_network(net_estimator, opt)
# Define Feature Selector
def define_FS(opt) :
    net_fs = find_network_using_name(opt.featureSelector, 'featureselect')
    return create_network(net_fs, opt)

def define_sequence(opt):
    netSequence_cls = find_network_using_name(opt.sequence_model, 'sequential')
    return create_network(netSequence_cls, opt)

# def define_G(opt):
#     netG_cls = find_network_using_name(opt.netG, 'generator')
#     return create_network(netG_cls, opt)
#
#
# def define_D(opt):
#     netD_cls = find_network_using_name(opt.netD, 'discriminator')
#     return create_network(netD_cls, opt)
#
#
# def define_E(opt):
#     # there exists only one encoder type
#     netE_cls = find_network_using_name('conv', 'encoder')
#     return create_network(netE_cls, opt)
