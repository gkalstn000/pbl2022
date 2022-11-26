import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.fc_layer import FC_layer
import util.util as util

class diffusionEstimator(BaseNetwork) :
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--Autoencoder_layers', type=str, default='512, 256, 128, 50', help='# of En/Decoder layer')
        opt, _ = parser.parse_known_args()

        return parser
    def __init__(self, opt):
        self.opt = opt

    def forward(self, x):
        return 0

class AutoencoderEstimator(BaseNetwork) :
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--Autoencoder_layers', type=str, default='512, 256, 128, 50', help='# of En/Decoder layer')
        opt, _ = parser.parse_known_args()

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.layers = [opt.featur_size]
        for num_node in opt.Autoencoder_layers.split(',') :
            self.layers.append(int(num_node))

        for i in range(len(self.layers)-1):
            block = FC_layer(self.layers[i], self.layers[i+1])
            setattr(self, 'encode' + str(i), block)

        for i in range(len(self.layers)-1):
            block = FC_layer(self.layers[::-1][i], self.layers[::-1][i+1])
            setattr(self, 'decode' + str(i), block)
    def forward(self, x):

        for i in range(len(self.layers)-1):
            model = getattr(self, 'encode' + str(i))
            x = model(x)

        for i in range(len(self.layers)-1):
            model = getattr(self, 'decode' + str(i))
            x = model(x)

        return x
