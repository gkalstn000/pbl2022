from models.networks.base_network import BaseNetwork
from models.networks.fc_layer import FC_layer
import torch.nn as nn
import torch

class RNNsequential(BaseNetwork) :
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--input_size', default= 7, help="dimension of sequential data")
        parser.add_argument('--num_rnn_layer', default=4, help="dimension of sequential data")

        return parser
    def __init__(self, opt):
        super(RNNsequential, self).__init__()
        self.opt = opt
        self.num_layer = opt.num_rnn_layer

        for i in range(self.num_layer):
            block = FC_layer(opt.n_component, opt.n_component)
            setattr(self, 'hidden' + str(i), block)

        self.rnn = nn.RNN(input_size= opt.input_size,
                          hidden_size=opt.n_component,
                          num_layers = 4,
                          dropout=0.3,
                          batch_first=True)
        self.outlayer = FC_layer(opt.n_component, 1)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x, init_state):
        init_hidden = []
        for i in range(self.num_layer):
            model = getattr(self, 'hidden' + str(i))
            init_hidden.append(model(init_state))
        init_hidden = torch.stack(init_hidden, 0)

        output, _ = self.rnn(x, init_hidden)

        output = self.outlayer(output.contiguous().view(-1, 50))
        output = output.contiguous().view(-1, 5)
        output = self.softmax(output)

        return output