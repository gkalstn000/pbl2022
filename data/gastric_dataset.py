from data.base_dataset import BaseDataset
import util.util as util
import pandas as pd
import numpy as np
import os
import torch


class GastricDataset(BaseDataset) :
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        df = pd.read_csv(os.path.join(opt.dataroot, opt.dataset_mode, f'{opt.phase}.csv'))
        df_patient, df_surgery, df_sequential, df_label, df_other = self.preprocess(df)

        df_static = pd.concat([df_patient, df_surgery, df_other], axis = 1)

        self.static_dataset = self.normalize_df(df_static).to_numpy()
        self.sequencial_dataset = self.normalize_df(df_sequential).to_numpy()
        self.label_dataset = self.normalize_df(df_label).to_numpy()

        stack = []
        for i in range(5) :
            start = i
            end = i + 7
            stack.append(self.sequencial_dataset[:, start: end])
        self.sequencial_dataset = np.stack(stack, axis = 1)

        assert self.static_dataset.shape[0] == self.sequencial_dataset.shape[0] == self.label_dataset.shape[0]
        self.length = self.static_dataset.shape[0]

    def __getitem__(self, index):
        static = self.static_dataset[index]
        sequence = self.sequencial_dataset[index]
        label = self.self.label_dataset[index]

        input_dict = {'static_input' : torch.Tensor(static),
                      'sequence_input' : torch.Tensor(sequence),
                      'label' : torch.Tensor(label)}

        return input_dict
    def __len__(self):
        return self.length
