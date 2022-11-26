from data.base_dataset import BaseDataset
import util.util as util
import pandas as pd
import numpy as np
import os



class GastricDataset(BaseDataset) :
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        df = pd.read_csv(os.path.join(opt.dataroot, opt.dataset_mode, f'{opt.phase}.csv'))
        df_patient, df_surgery, df_sequential, df_label, df_other = self.preprocess(df)

        df_static = pd.concat([df_patient, df_surgery, df_other], axis = 1)
        
    # def initialize(self):
    #     # self.opt = opt
    #     df = pd.read_csv(os.path.join('./datasets', 'gastric', 'train.csv'))
    #     df_patient, df_surgery, df_sequential, df_label, df_other = self.preprocess(df)

    #     df_static = pd.concat([df_patient, df_surgery, df_other], axis = 1)



    def __getitem__(self, index):
        pass
    def __len__(self):
        return 0
