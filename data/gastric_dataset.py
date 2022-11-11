from data.base_dataset import BaseDataset
import util.util as util
import pandas as pd
import numpy as np
import os

new_columns = ['Patient_number', 'Prediction_day', 'sex', 'age', 'hospitalization_data', 'surgery_date', 'discharge_date', 'height', 'weight', 'BMI', 'smoking', 'cancer_heredity', # 환자 기본 정보
               'ASA_score', 'HTN', 'DM', 'Dyslipidemia', 'cardiovascular_disease', 'cerebrovascular_disease', 'kidney_disease', 'respiratory_disease', 'primary_cancer', 'other_disease', 'abdominal_surgery_history', 'abdominal_surgery_history stomach', 'other_surgery_history', # 환자 수술 이력
               'before_echocardiography', 'before_pulmonary', 'before_cTNM', 'before_chest_CT', 'post_ESD', 'before_AGC', 'before_EGC', 'before_tubular', 'before_circular', 'before_clipping', 'before_EUS', 'before_PET_CT', # 수술 전 데이터
               'ESD_EMR',
               'emergency', 'surgery_name', 'open_surgery', 'anastomosis', 'surgery_time', 'bleeding', 'consolidation_resection', 'adhesion', 'invasion', 'radicality', 'LN_dissection', 'vascular_mutation', 'transfusion', # 수술 데이터
               'gasout', 'sd_start',
               'cancer_count', 'after_AGC1', 'after_EGC1', 'after_tubular1', 'after_circular1', 'size1', 'after_AGC2', 'after_EGC2', 'after_tubular2', 'after_circular2', 'size2', 'after_AGC3', 'after_EGC3', 'after_tubular3', 'after_circular3', 'size3', # 수술 후 데이터
               'margin_p', 'margin_d', 'depth', 'TNM', 'Stage', 'g_lymph', 'm_lymph', 'WHO_classification', 'WHO_cell_dist', 'L_classification', 'M_classification',  'lymphatics_invasion', 'vascular_invasion', 'perinerual_invasion', 'additional_findings', # 수술 후 분석 데이터
               'CEA', 'CA19',
               'WBC_Pre', 'WBC_Post', 'WBC_POD#1', 'WBC_POD#2', 'WBC_POD#3', 'WBC_POD#5', 'WBC_POD#7', # ======아래부터 sequential data
               'Hb_Pre', 'Hb_Post', 'Hb_POD#1', 'Hb_POD#2', 'Hb_POD#3', 'Hb_POD#5', 'Hb_POD#7',
               'AST_Pre', 'AST_Post', 'AST_POD#1', 'AST_POD#2', 'AST_POD#3', 'AST_POD#5', 'AST_POD#7',
               'ALT_Pre', 'ALT_Post', 'ALT_POD#1', 'ALT_POD#2', 'ALT_POD#3', 'ALT_POD#5', 'ALT_POD#7',
               'CRP_Pre', 'CRP_Post', 'CRP_POD#1', 'CRP_POD#2', 'CRP_POD#3', 'CRP_POD#5', 'CRP_POD#7',
               'JP_Amy_Rt_POD#1', 'JP_Amy_Rt_POD#2', 'JP_Amy_Rt_POD#3', 'JP_Amy_Rt_POD#5', 'JP_Rt_color change',
               'JP_Lip_Rt_POD#1', 'JP_Lip_Rt_POD#2', 'JP_Lip_Rt_POD#3', 'JP_Lip_Rt_POD#5',
               'JP_Amy_Lt_POD#1', 'JP_Amy_Lt_POD#2', 'JP_Amy_Lt_POD#3', 'JP_Amy_Lt_POD#5', 'JP_Lt_color change',
               'JP_Lip_Lt_POD#2', 'JP_Lip_Lt_POD#3', 'JP_Lip_Lt_POD#5',
               'DSL', ' onset' # label
               ]

class GastricDataset(BaseDataset) :
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        df = pd.read_csv(os.path.join(opt.dataroot, opt.dataset_mode, f'{opt.phase}.csv'))
        static_data, sequence_data, label = self.preprocess(df)

    def preprocess(self, dataframe):
        # Column 이름 변경
        dataframe.columns = new_columns
        # 필요없는 column 삭제, 설명회 기반
        drop_columns = ['before_cTNM', 'TNM', 'CEA', 'CA19']
        dataframe = dataframe[[column for column in dataframe.columns if column not in drop_columns]]
        # Column split / 환자데이터(환자기본정보+환자수술이력), 수술데이터(수술 전 데이터, 수술 데이터, 수술 후 데이터, 수술 후 분석 데이터), sequential, label, other(ESD_EMR, gasout, sd_start)
        patient_columns = dataframe.columns[:25]
        surgery_columns = dataframe.columns[25:36].append(dataframe.columns[37:50]).append(dataframe.columns[52:82])
        sequential_columns = dataframe.columns[82:-2]
        label_columns = dataframe.columns[-2:]
        other_columns = ['ESD_EMR', 'gasout', 'sd_start']

        df_patient = dataframe[patient_columns]
        df_surgery = dataframe[surgery_columns]
        df_sequential = dataframe[sequential_columns]
        df_label = dataframe[label_columns]
        df_other = dataframe[other_columns]

        assert len(dataframe.columns) == len(df_patient.columns) + len(df_surgery.columns) + len(df_sequential.columns) + len(df_label.columns) + len(df_other.columns)

        # category setting + make dummy variable
        return None, None, None

    def __getitem__(self, index):
        pass
    def __len__(self):
        return 0
