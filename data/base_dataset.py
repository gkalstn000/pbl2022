"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.utils.data as data
import numpy as np
import pandas as pd

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        pass
    def preprocess(self, dataframe):
        '''
        필요없는 column drop
        data type 변경
        :param dataframe:
        :return:
        '''
        # Column 이름 변경
        dataframe = dataframe.replace('*', np.nan)
        dataframe.columns = new_columns

        # 필요없는 column 삭제, 설명회 기반
        drop_columns = ['before_cTNM', 'TNM', 'CEA', 'CA19']
        dataframe = dataframe[[column for column in dataframe.columns if column not in drop_columns]]

        # Column split / 환자데이터(환자기본정보+환자수술이력), 수술데이터(수술 전 데이터, 수술 데이터, 수술 후 데이터, 수술 후 분석 데이터), sequential, label, other(ESD_EMR, gasout, sd_start)
        patient_columns = dataframe.columns[:25].drop('Prediction_day')
        surgery_columns = dataframe.columns[25:36].append(dataframe.columns[37:50]).append(dataframe.columns[52:82])
        sequential_columns = dataframe.columns[82:-2]
        label_columns = ['DSL', ' onset', 'Prediction_day']
        other_columns = ['ESD_EMR', 'gasout', 'sd_start']

        df_patient = dataframe[patient_columns]
        df_surgery = dataframe[surgery_columns]
        df_sequential = dataframe[sequential_columns]
        df_label = dataframe[label_columns]
        df_other = dataframe[other_columns]

        assert len(dataframe.columns) == len(df_patient.columns) + len(df_surgery.columns) + len(df_sequential.columns) + len(df_label.columns) + len(df_other.columns)

        # patient columns 에서 필요없는 column drop
        drop_columns = ['Patient_number', 'hospitalization_data', 'surgery_date', 'discharge_date']
        df_patient = df_patient.drop(drop_columns, axis = 1)

        # surgery columns 에서 필요없는 column drop
        drop_columns = ['post_ESD', 'before_AGC', 'before_EGC', 'before_tubular', 'before_circular', 'before_clipping', 'before_EUS', 'before_PET_CT',
                        'cancer_count', 'after_AGC1', 'after_EGC1', 'after_tubular1', 'after_circular1', 'after_AGC2', 'after_EGC2', 'after_tubular2', 'after_circular2', 'after_AGC3', 'after_EGC3', 'after_tubular3', 'after_circular3',
                        'WHO_classification', 'WHO_cell_dist', 'L_classification', 'M_classification', 'lymphatics_invasion', 'surgery_time']
        df_surgery = df_surgery.drop(drop_columns, axis = 1)


        # category setting + make dummy variable
        # patient data 전처리(수치형으로 변경)
        numercial_float = ['age', 'ASA_score', 'height', 'weight', 'BMI', ]
        categorycal_int = ['sex', 'smoking', 'cancer_heredity', 'HTN', 'DM', 'Dyslipidemia', 'cardiovascular_disease',
                           'cerebrovascular_disease', 'kidney_disease', 'respiratory_disease',
                           'primary_cancer', 'other_disease', 'abdominal_surgery_history',
                           'abdominal_surgery_history stomach', 'other_surgery_history']

        for column in categorycal_int:
            df_patient[column] = np.where((~df_patient[column].isnull()) & (df_patient[column] != '0'), '1', df_patient[column])
        df_patient = df_patient.astype(np.float)

        # surgery data 전처리(수치형, dummy variable로 변경)
        # margin 나누기(1번, 2번, 3번)
        margin_p = df_surgery['margin_p'].str.split('/', n = 3, expand=True)
        margin_p.columns = [f'margin_p{i}' for i in range(1, 4)]
        margin_p = margin_p.fillna(0)
        margin_d = df_surgery['margin_d'].str.split('/', n = 3, expand=True)
        margin_d.columns = [f'margin_d{i}' for i in range(1, 4)]
        margin_d = margin_d.fillna(0)
        df_surgery = pd.concat([df_surgery.drop(['margin_p', 'margin_d'], axis = 1), margin_p, margin_d], axis = 1)
        numercial_column = ['bleeding', 'size1', 'size2', 'size3', 'margin_p1', 'margin_p2','margin_p3', 'margin_d1', 'margin_d2', 'margin_d3', 'g_lymph', 'm_lymph']
        df_surgery[numercial_column]=df_surgery[numercial_column].astype(np.float)

        category_column = [col for col in df_surgery.columns if col not in numercial_column]
        df_surgery = pd.get_dummies(df_surgery, columns=category_column)

        # sequential data
        drop_columns = ['JP_Amy_Rt_POD#1', 'JP_Amy_Rt_POD#2', 'JP_Amy_Rt_POD#3', 'JP_Amy_Rt_POD#5',
                        'JP_Rt_color change',
                        'JP_Lip_Rt_POD#1', 'JP_Lip_Rt_POD#2', 'JP_Lip_Rt_POD#3', 'JP_Lip_Rt_POD#5',
                        'JP_Amy_Lt_POD#1', 'JP_Amy_Lt_POD#2', 'JP_Amy_Lt_POD#3', 'JP_Amy_Lt_POD#5',
                        'JP_Lt_color change',
                        'JP_Lip_Lt_POD#2', 'JP_Lip_Lt_POD#3', 'JP_Lip_Lt_POD#5']
        df_sequential = df_sequential.drop(drop_columns, axis = 1)
        df_sequential = df_sequential.astype(np.float)

        # label
        df_label = df_label.fillna(0)
        df_label = df_label.astype(np.float)

        # other data
        df_other = df_other.astype(np.float)

        return df_patient, df_surgery, df_sequential, df_label, df_other

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