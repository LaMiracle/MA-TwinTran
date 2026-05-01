import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 指定GPU编号
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
import pandas as pd
import torch.cuda

from utils.preparedata import NASADataPreparer, DataPreparer
from MA_models import Model
from LSTM import LSTMModel

def main():
    # 指定文件夹路径
    download_save_path = '../../Dataset/Data_Download'
    exception_save_path = '../../Dataset/Exception_Data'
    print(os.path.abspath(download_save_path))
    # 获取文件夹下的所有文件名称
    download_folder_names = [item for item in os.listdir(download_save_path) if
                             os.path.isdir(os.path.join(download_save_path, item))]
    exception_folder_names = [item for item in os.listdir(exception_save_path) if
                              os.path.isdir(os.path.join(exception_save_path, item))]
    # # 生成所有文件夹路径
    # download_folder_paths = [os.path.join(download_save_path, item) for item in download_folder_names]
    # exception_folder_paths = [os.path.join(exception_save_path, item) for item in exception_folder_names]

    # 规定变量列表
    # variable_list = ['ALT', 'IVV', "TAS", 'GS', 'AOA1', 'AOA2', 'PTCH', 'WS', "WD", 'SAT', 'TAT', 'PI', 'PT']
    variable_list = ['TAS', 'GS', 'IVV', 'FPAC', 'CTAC',
                     'PTCH', 'AOAC', 'ROLL', 'DA', 'TH', 'TRK',
                     'WS', 'WD', 'SAT', 'TAT', 'PI', 'PT', 'ALT']

    # 规定工作数据集
    work_folder_name = 'Tail_652_4'

    # 生成原始数据集
    nasa_data_preparer = NASADataPreparer(download_save_path, exception_save_path, variable_list)

    # 规定训练出的模型以及预测结果csv的存储根目录
    result_data_savepath = "./Result"
    print(os.path.abspath(result_data_savepath))

    for test_fold_idx in range(10): # 10折
        # 提取训练/验证/测试集
        origin_data = nasa_data_preparer.prepareData(work_folder_name, test_fold_idx=test_fold_idx)

        # 规定时步频率
        time_freq = 1

        # 规定历史序列长度和预测序列长度
        decay_len, pred_len = 45, 15
        for pred_len in [5, 10, 15]:
        # for pred_len in [15]:
            '''
            如果pred_len>decay_len，会非常难收敛
            (8, 8)下batch=64, epoch=75
            (100, 100)下batch=32, epoch=75
            '''
            # 规定训练集、验证集和测试集分割方法
            num_dataset = len(origin_data)
            train_end = len(nasa_data_preparer.train_data)
            val_end = train_end + len(nasa_data_preparer.val_data)

            download_data_preparer = DataPreparer(batch_size=512,
                                        decay_len=decay_len, pred_len=pred_len,
                                        data_list=origin_data,
                                        n_feature=len(variable_list),
                                        out_col_slice=slice(0, len(variable_list)),
                                        train_end=train_end, val_end=val_end, test_end=num_dataset)

            # 获取对数据集进行标准化的均值和标准差
            dataset_scaler = download_data_preparer.scaler
            # train_mean, train_std = download_data_preparer.scaler.mean_, download_data_preparer.scaler.scale_


            # 规定模型类型
            model_type = 'MA-TwinTran'

            # 规定预测数据集的存储位置
            model_savepath = os.path.join(result_data_savepath, model_type)
            fold_model_savepath = os.path.join(model_savepath, 'fold_' + str(test_fold_idx))
            prediction_csv_savepath = os.path.join(fold_model_savepath, 'prediction_csv')
            if not os.path.exists(model_savepath):
                os.makedirs(model_savepath)
            if not os.path.exists(fold_model_savepath):
                os.makedirs(fold_model_savepath)
            if not os.path.exists(prediction_csv_savepath):
                os.makedirs(prediction_csv_savepath)

            if model_type == 'LSTM':
                model = LSTMModel(download_data_preparer,
                               decay_len, pred_len, time_freq, variable_list, isTrain=True, isPlot=True)
            else:
                model = Model(download_data_preparer,
                               decay_len, pred_len, time_freq,
                               variable_list,
                               isTrain=True, isImproveTrain=True,
                               earlyStop=False,
                               isNPODE=False, isMechanism=True,
                               isPlot=False)

            model.trainProcessBranch(work_folder_name, fold_model_savepath=fold_model_savepath)
            scaled_preds_list, scaled_labels_list = model._test()

            # 将预测值和真实值还原回原本量纲以计算准确的RMSE
            preds_list = [dataset_scaler.inverse_transform(preds) for preds in scaled_preds_list]
            labels_list = [dataset_scaler.inverse_transform(labels) for labels in scaled_labels_list]

            for i, scaled_preds in enumerate(scaled_preds_list):
                df = pd.DataFrame(dict(zip(variable_list, scaled_preds.T)))
                test_mat_name = nasa_data_preparer.test_mat_name_list[i]
                df.to_csv(os.path.join(prediction_csv_savepath, f'scaled_preds_{pred_len}_{test_mat_name}.csv'),
                          mode='a', header=False, index=False)
                # df.to_csv(f'./prediction_csv/scaled_preds_foldIdx{test_fold_idx}_{test_mat_name}_p45-15.csv', mode='a', header=False, index=False)
            for i, scaled_labels in enumerate(scaled_labels_list):
                df = pd.DataFrame(dict(zip(variable_list, scaled_labels.T)))
                test_mat_name = nasa_data_preparer.test_mat_name_list[i]
                df.to_csv(os.path.join(prediction_csv_savepath, f'scaled_labels_{pred_len}_{test_mat_name}.csv'),
                          mode='a', header=False, index=False)
                # df.to_csv(f'./prediction_csv/scaled_labels_foldIdx{test_fold_idx}_{test_mat_name}_p45-15.csv', mode='a', header=False, index=False)
            for i, preds in enumerate(preds_list):
                df = pd.DataFrame(dict(zip(variable_list, preds.T)))
                test_mat_name = nasa_data_preparer.test_mat_name_list[i]
                df.to_csv(os.path.join(prediction_csv_savepath, f'preds_{pred_len}_{test_mat_name}.csv'),
                          mode='a', header=False, index=False)
                # df.to_csv(f'./prediction_csv/preds_foldIdx{test_fold_idx}_{test_mat_name}_p45-15.csv', mode='a', header=False, index=False)
            for i, labels in enumerate(labels_list):
                df = pd.DataFrame(dict(zip(variable_list, labels.T)))
                test_mat_name = nasa_data_preparer.test_mat_name_list[i]
                df.to_csv(os.path.join(prediction_csv_savepath, f'labels_{pred_len}_{test_mat_name}.csv'),
                          mode='a', header=False, index=False)
                # df.to_csv(f'./prediction_csv/labels_foldIdx{test_fold_idx}_{test_mat_name}_p45-15.csv', mode='a', header=False, index=False)

            # scaled_flight_preds_list, scaled_flight_labels_list = model.flightTest()

            # patch_array, cluster_labels, cluster_mean_list = UncertaintyEvaluate(scaled_preds_list, scaled_labels_list, 10, isMovingWindow=False)

        #     break
        # break

if __name__ == '__main__':
    print(f'Current GPU id: {torch.cuda.current_device()}')
    main()