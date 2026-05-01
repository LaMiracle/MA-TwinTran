import os

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import root_mean_squared_error as RMSE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import r2_score

import numpy as np
from copy import deepcopy

from utils.earlystop import EarlyStopping
from forecast import ForecastConvTransformer
from mechanism import MechanismModel
from NP_ODE import NP_ODE_Model
from utils.plotter import training_plot, improve_training_plot, testing_plot

device = 'cpu'
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print(torch.cuda.device_count())
    torch.cuda.set_device(0)
print(torch.cuda.current_device())

class Utils:
    def __init__(self):
        pass

    def scaleDataReshape(self, data):
        data = np.array(data)
        data = data.reshape(-1, data.shape[-1])
        return data

class Model:
    def __init__(self, data_preparer, decay_len, pred_len, time_freq, variable_list,
                 isTrain=True, isImproveTrain=False, earlyStop=True, isNPODE=False, isMechanism=False, isPlot=True):
        '''
        同样epoch下，batch设32比设16更好。64比32更好，128没有64好。
        原因：自注意力机制在长时间序列上表现更好
        '''
        self.data_preparer = data_preparer
        self.decay_len = decay_len
        self.pred_len = pred_len
        self.time_freq = time_freq
        self.variable_list = variable_list
        self.utils = Utils()

        self.isTrain = isTrain
        self.isImproveTrain = isImproveTrain
        self.earlyStop = earlyStop
        self.isNPODE = isNPODE
        self.isMechanism = isMechanism
        self.isPlot = isPlot

        # 连续10次loss不下降就停止
        self.early_stopping = EarlyStopping(patience=3)

        self.train_inputs, self.train_labels = data_preparer.get_train()
        self.val_inputs_list, self.val_labels_list = data_preparer.get_val()
        self.test_inputs_list, self.test_labels_list = data_preparer.get_test()

        # 新建模型
        torch.manual_seed(3407)

        # 构建transformer模型
        self.transformer_model = ForecastConvTransformer(k=len(variable_list), headers=8, depth=4, decay_len=decay_len, pred_len=pred_len, kernel_size=9, mask_next=True, sparse=True)

        # 机理融合模型
        self.mechanism_model = MechanismModel(len(variable_list), pred_len)

        # NP_ODE模型
        input_dim, hidden_dim, output_dim = len(variable_list), 10, len(variable_list)
        self.NPODE_model = NP_ODE_Model(input_dim, hidden_dim, output_dim, device=device)
        self.pred_sequence = torch.linspace(1, pred_len, pred_len).to(device)

    def predict(self, model, inputs, labels, isNPODE=False, isMechanism=False, improve_model=None, pred_sequence=None, is_test=False):
        scaled_preds = []
        scaled_labels = []
        for input_tensor, label_tensor in zip(inputs, labels):
            output = model(input_tensor)

            if isNPODE:
                output = improve_model(output, pred_sequence)
            if isMechanism:
                mechanism_output = improve_model.mechanism(output, self.data_preparer.scaler)
                output = mechanism_output + improve_model(output - mechanism_output)

            scaled_preds.extend(output.detach().cpu().numpy())
            scaled_labels.extend(label_tensor.detach().cpu().numpy())
        scaled_preds = self.utils.scaleDataReshape(scaled_preds)
        scaled_labels = self.utils.scaleDataReshape(scaled_labels)
        return scaled_preds, scaled_labels

    def _train(self, epochs=100):
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.transformer_model.parameters(), lr=0.001)

        # 训练模型
        epochs = 100
        train_loss, val_loss = [], []
        performance_history = [[], []]
        for epoch in range(epochs):
            optimizer.zero_grad()
            epoch_loss = []
            batch_idx = 0
            for input_tensor, label_tensor in zip(self.train_inputs, self.train_labels):
                # print(input_tensor.size(), label_tensor.size())
                batch_idx += 1
                output = self.transformer_model(input_tensor)
                loss = criterion(label_tensor, output)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
                print \
                    ('Epoch [{}/{}], Batch [{}], Loss: {:.4f}'.format(epoch + 1, epochs, batch_idx,
                                                                      np.mean(epoch_loss)))
            train_loss.extend(epoch_loss)

            # 每个epoch都对训练集、验证集、测试集进行模型测试，并记录mse和mape
            with torch.no_grad():
                # 将模型设置为评估模式
                self.transformer_model.eval()

                # 使用模型对训练label进行预测
                train_scaled_preds, train_scaled_labels = self.predict(self.transformer_model, self.train_inputs, self.train_labels)
                # 使用模型对验证集label进行预测
                val_scaled_preds_list, val_scaled_labels_list = [], []
                for iData in range(self.data_preparer.val_end - self.data_preparer.train_end):
                    val_inputs, val_labels = self.val_inputs_list[iData], self.val_labels_list[iData]
                    val_scaled_preds, val_scaled_labels = self.predict(self.transformer_model, val_inputs, val_labels)
                    val_scaled_preds_list.append(val_scaled_preds)
                    val_scaled_labels_list.append(val_scaled_labels)

                if self.earlyStop:
                    # 如果各变量平均loss连续3次不下降则跳出循环
                    self.early_stopping(np.mean
                                   ([MSE(val_scaled_preds, val_scaled_labels) for val_scaled_preds, val_scaled_labels in
                                     zip(val_scaled_preds_list, val_scaled_labels_list)]))
                    if self.early_stopping.early_stop:
                        print(f'Early Stopping')
                        break

                # 使用模型对测试集label进行预测
                test_scaled_preds_list, test_scaled_labels_list = [], []
                for iData in range(self.data_preparer.test_end - self.data_preparer.val_end):
                    test_inputs, test_labels = self.test_inputs_list[iData], self.test_labels_list[iData]
                    test_scaled_preds, test_scaled_labels = self.predict(self.transformer_model, test_inputs, test_labels)
                    test_scaled_preds_list.append(test_scaled_preds)
                    test_scaled_labels_list.append(test_scaled_labels)

                # 分别计算均方误差
                mse_list = [MSE(train_scaled_preds, train_scaled_labels),
                            np.mean([MSE(val_scaled_preds, val_scaled_labels) for val_scaled_preds, val_scaled_labels in
                                     zip(val_scaled_preds_list, val_scaled_labels_list)]),
                            np.mean
                            ([MSE(test_scaled_preds, test_scaled_labels) for test_scaled_preds, test_scaled_labels in
                              zip(test_scaled_preds_list, test_scaled_labels_list)])]
                mape_list = [MAPE(train_scaled_preds, train_scaled_labels),
                             np.mean
                             ([MAPE(val_scaled_preds, val_scaled_labels) for val_scaled_preds, val_scaled_labels in
                               zip(val_scaled_preds_list, val_scaled_labels_list)]),
                             np.mean
                             ([MAPE(test_scaled_preds, test_scaled_labels) for test_scaled_preds, test_scaled_labels in
                               zip(test_scaled_preds_list, test_scaled_labels_list)])]
                performance_history[0].append(mse_list)
                performance_history[1].append(mape_list)

        performance_history = np.array(performance_history)

        return train_loss, performance_history # MSE & MAPE 指标

    def _improveTrain(self, epochs=100):
        # 定义损失函数和优化器
        # criterion = nn.CrossEntropyLoss() # 交叉熵表现很差，可能在分类问题中才适用
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.improve_model.parameters(), lr=0.001)

        # 训练模型
        train_loss, val_loss = [], []
        performance_history = [[], []]
        for epoch in range(epochs):
            optimizer.zero_grad()
            epoch_loss = []
            batch_idx = 0
            for input_tensor, label_tensor in zip(self.train_inputs, self.train_labels):
                # print(input_tensor.size(), label_tensor.size())
                batch_idx += 1
                output = self.transformer_model(input_tensor)

                if self.isNPODE:
                    self.improve_model = self.NPODE_model
                    output = self.improve_model(output, self.pred_sequence)
                elif self.isMechanism:
                    self.improve_model = self.mechanism_model
                    # 修改输入tensor，使其符合机制模型的输入要求
                    mechanism_revised_output = deepcopy(output.detach())
                    mechanism_revised_output = self.improve_model.mechanism(mechanism_revised_output,
                                                                            self.data_preparer.scaler)
                    output = mechanism_revised_output + self.improve_model(output - mechanism_revised_output)
                    # output = improve_model(output)
                    # print(sum(output[:, :, 7].cpu().detach().numpy() == input_tensor[:, :, 7].cpu().detach().numpy()))

                else: # 如果没有引入其他模型、还选择了训练提升，就用transfomer再训练一遍
                    self.transformer_model.train()

                loss = criterion(label_tensor, output)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
                print \
                    ('Epoch [{}/{}], Batch [{}], Loss: {:.4f}'.format(epoch + 1, epochs, batch_idx,
                                                                      np.mean(epoch_loss)))
            train_loss.extend(epoch_loss)

            # 每个epoch都对训练集、验证集、测试集进行模型测试，并记录mse和mape
            with torch.no_grad():
                self.transformer_model.eval()
                # if isNPODE or isMechanism:
                # 将模型设置为评估模式
                self.improve_model.eval()

                # 使用模型对训练label进行预测
                if self.isNPODE:
                    train_scaled_preds, train_scaled_labels = self.predict(self.transformer_model, self.train_inputs, self.train_labels, isNPODE=True,
                                                                      improve_model=self.improve_model,
                                                                      pred_sequence=self.pred_sequence)
                elif self.isMechanism:
                    train_scaled_preds, train_scaled_labels = self.predict(self.transformer_model, self.train_inputs, self.train_labels,
                                                                      isMechanism=True, improve_model=self.mechanism_model)
                else:
                    train_scaled_preds, train_scaled_labels = self.predict(self.transformer_model, self.train_inputs, self.train_labels)

                # 使用模型对验证集label进行预测
                val_scaled_preds_list, val_scaled_labels_list = [], []
                for iData in range(self.data_preparer.val_end - self.data_preparer.train_end):
                    val_inputs, val_labels = self.val_inputs_list[iData], self.val_labels_list[iData]
                    if self.isNPODE:
                        val_scaled_preds, val_scaled_labels = self.predict(self.transformer_model, val_inputs, val_labels, isNPODE=True,
                                                                      improve_model=self.improve_model,
                                                                      pred_sequence=self.pred_sequence)
                    elif self.isMechanism:
                        val_scaled_preds, val_scaled_labels = self.predict(self.transformer_model, val_inputs, val_labels, isMechanism=True,
                                                                      improve_model=self.mechanism_model)
                    else:
                        val_scaled_preds, val_scaled_labels = self.predict(self.transformer_model, val_inputs, val_labels)
                    val_scaled_preds_list.append(val_scaled_preds)
                    val_scaled_labels_list.append(val_scaled_labels)

                if self.earlyStop:
                    # 如果各变量平均loss连续一定次数不下降则跳出循环
                    self.early_stopping(np.mean
                                   ([MSE(val_scaled_preds, val_scaled_labels) for val_scaled_preds, val_scaled_labels in
                                     zip(val_scaled_preds_list, val_scaled_labels_list)]))
                    if self.early_stopping.early_stop:
                        print(f'Early Stopping')
                        break

                # 分别计算均方误差
                mse_list = [MSE(train_scaled_preds, train_scaled_labels),
                            np.mean([MSE(val_scaled_preds, val_scaled_labels) for val_scaled_preds, val_scaled_labels in
                                     zip(val_scaled_preds_list, val_scaled_labels_list)]),
                            ]
                mape_list = [MAPE(train_scaled_preds, train_scaled_labels),
                             np.mean
                             ([MAPE(val_scaled_preds, val_scaled_labels) for val_scaled_preds, val_scaled_labels in
                               zip(val_scaled_preds_list, val_scaled_labels_list)]),
                             ]
                performance_history[0].append(mse_list)
                performance_history[1].append(mape_list)

        performance_history = np.array(performance_history)

        return train_loss, performance_history

    def trainProcessBranch(self, train_folder_name, fold_model_savepath="../../../Result/fold_0"):
        # 确定模型和预测数据存储位置
        '''
        - /fold idx
            - base_model.pt
            - improve_model.pt
            - /train_process
                - base_train_loss.npy
                - base_performance_history.npy
                - improve_train_loss.npy
                - improve_performance_history.npy
            - /prediction_csv
                - {test_idx}_{mat_name}.csv
                ...
            - README.md
                - train_folder_name
                - decay timesteps / prediction timesteps
                - fold splitting
        '''
        train_process_savepath = os.path.join(fold_model_savepath, 'train_process')
        if not os.path.exists(train_process_savepath):
            os.makedirs(train_process_savepath)

        if self.isTrain:
            train_loss, performance_history = self._train(epochs=100)
            # 保存模型
            torch.save(self.transformer_model.state_dict(), os.path.join(fold_model_savepath, f'base_model_{self.pred_len}.pt'))

            # 打印train_loss, mse和mape历史
            np.save(os.path.join(train_process_savepath, f'base_train_loss_{self.pred_len}.npy'), train_loss)
            np.save(os.path.join(train_process_savepath, f'base_performance_history_{self.pred_len}.npy'), performance_history)
            training_plot(train_loss, performance_history)

        else: # 加载transformer模型
            self.transformer_model.load_state_dict(torch.load(os.path.join(fold_model_savepath, f'base_model_{self.pred_len}.pt')))
            self.transformer_model.eval() # 固定dropout和归一化层

        if self.isImproveTrain:
            '''融合机理模型/NP-ODE模型'''
            if self.isNPODE:
                self.improve_model = self.NPODE_model
            elif self.isMechanism:
                self.improve_model = self.mechanism_model

            train_loss, performance_history = self._improveTrain(epochs=100)

            if self.isNPODE or self.isMechanism:
                # 保存模型
                torch.save(self.improve_model.state_dict(), os.path.join(fold_model_savepath, f'improve_model_{self.pred_len}.pt'))

            np.save(os.path.join(train_process_savepath, f'improve_train_loss_{self.pred_len}.npy'), train_loss)
            np.save(os.path.join(train_process_savepath, f'improve_performance_history_{self.pred_len}.npy'), performance_history)
            improve_training_plot(train_loss, performance_history)

        else: # 如果不对提升模型进行提升
            # 加载机理模型
            if self.isMechanism:
                self.mechanism_model.load_state_dict(torch.load(os.path.join(fold_model_savepath, f'improve_model_{self.pred_len}.pt')))
                self.mechanism_model.eval() # 固定dropout和归一化层
                self.improve_model = self.mechanism_model
            elif self.isNPODE:
                self.NPODE_model.load_state_dict(torch.load(f'./Models/NP-ODE/improve_model_{self.pred_len}.pt'))
                self.NPODE_model.eval() # 固定dropout和归一化层
                self.improve_model = self.NPODE_model

    def _test(self):
        # 禁用自动求导
        with torch.no_grad():
            # 将模型设置为评估模式
            self.transformer_model.eval()
            if self.isNPODE or self.isMechanism:
                # 将模型设置为评估模式
                self.improve_model.eval()

            # 记录所有测试集上的测试数据
            test_scaled_preds_list, test_scaled_labels_list = [], []

            for test_inputs, test_labels in zip(self.test_inputs_list, self.test_labels_list):
                # 使用模型对测试集label进行预测
                if self.isNPODE:
                    scaled_preds, scaled_labels = self.predict(self.transformer_model, test_inputs, test_labels,
                                                               isNPODE=True, improve_model=self.improve_model, pred_sequence=self.pred_sequence, is_test=True)
                elif self.isMechanism:
                    scaled_preds, scaled_labels = self.predict(self.transformer_model, test_inputs, test_labels,
                                                               isMechanism=True, improve_model=self.improve_model, is_test=True)
                else:
                    scaled_preds, scaled_labels = self.predict(self.transformer_model, test_inputs, test_labels, is_test=True)

                test_scaled_preds_list.append(scaled_preds)
                test_scaled_labels_list.append(scaled_labels)

                scaled_GT = []
                for label_tensor in test_labels:
                    batch_array = label_tensor.cpu().numpy()
                    batch_array = batch_array.reshape(-1, batch_array.shape[-1]) # 转化成[total time length, input_size]的形状
                    scaled_GT.append(batch_array)
                scaled_GT = np.concatenate(scaled_GT, axis=0)

                test_preds, test_GT = self.data_preparer.scaler.inverse_transform(scaled_preds), self.data_preparer.scaler.inverse_transform(scaled_GT)

                # 计算均方误差和r2值
                scaled_mse = MSE(scaled_preds, scaled_labels)
                rmse = RMSE(test_preds, test_GT)
                r2 = r2_score(scaled_preds, scaled_labels)
                r2_adjusted = 1 - (1 - r2) * (len(scaled_preds) - 1) / (len(scaled_preds) - len(self.variable_list) - 1)

                if self.isPlot:
                    testing_plot(self.variable_list, self.time_freq, scaled_GT, scaled_preds, scaled_mse, rmse, r2_adjusted)
            # break

        return test_scaled_preds_list, test_scaled_labels_list

    def _flightTest(self): # 进行全程时序预测
        with torch.no_grad():
            # 将模型设置为评估模式
            self.transformer_model.eval()
            if self.isNPODE or self.isMechanism:
                # 将模型设置为评估模式
                self.improve_model.eval()

            # 记录所有测试集上的测试数据
            test_scaled_preds_list, test_scaled_labels_list = [], []

            iData = 0
            end_length = 1 # 末尾的5分钟预测效果很差
            for test_inputs, test_labels in zip(self.test_inputs_list, self.test_labels_list):
                # 使用模型对测试集全程进行时序预测
                patch_data = self.data_preparer.data_list[self.data_preparer.val_end + iData]
                t_max = 120 # 追加窗口

                all_scaled_GT = self.data_preparer.scaler.transform(patch_data)[:-end_length]
                all_scaled_GT = all_scaled_GT[:t_max]

                all_scaled_preds = np.zeros(all_scaled_GT.shape)
                all_scaled_preds[:self.decay_len] = all_scaled_GT[:self.decay_len]

                # 使用模型对前120s航程进行滑动窗口时序预测，每隔5s/10s/15s追加一次
                step_len = 10
                for start in range(0, t_max - self.decay_len - self.pred_len + 1, step_len):
                    input_tensor = torch.from_numpy(all_scaled_GT[start:start + self.decay_len].astype(np.float32)).to(device) # 追加真实值
                    input_tensor = input_tensor.unsqueeze(0)
                    output = self.transformer_model(input_tensor, is_test=True)
                    if self.isMechanism:
                        mechanism_output = self.improve_model.mechanism(output, self.data_preparer.scaler)
                        output = mechanism_output + self.improve_model(output - mechanism_output)
                    output = output.to('cpu').detach().numpy()[0]
                    # print(pred_results.shape)
                    if start == 0:
                        all_scaled_preds[self.decay_len + start : self.decay_len + start + self.pred_len] = output[:self.pred_len, :]
                    else:
                        all_scaled_preds[self.decay_len + start + self.pred_len - step_len : self.decay_len + start + self.pred_len] = output[self.pred_len -step_len: self.pred_len, :] # 单点预测误差

                all_scaled_preds = np.array(all_scaled_preds)

                test_scaled_preds_list.append(all_scaled_preds)
                test_scaled_labels_list.append(all_scaled_GT)

                # 计算均方误差
                mse = MSE(all_scaled_preds, all_scaled_GT)
                r2 = r2_score(all_scaled_preds, all_scaled_GT)
                mse_rounded = round(mse, 3)
                r2_adjusted = 1 - (1 - r2) * (len(all_scaled_preds) - 1) / (len(all_scaled_preds) - len(self.variable_list) - 1)

                iData += 1
                # np.save(f'../result/@models/pred_len={pred_len}.npy', all_scaled_preds)
                # break
            return test_scaled_preds_list, test_scaled_labels_list
