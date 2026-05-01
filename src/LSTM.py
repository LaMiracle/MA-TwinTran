import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import root_mean_squared_error as RMSE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import r2_score
from utils.plotter import training_plot, testing_plot
from MC_ZonalTran import Utils

# 定义LSTM模型

class InitLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, pred_len, num_layers=3):
        super(InitLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.pred_len = pred_len
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len * output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # 初始化短时记忆隐藏层
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # 初始化长时记忆隐藏层

        out, _ = self.lstm(x, (h0, c0))  # LSTM前向传播，输出batch_num * decay_len * input_size
        # print(out.size())
        out = self.fc(out[:, -1, :])  # 取LSTM最后一个节点隐藏层的状态（batch_num * 1 * input_size），用FC层转化为对未来pred_len时段的预测
        # print(x.size(), out.size())
        out = out.reshape(-1, self.pred_len, self.output_size) # batch_num * pred_len * output_size

        return out

class LSTMModel:
    def __init__(self, download_data_preparer,
                               decay_len, pred_len, time_freq, variable_list, isTrain=True, isPlot=True, hidden_size=100, num_layers=3):
        self.hidden_size = hidden_size
        self.input_size = len(variable_list)
        self.output_size = len(variable_list)
        self.decay_len = decay_len
        self.pred_len = pred_len
        self.time_freq = time_freq
        self.variable_list = variable_list
        self.num_layers = num_layers

        self.data_preparer = download_data_preparer
        self.train_inputs, self.train_labels = self.data_preparer.get_train()
        self.val_inputs_list, self.val_labels_list = self.data_preparer.get_val()
        self.test_inputs_list, self.test_labels_list = self.data_preparer.get_test()

        self.utils = Utils()

        self.isTrain = isTrain
        self.isPlot = isPlot

        self.lstm = InitLSTM(self.input_size, self.hidden_size, self.output_size, self.pred_len, self.num_layers)

        torch.manual_seed(3407)  # 种子改良

    def predict(self, model, inputs, labels, is_test=False):
        scaled_preds = []
        scaled_labels = []
        for input_tensor, label_tensor in zip(inputs, labels):
            output = model(input_tensor)
            scaled_preds.extend(output.detach().cpu().numpy())
            scaled_labels.extend(label_tensor.detach().cpu().numpy())
        scaled_preds = self.utils.scaleDataReshape(scaled_preds)
        scaled_labels = self.utils.scaleDataReshape(scaled_labels)
        return scaled_preds, scaled_labels

    def _train(self, epochs=100):
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.lstm.parameters(), lr=0.001)

        # 训练模型
        train_loss, val_loss = [], []
        performance_history = [[], []]
        for epoch in range(epochs):
            self.lstm.train()
            optimizer.zero_grad()
            epoch_loss = []
            batch_idx = 0
            for input_tensor, label_tensor in zip(self.train_inputs, self.train_labels):
                # print(input_tensor.size(), label_tensor.size())
                batch_idx += 1
                output = self.lstm(input_tensor)
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
                self.lstm.eval()

                # 使用模型对训练label进行预测
                train_scaled_preds, train_scaled_labels = self.predict(self.lstm, self.train_inputs,
                                                                       self.train_labels)
                # 使用模型对验证集label进行预测
                val_scaled_preds_list, val_scaled_labels_list = [], []
                for iData in range(self.data_preparer.val_end - self.data_preparer.train_end):
                    val_inputs, val_labels = self.val_inputs_list[iData], self.val_labels_list[iData]
                    val_scaled_preds, val_scaled_labels = self.predict(self.lstm, val_inputs, val_labels)
                    val_scaled_preds_list.append(val_scaled_preds)
                    val_scaled_labels_list.append(val_scaled_labels)

                # if self.earlyStop:
                #     # 如果各变量平均loss连续3次不下降则跳出循环
                #     self.early_stopping(np.mean
                #                         ([MSE(val_scaled_preds, val_scaled_labels) for
                #                           val_scaled_preds, val_scaled_labels in
                #                           zip(val_scaled_preds_list, val_scaled_labels_list)]))
                #     if self.early_stopping.early_stop:
                #         print(f'Early Stopping')
                #         break

                # 使用模型对测试集label进行预测
                test_scaled_preds_list, test_scaled_labels_list = [], []
                for iData in range(self.data_preparer.test_end - self.data_preparer.val_end):
                    test_inputs, test_labels = self.test_inputs_list[iData], self.test_labels_list[iData]
                    test_scaled_preds, test_scaled_labels = self.predict(self.lstm, test_inputs,
                                                                         test_labels)
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

        return train_loss, performance_history  # MSE & MAPE 指标

    def trainProcessBranch(self, work_folder_name, fold_model_savepath="../../../@Result/@dataStorage_paper_MCZonalTran/LSTM/fold_0"):
        train_process_savepath = os.path.join(fold_model_savepath, 'train_process')
        if not os.path.exists(train_process_savepath):
            os.makedirs(train_process_savepath)

        if self.isTrain:
            train_loss, performance_history = self._train(epochs=100)
            # 保存模型
            torch.save(self.lstm.state_dict(), os.path.join(fold_model_savepath, 'base_model.pt'))

            # 打印train_loss, mse和mape历史
            np.save(os.path.join(train_process_savepath, 'base_train_loss.npy'), train_loss)
            np.save(os.path.join(train_process_savepath, 'base_performance_history.npy'), performance_history)
            training_plot(train_loss, performance_history)

        else:  # 加载lstm模型
            self.lstm.load_state_dict(torch.load(os.path.join(fold_model_savepath, 'base_model.pt')))
            self.lstm.eval()  # 固定dropout和归一化层

    def _test(self):
        # 禁用自动求导
        with torch.no_grad():
            # 将模型设置为评估模式
            self.lstm.eval()

            # 记录所有测试集上的测试数据
            test_scaled_preds_list, test_scaled_labels_list = [], []

            for test_inputs, test_labels in zip(self.test_inputs_list, self.test_labels_list):
                # 使用模型对测试集label进行预测
                scaled_preds, scaled_labels = self.predict(self.lstm, test_inputs, test_labels, is_test=True)

                test_scaled_preds_list.append(scaled_preds)
                test_scaled_labels_list.append(scaled_labels)

                scaled_GT = []
                for label_tensor in test_labels:
                    batch_array = label_tensor.cpu().numpy()
                    batch_array = batch_array.reshape(-1,
                                                      batch_array.shape[-1])  # 转化成[total time length, input_size]的形状
                    scaled_GT.append(batch_array)
                scaled_GT = np.concatenate(scaled_GT, axis=0)
                # GT = data_preparer.inverse_transform(scaled_GT)

                test_preds, test_GT = self.data_preparer.scaler.inverse_transform(
                    scaled_preds), self.data_preparer.scaler.inverse_transform(scaled_GT)

                # 计算均方误差和r2值
                scaled_mse = MSE(scaled_preds, scaled_labels)
                rmse = RMSE(test_preds, test_GT)
                # mape = MAPE(scaled_preds, scaled_labels)
                r2 = r2_score(scaled_preds, scaled_labels)
                r2_adjusted = 1 - (1 - r2) * (len(scaled_preds) - 1) / (len(scaled_preds) - len(self.variable_list) - 1)
                # print(f'Mean Square Error: {round(mse, 3)}; Mean Absolute Percentage Error: {round(mape, 3)}')

                if self.isPlot:
                    testing_plot(self.variable_list, self.time_freq, scaled_GT, scaled_preds, scaled_mse, rmse,
                                 r2_adjusted)
            # break

        return test_scaled_preds_list, test_scaled_labels_list

# 创建数据集（45秒数据预测未来15秒）
def create_sequences(input_data, decay_len, pred_len):
    sequences = []
    for i in range(len(input_data) - pred_len - decay_len):
        seq = input_data[i:i + decay_len]
        label = input_data[i + decay_len : i + decay_len + pred_len]
        sequences.append((seq, label))
    return sequences

def main():
    # 生成一些示例数据 (假设这是60秒的时间序列数据)
    data = np.sin(np.linspace(0, 60 * np.pi, 1000))  # 1000个数据点
    data = data.reshape(-1, 1)

    # 归一化数据
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)

    # 转换为PyTorch的tensor
    data_normalized = torch.FloatTensor(data_normalized).view(-1)

    # 将数据分割成训练集 (前85%) 和测试集 (后15%)
    train_size = int(len(data_normalized) * 0.85)
    train_data = data_normalized[:train_size]
    test_data = data_normalized[train_size:]

    # 定义超参数
    batch_size = 10
    hidden_size = 100  # 隐藏层的神经元数
    num_layers = 2  # LSTM层数
    decay_len = 45  # 输入45个时间步
    pred_len = 15  # 预测15个时间步
    input_size = 1  # 每个时间步的输入特征数
    output_size = 1  # 每个时间步的输出特征数
    train_sequences = create_sequences(train_data, decay_len, pred_len)
    test_sequences = create_sequences(test_data, decay_len, pred_len)

    # 转换为PyTorch的数据格式
    train_X = torch.stack([torch.FloatTensor(s[0]) for s in train_sequences]) # sample_num * decay_len
    train_y = torch.stack([s[1] for s in train_sequences])

    test_X = torch.stack([torch.FloatTensor(s[0]) for s in test_sequences]) # sample_num * pred_len
    test_y = torch.stack([s[1] for s in test_sequences])

    print(train_X.shape, train_y.shape)
    print(test_X.shape, test_y.shape)

    batch_train_X = train_X.reshape(-1, batch_size, decay_len, 1)
    batch_train_y = train_y.reshape(-1, batch_size, pred_len, 1)
    batch_test_X = test_X.reshape(-1, batch_size, decay_len, 1)
    batch_test_y = test_y.reshape(-1, batch_size, pred_len, 1)

    # 模型实例化
    model = InitLSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size, pred_len=pred_len, num_layers=num_layers)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # 进行前向传播
        for batch_X, batch_y in zip(batch_train_X, batch_train_y):
            output = model(batch_X)  # batch_size * seq_len * input_size
            loss = criterion(output, batch_y)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}')

    # 预测未来的15秒数据
    model.eval()
    # test_seq = torch.from_numpy(data[-decay_len-pred_len:-pred_len])  # 最后45个时间步作为输入
    prediction = model(test_X.unsqueeze(-1))
    print(prediction.size())
    # predictions = []
    # for _ in range(15):  # 预测未来15秒
    #     with torch.no_grad():
    #         prediction = model(test_seq)
    #         predictions.append(prediction.item())
    #         # 将预测值作为下一个输入
    #         new_seq = torch.cat((test_seq[:, 1:, :], prediction.unsqueeze(0).unsqueeze(-1)), 1)
    #         test_seq = new_seq
    # # 反归一化数据
    # predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    prediction = np.array([scaler.inverse_transform(pred) for pred in prediction.detach().numpy()])
    preds = [pred[0] for pred in prediction]
    preds.extend(prediction[-1][1:])
    latter_preds = prediction[0][:-1].tolist()
    latter_preds.extend([pred[-1] for pred in prediction])

    # 结果可视化
    plt.plot(np.arange(len(data)-len(preds), len(data)), data[-len(preds):], label='Original Data')
    plt.plot(np.arange(len(data)-len(preds), len(data)), preds, label='Prediction (Next Step)')
    plt.plot(np.arange(len(data)-len(latter_preds), len(data)), latter_preds, label='Prediction (Last Step)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()