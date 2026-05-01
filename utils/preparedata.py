'''数据准备器，将数据转化为训练集/验证集/测试集'''
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

device = 'cpu'
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = 'cuda'
# print(device)

class DataPreparer:
    def __init__(self, batch_size, decay_len, pred_len, data_list, n_feature, out_col_slice, train_end, val_end, test_end):
        '''
        读取的是[dataset num, total time length, inputs_size...]的数据
        train_end, val_end 决定哪些数据集用来训练和验证
        '''
        self.batch_size = batch_size
        self.decay_len = decay_len
        self.pred_len = pred_len
        self.data = data_list
        self.n_feature = n_feature
        self.out_slice = out_col_slice
        self.train_end = train_end
        self.val_end = val_end
        self.test_end = test_end

        self._prepare_data()

    # 读取数据
    def _prepare_data(self):
        '''
        扩展数据以缩短模型的训练时间至关重要；
        将缩放器安装在训练集上只是为了避免验证和测试集中的数据泄漏
        '''
        n = len(self.data)
        train_data = self.data[:self.train_end]
        # print(self.train_end, self.val_end)
        val_data = self.data[self.train_end: self.val_end]
        # print(val_data[0].shape)
        test_data = self.data[self.val_end: n]

        train_data_concat = [[] for _ in range(self.n_feature)]
        for data in train_data:
            # print(data.shape)
            for idx in range(self.n_feature):
                train_data_concat[idx].extend(data[:, idx])
        train_data_concat = np.array(train_data_concat).T
        # print(train_data_concat.shape)

        # 用全体训练集进行归一化
        self.scaler = StandardScaler()
        self.scaler.fit(train_data_concat)

        self.train_data = [self.scaler.transform(data) for data in train_data]
        self.val_data = [self.scaler.transform(data) for data in val_data]
        self.test_data = [self.scaler.transform(data) for data in test_data]

    # 将数据窗口分割为输入和标签
    def _split_window(self, data):
        inputs = data[:, : self.decay_len, :]
        labels = data[:, self.decay_len:, self.out_slice]

        # inputs.set_shape([None, self.decay_len, None])
        # labels.set_shape([None, self.pred_len, None])
        return inputs, labels

    # 将数据重组为[num of samples, seq_len, input_size]的形式
    def _make_dataset(self, dataset, shuffle=True):
        # 输入的data是 dataset num * [total time length, input_size]的形式
        all_data = []
        for data in dataset:
            sample_size = data.shape[0] // (self.decay_len + self.pred_len)
            data = data[: sample_size * (self.decay_len + self.pred_len), :]  # 先在每个子集中截取样本
            data = data.reshape(sample_size, self.decay_len + self.pred_len, self.n_feature)  # 再将所有样本重组，乱序训练
            all_data.append(data)
        # 整理为[total sample num, window time length, input_size]的形式
        data = np.concatenate(all_data, axis=0)
        # print(data.shape)
        # 分割输入和标签
        inputs, labels = self._split_window(data)
        print(inputs.shape, labels.shape)
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)
        labels_tensor = torch.tensor(labels, dtype=torch.float32).to(device)
        # 创建一个 DataLoader
        inputs_loader = DataLoader(inputs_tensor, batch_size=self.batch_size, shuffle=shuffle,
                                   generator=torch.Generator(device=device))
        labels_loader = DataLoader(labels_tensor, batch_size=self.batch_size, shuffle=shuffle,
                                   generator=torch.Generator(device=device))

        return inputs_loader, labels_loader

    def _make_serial_dataset(self, dataset, shuffle=False):
        # 输入的data是dataset num * [total time length, input_size]的形式
        inputs_loader_list, labels_loader_list = [], []
        for iData, data in enumerate(dataset):
            sample_size = data.shape[0] // (self.decay_len + self.pred_len)
            data = data[: sample_size * (self.decay_len + self.pred_len), :]  # 在子集中截取样本
            data = data.reshape(sample_size, self.decay_len + self.pred_len,
                                self.n_feature)  # 重组子集样本为[sample num, window time length, input_size]的形式
            # 分割输入和标签
            inputs, labels = self._split_window(data)
            inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)
            labels_tensor = torch.tensor(labels, dtype=torch.float32).to(device)
            # 创建一个 DataLoader
            inputs_loader = DataLoader(inputs_tensor, batch_size=self.batch_size, shuffle=shuffle,
                                       generator=torch.Generator(device=device))
            labels_loader = DataLoader(labels_tensor, batch_size=self.batch_size, shuffle=shuffle,
                                       generator=torch.Generator(device=device))
            # 将单个loader加入列表
            inputs_loader_list.append(inputs_loader)
            labels_loader_list.append(labels_loader)

        return inputs_loader_list, labels_loader_list

    # 通过对预测进行逆变换，生成训练集、验证集和测试集
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def get_train(self, shuffle=True):
        # 在训练集中打乱顺序进行训练
        return self._make_dataset(self.train_data, shuffle=shuffle)

    def get_val(self):
        return self._make_serial_dataset(self.val_data, shuffle=False)

    def get_test(self):
        return self._make_serial_dataset(self.test_data, shuffle=False)



from copy import deepcopy
import os
import utils.wshrRelabelLight as WRL

class NASADataPreparer:
    def __init__(self, download_save_path, exception_save_path, variable_list):
        self.download_save_path = download_save_path
        self.exception_save_path = exception_save_path
        self.variable_list = variable_list

        # 获取文件夹下的所有文件名称
        self.download_folder_names = [item for item in os.listdir(download_save_path) if
                                 os.path.isdir(os.path.join(download_save_path, item))]
        self.exception_folder_names = [item for item in os.listdir(exception_save_path) if
                                  os.path.isdir(os.path.join(exception_save_path, item))]
        # 生成所有文件夹路径
        self.download_folder_paths = [os.path.join(download_save_path, item) for item in self.download_folder_names]
        self.exception_folder_paths = [os.path.join(exception_save_path, item) for item in self.exception_folder_names]

    def gammaCalc(self, data):
        # 计算飞行轨迹角
        gamma = data[:, 6] - np.mean(data[:, 4:6])
        # 重组数据
        data = np.hstack((data, gamma.reshape(-1, 1)))
        return data

    def ALTendsDel(self, data, win_len=60, sigma_num=3):
        ALT_idx = np.where(np.array(self.variable_list) == 'ALT')[0][0]
        ini_ALT, end_ALT = data[ALT_idx][:win_len], data[ALT_idx][-win_len:]
        ini_UCL = np.mean(ini_ALT) + sigma_num * np.std(ini_ALT) + 1
        end_UCL = np.mean(end_ALT) + sigma_num * np.std(end_ALT) + 1
        f_half_data, b_half_data = data[:int(len(data) / 2)], data[int(len(data) / 2):]
        data = np.concatenate((f_half_data[f_half_data[:, ALT_idx] > ini_UCL], b_half_data[b_half_data[:, ALT_idx] > end_UCL]),
                              axis=0)
        return data

    def MLEsplit(self, data, sigma_num=3):
        accept = True

        IVV_idx = np.where(np.array(self.variable_list) == 'IVV')[0][0]

        IVV_origin_data = data[:, IVV_idx]
        IVV_data = deepcopy(IVV_origin_data)

        # 将数据从中间分开
        IVV_takeoff_data = IVV_data[:len(IVV_data) // 2]
        IVV_landing_data = IVV_data[len(IVV_data) // 2:]

        # 将各部分的绝对值进行标准化，使得所有值加合为1（转化成对每个时刻的概率统计）
        abs_sum_IVV_takeoff = sum(abs(IVV_takeoff_data))
        abs_sum_IVV_landing = sum(abs(IVV_landing_data))

        takeoff_data, landing_data = [], []
        if abs_sum_IVV_landing == 0 or abs_sum_IVV_landing == 0:
            accept = False
        else:
            IVV_takeoff_data_dens = abs(IVV_takeoff_data) / abs_sum_IVV_takeoff
            IVV_landing_data_dens = abs(IVV_landing_data) / abs_sum_IVV_landing

            def GaussianMLE(dens):
                mu_est = sum([i * dens[i] for i in range(len(dens))])
                sigma_est = np.sqrt(sum([(i - mu_est) ** 2 * dens[i] for i in range(len(dens))]))
                return mu_est, sigma_est

            # 使用正态分布进行拟合，计算各部分的极大似然估计量
            mu1_est, sigma1_est = GaussianMLE(IVV_takeoff_data_dens)
            mu2_est, sigma2_est = GaussianMLE(IVV_landing_data_dens)
            mu2_est += len(IVV_takeoff_data)  # 降落阶段的时间点为起飞阶段的时间点加上总时间的一半

            # 获取控制线
            takeoff_control_limits = [int(max(0, mu1_est - sigma_num * sigma1_est)),
                                      int(min(len(IVV_data), mu1_est + sigma_num * sigma1_est))]
            landing_control_limits = [int(max(0, mu2_est - sigma_num * sigma2_est)),
                                      int(min(len(IVV_data), mu2_est + sigma_num * sigma2_est))]

            takeoff_data = data[takeoff_control_limits[0]: takeoff_control_limits[1], :]
            landing_data = data[landing_control_limits[0]: landing_control_limits[1], :]

        return takeoff_data, landing_data, accept

    def reconstructData(self, folder_name, fold_idx_list, mid_del=False, test=False, exception=False):
        folder_path = os.path.join(self.download_save_path, folder_name)
        exception_folder_path = os.path.join(self.exception_save_path, folder_name)
        # print(folder_path, exception_folder_path)
        # 只在正常数据集上进行训练
        if not exception:
            mat_name_list = [name for name in os.listdir(folder_path) if name not in os.listdir(exception_folder_path)]
        else:
            mat_name_list = [name for name in os.listdir(exception_folder_path)]

        # # 将最长的航程（大概率最完整）作为完整训练资料录入
        # max_len_mat_idx = -1
        # if train:
        #     max_len_mat_idx = np.argmax([len(loadmat(os.path.join(folder_path, mat_name))) for mat_name in mat_name_list[start: end]])
        data_array = []
        for fold_idx in fold_idx_list:
            start, end = self.fold_se_idx_list[fold_idx]
            if test:
                self.test_mat_name_list = mat_name_list[start: end]
            for mat_idx, mat_name in enumerate(mat_name_list[start: end]):
                # if mat_idx == len(mat_name_list[start: end]) - 1 and train:
                #     print(mat_name_list[mat_idx])
                #     mid_del = False
                data, wshr_label = WRL.dataConstruct(folder_path, mat_name, self.variable_list, normalized=False)

                # # 添加gamma变量
                # data = self.gammaCalc(data)

                # 基于ALT删除航程开头和结尾数据
                data = self.ALTendsDel(data)
                if mid_del:
                    # 使用极大似然截取起飞和降落航程数据
                    takeoff_data, landing_data, accept = self.MLEsplit(data)
                    if accept:
                        data_array.append(takeoff_data)
                        data_array.append(landing_data)
                    # print(takeoff_data.shape)
                    # print(landing_data.shape)
                else:
                    if data.shape[0] >= 60: # 删除记录不到1分钟的无效数据
                        data_array.append(data)
                    # print(data.shape)
        # data_array = np.array(data_array)
        return data_array

    def prepareData(self, work_folder_name, used_mat_num=100, fold_num=10, test_fold_idx=9):
        # 读取数据
        # # train_folder_path = exception_folder_paths[3]
        # # train_mat_name = os.listdir(train_folder_path)[2]
        # # test_folder_path = exception_folder_paths[3]
        # # test_mat_name = os.listdir(train_folder_path)[8]
        # train_folder_name = self.download_folder_names[3]
        # val_folder_name = self.download_folder_names[3]
        # test_folder_name = self.download_folder_names[3]
        train_folder_name = work_folder_name
        val_folder_name = work_folder_name
        test_folder_name = work_folder_name

        used_mat_num = np.min([used_mat_num, len(os.listdir(os.path.join(self.download_save_path, work_folder_name)))])

        self.fold_se_idx_list = []
        for i in range(fold_num):
            self.fold_se_idx_list.append([int(used_mat_num / fold_num) * i, int(used_mat_num / fold_num) * (i + 1)])

        # train_end = int(used_mat_num * 0.7)
        # val_end = int(used_mat_num * (0.7 + 0.2))
        # test_end = used_mat_num
        valid_fold_idx_list = [] # 使用测试集前序两折数据集作为验证集
        if test_fold_idx >= 2:
            valid_fold_idx_list = [test_fold_idx-1, test_fold_idx-2]
        elif test_fold_idx == 1:
            valid_fold_idx_list = [0, 9]
        else:
            valid_fold_idx_list = [9, 8]
        train_fold_idx_list = [i for i in range(10) if i not in valid_fold_idx_list and i != test_fold_idx]

        self.train_data = self.reconstructData(folder_name=train_folder_name, fold_idx_list=train_fold_idx_list, mid_del=True)
        self.val_data = self.reconstructData(folder_name=val_folder_name, fold_idx_list=valid_fold_idx_list, mid_del=False)
        self.test_data = self.reconstructData(folder_name=test_folder_name, fold_idx_list=[test_fold_idx], mid_del=False, test=True)

        # # 在异常数据集上批量测试
        # # exception_folder_name = self.exception_folder_names[3]
        # exception_folder_name = 'Tail_625_4'
        # exception_data = self.reconstructData(folder_name=exception_folder_name, start=0,
        #                                   end=len(os.listdir(self.exception_folder_paths[3])), mid_del=False, exception=True)

        # # 重构变量列表
        # self.variable_list.append('GAMMA')
        # print(self.variable_list)

        # from sklearn.decomposition import PCA
        # # 基于PCA进行降维去噪
        # # 初始化标准化模块
        # scaler = StandardScaler()

        # # 初始化PCA模块，保留95%方差解释比例的主成分
        # n_components = 5
        # pca = PCA(n_components)
        # pc_list = [f'PC {i}' for i in range(n_components)]

        # for train_idx in range(len(train_data)):
        #     selected_data = train_data[train_idx]
        #     selected_data_scaled = scaler.fit_transform(selected_data)
        #     train_data[train_idx] = pca.fit_transform(selected_data_scaled)
        # for idx in range(len(val_data)):
        #     selected_data = val_data[idx]
        #     selected_data_scaled = scaler.transform(selected_data)
        #     val_data[idx] = pca.transform(selected_data_scaled)
        # for idx in range(len(test_data)):
        #     selected_data = test_data[idx]
        #     selected_data_scaled = scaler.transform(selected_data)
        #     test_data[idx] = pca.transform(selected_data_scaled)

        # 合并数据
        origin_data = []
        origin_data.extend(self.train_data)
        origin_data.extend(self.val_data)
        origin_data.extend(self.test_data)
        # print(len(origin_data), len(train_data), len(val_data), len(test_data))
        # for i in range(len(origin_data)):
        #     print(origin_data[i].shape)

        return origin_data