import matplotlib.pyplot as plt
from math import ceil

def training_plot(train_loss, performance_history):
    # 创建 1 行 3 列的子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 第一个子图：训练损失
    axes[0].plot(train_loss, label='loss', color='blue')
    axes[0].set_title('Train Loss')
    axes[0].set_xlabel('Train step')
    axes[0].set_ylabel('Train loss')

    # 第二个子图：MSE
    for type_idx, data_type in enumerate(['train', 'validate', 'test']):
        axes[1].plot(performance_history[0, :, type_idx], label=data_type)
    axes[1].set_title('MSE')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('MSE')
    axes[1].legend()

    # 第三个子图：MAPE
    for type_idx, data_type in enumerate(['train', 'validate', 'test']):
        axes[2].plot(performance_history[1, :, type_idx], label=data_type)
    axes[2].set_title('MAPE')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('MAPE')
    axes[2].legend()

    # 设置整体标题和布局
    fig.suptitle('Transformer Performance')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局避免标题重叠

    # 显示图形
    plt.show()

def improve_training_plot(train_loss, performance_history):
    # 创建 1 行 3 列的子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 第一个子图：训练损失
    axes[0].plot(train_loss, label='loss', color='blue')
    axes[0].set_title('Train Loss')
    axes[0].set_xlabel('Train step')
    axes[0].set_ylabel('Train loss')

    # 第二个子图：MSE
    for type_idx, data_type in enumerate(['train', 'validate']):
        axes[1].plot(performance_history[0, :, type_idx], label=data_type)
    axes[1].set_title('MSE')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('MSE')
    axes[1].legend()

    # 第三个子图：MAPE
    for type_idx, data_type in enumerate(['train', 'validate']):
        axes[2].plot(performance_history[1, :, type_idx], label=data_type)
    axes[2].set_title('MAPE')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('MAPE')
    axes[2].legend()

    # 设置整体标题和布局
    fig.suptitle('Transformer Performance')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局避免标题重叠

    # 显示图形
    plt.show()

def testing_plot(variable_list, time_freq, scaled_GT, scaled_preds, scaled_mse, rmse, r2):
    # 创建 5 行 3 列的子图
    fig, axes = plt.subplots(ceil(len(variable_list)/3), 3, figsize=(15, 20))

    # 遍历每个变量并绘制图形
    for idx in range(len(variable_list)):
        row_idx, col_idx = idx // 3, idx % 3

        # 绘制 GT 曲线
        axes[row_idx, col_idx].plot([i / time_freq for i in range(scaled_GT.shape[0])], scaled_GT[:, idx], label='GT',
                                    color='blue')

        # 绘制预测曲线
        axes[row_idx, col_idx].plot([i / time_freq for i in range(scaled_preds.shape[0])], scaled_preds[:, idx],
                                    label='pred', color='red')

        # 设置 x 轴标题为变量名称
        axes[row_idx, col_idx].set_xlabel(variable_list[idx])

        # 只有第一列的子图设置 y 轴标签
        if col_idx == 0:
            axes[row_idx, col_idx].set_ylabel('Value')

        # 添加图例
        axes[row_idx, col_idx].legend()

    # 设置整体标题和布局
    fig.suptitle(f'Transformer Prediction vs. Ground Truth [(scaled_MSE, RMSE, r2) = ({scaled_mse:.3f}, {rmse:.3f}, {r2:.3f})]', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局避免标题重叠

    # 显示图形
    plt.show()

import numpy as np
import pandas as pd
import os
from sklearn.metrics import root_mean_squared_error as RMSE, r2_score
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
import vapeplot
import scicomap

class Plotter:
    def __init__(self, method_list, variable_list, result_data_savepath, fig_savepath):
        self.variable_list = variable_list
        self.result_data_savepath = result_data_savepath
        self.fig_savepath = fig_savepath
        self.method_list = method_list

    def r2Adjust(self, label_array, pred_array, num_samples):
        r2 = r2_score(label_array, pred_array)
        r2_adjusted = 1 - (1 - r2) * (num_samples - 1) / (num_samples - len(self.variable_list) - 1)
        if r2_adjusted < 0:
            r2_adjusted = 0
        return r2_adjusted

    def metricsVolumePlot(self, metrics_array, var_name, metrics_name='Adjusted R2'):
        # 以fold_idx为x轴，以test_idx为y轴，以r2为z轴，绘制散点图
        fig = plt.figure(figsize=(16, 8))

        gs = GridSpec(2, 6, figure=fig)  # 定义2行6列的网格

        # 预留左侧和右侧空间
        subplot_list = [[0, 0], [1, 0], [0, 1], [1, 1],
                        [0, 4], [1, 4], [0, 5], [1, 5]]

        for method_idx in range(metrics_array.shape[0]):
            # 构建以fold_idx和test_idx构成的网格
            x, y = np.meshgrid(range(1, metrics_array.shape[1] + 1), range(1, metrics_array.shape[2] + 1))
            method_name = self.method_list[method_idx]
            if method_name == 'MC-TwinTransformer':
                # 绘制中间独占两列空间的图
                ax = fig.add_subplot(gs[:, 2:4], projection='3d')
            else:
                ax = fig.add_subplot(gs[subplot_list[0][0], subplot_list[0][1]], projection='3d')
                # 删除使用过的空间
                subplot_list.pop(0)

            z = metrics_array[method_idx, :, :]
            # 计算下方比体积，每个底面1*1小正方形上方的几何体体积 = 底面面积 * 各顶面顶点高度平均值
            z_volume = np.mean([np.mean([z[i+m][j+n] for m in [0, 1] for n in [0, 1]])
                               for i in range(metrics_array.shape[1]-1) for j in range(metrics_array.shape[2]-1)])

            # 绘制散点图
            ax.plot_surface(x, y, z, color='lightgreen', edgecolor='k', alpha=0.6)

            # 设置轴标签和标题
            ax.set_xlabel('Fold Index')
            ax.set_ylabel('Test Index')
            ax.set_zlabel(f'{metrics_name} Value')
            # ax.set_zlim(0, 1)
            ax.set_title(f'{method_name}: {z_volume:.2f}')

        if var_name == 'Overall':
            plt.suptitle(f'{metrics_name} Value of Overall Forecast Performance by Different Methods')
        else:
            plt.suptitle(f'{metrics_name} Value of {var_name} Forecast Performance by Different Methods')

        # 显示图形
        plt.tight_layout()
        fig_savepath = os.path.join(self.fig_savepath, var_name)
        if not os.path.exists(fig_savepath):
            os.makedirs(fig_savepath)
        plt.savefig(os.path.join(fig_savepath, f'{metrics_name}_volume.svg'), dpi=300, format='svg')
        plt.show()

    def metricsBoxPlot(self, metrics_array, var_name, metrics_name='RMSE'):
        # 以fold_idx为x轴，以metrics值为y轴，把每个fold中的测试集表现绘制成箱线图
        fig = plt.figure(figsize=(10, 6))
        gs = GridSpec(1, 5, figure=fig)  # 定义1行5列的网格
        ax = fig.add_subplot(gs[:, :4]) # 使用前4列绘制主图

        # 预定义不同方法箱线图的外框特征
        # color_list = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        color_list = vapeplot.palette('vaporwave') # mallsoft
        color_list = [(91, 169, 189), (149, 190, 84), (219, 204, 87), (224, 187, 109),
                      (176, 120, 61), (213, 166, 160), (157, 96, 103), (220, 222, 139),
                      (169, 57, 53)]
        color_list = [(r / 255, g / 255, b / 255) for r, g, b in color_list] # 转换为浮点数
        # color_list = [color for idx, color in enumerate(color_list) if idx % 2 != 0] # 提取奇数元素以提升区分度
        boxprop_list = [dict(color=color_list[i], linewidth=2, linestyle=':') for i in range(len(self.method_list))]
        whiskerprop_list = [dict(color=color_list[i], linewidth=2, linestyle='-') for i in range(len(self.method_list))]  # 设置须的颜色
        meanprop_list = [dict(marker='o', markerfacecolor=color_list[i], markeredgecolor='black', markersize=8) for i in range(len(self.method_list))]  # 设置平均值的样式和颜色
        medianprop_list = [dict(color=None, linewidth=0) for i in range(len(self.method_list))]  # 设置中位数的颜色
        capprop_list = [dict(color=color_list[i], linewidth=2, linestyle='--') for i in range(len(self.method_list))]  # 设置须帽的颜色
        flierprop_list = [dict(marker='o', color=color_list[i], markersize=8) for i in range(len(self.method_list))]  # 设置离群点的样式和颜色

        # 预定义填充颜色的透明度
        opacity = 1

        # 使用method作为横坐标
        bplot = ax.boxplot(metrics_array.reshape(len(self.method_list), -1).T, widths=0.6,
                           medianprops=dict(color='black', linewidth=2, linestyle=':'),
                           showfliers=False, showmeans=False, showcaps=False, patch_artist=True
                           )
        for method_idx, patch in enumerate(bplot['boxes']):
            rgb_value = list(to_rgb(color_list[method_idx]))
            rgb_value.append(opacity)
            patch_color = tuple(rgb_value)
            patch.set_facecolor(patch_color) # 设置填充颜色
        ax.set_xticks(np.arange(len(self.method_list)) + 1)
        ax.set_xticklabels(self.method_list, rotation=45)
        ax.set_xlabel('Method Names')
        ax.set_ylabel(f'{metrics_name} Value')

        if var_name == 'Overall':
            plt.suptitle(f'{metrics_name} Value of Overall Forecast Performance by Different Methods')
        else:
            plt.suptitle(f'{metrics_name} Value of {var_name} Forecast Performance by Different Methods')

        # 手动创建图例
        legend_elements = [Line2D([0], [0], color=color, lw=2, label=method)
                           for color, method in zip(color_list, self.method_list)]

        ax = fig.add_subplot(gs[:, 4:]) # 使用最后一列绘制图例
        ax.axis('off') # 隐藏坐标轴
        ax.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        fig_savepath = os.path.join(self.fig_savepath, var_name)
        if not os.path.exists(fig_savepath):
            os.makedirs(fig_savepath)
        plt.savefig(os.path.join(fig_savepath, f'{metrics_name}_boxplot.svg'), dpi=300, format='svg')
        plt.show()

    def metricsCurvePlot(self, metrics_array, var_name, metrics_name='RMSE'):
        # 以fold_idx为x轴，以metrics值为y轴，把每个fold中的测试集表现绘制成箱线图
        fig = plt.figure(figsize=(10, 6))
        gs = GridSpec(1, 5, figure=fig)  # 定义1行5列的网格
        ax = fig.add_subplot(gs[:, :4]) # 使用前4列绘制主图

        # 预定义不同方法箱线图的外框特征
        # color_list = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        color_list = vapeplot.palette('vaporwave') # mallsoft
        color_list = [(91, 169, 189), (149, 190, 84), (219, 204, 87), (224, 187, 109),
                      (176, 120, 61), (213, 166, 160), (157, 96, 103), (220, 222, 139),
                      (169, 57, 53)]
        color_list = [(r / 255, g / 255, b / 255) for r, g, b in color_list] # 转换为浮点数
        marker_shape_list = ['o', 's', '^', 'D', 'v', 'p', 'P', '*', 'h', 'H']

        for method_idx, method_name in enumerate(self.method_list):
            ax.set_xticks(np.arange(metrics_array.shape[1]) + 1)
            ax.set_xticklabels([f'{fold_idx+1}' for fold_idx in range(metrics_array.shape[1])], rotation=0)
            ax.set_xlabel('Fold Index')
            ax.set_ylabel(f'{metrics_name} Value')
            # 绘制均值连接线
            plt.plot([fold_idx+1 for fold_idx in range(metrics_array.shape[1])], np.mean(metrics_array[method_idx, :, :], axis=1),
                     marker=marker_shape_list[method_idx], markeredgecolor='black', markersize=10, color=color_list[method_idx], linewidth=2)

        if var_name == 'Overall':
            plt.suptitle(f'{metrics_name} Value of Overall Forecast Performance by Different Methods')
        else:
            plt.suptitle(f'{metrics_name} Value of {var_name} Forecast Performance by Different Methods')

        # 手动创建图例
        legend_elements = []
        for method_idx, method_name in enumerate(self.method_list):
            color = color_list[method_idx]
            marker_shape = marker_shape_list[method_idx]
            legend_elements.append(Line2D([0], [0], marker=marker_shape, markeredgecolor='black', markersize=10,
                                      color=color, lw=2, label=method_name))

        ax = fig.add_subplot(gs[:, 4:]) # 使用最后一列绘制图例
        ax.axis('off') # 隐藏坐标轴
        ax.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        fig_savepath = os.path.join(self.fig_savepath, var_name)
        if not os.path.exists(fig_savepath):
            os.makedirs(fig_savepath)
        plt.savefig(os.path.join(fig_savepath, f'{metrics_name}_curve.svg'), dpi=300, format='svg')
        plt.show()

    def relativeImproveBoxplot(self, metrics_array, var_name, metrics_name='RMSE'):
        # 以基本method为x轴，以metrics值为y轴，把每个fold中的测试集表现绘制成箱线图
        fig = plt.figure(figsize=(5, 6))
        gs = GridSpec(1, 3, figure=fig)  # 定义1行5列的网格
        ax = fig.add_subplot(gs[:, :2])  # 使用前4列绘制主图

        # 预定义不同方法箱线图的外框特征
        # color_list = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        color_list = vapeplot.palette('vaporwave')  # mallsoft
        color_list = [(91, 169, 189), (149, 190, 84), (219, 204, 87), (224, 187, 109),
                      (176, 120, 61), (213, 166, 160), (157, 96, 103), (220, 222, 139),
                      (169, 57, 53)]
        color_list = [(r / 255, g / 255, b / 255) for r, g, b in color_list]  # 转换为浮点数
        # color_list = [color for idx, color in enumerate(color_list) if idx % 2 != 0] # 提取奇数元素以提升区分度

        # 预定义填充颜色的透明度
        opacity = 1

        # 使用method作为横坐标，排除前三个baseline后，剩下六个方法一一对应计算均值提升相对值
        relative_improve_array = []
        for method_idx in range(3, 6):
            relative_improvement = 100 * np.mean((metrics_array[method_idx+3] - metrics_array[method_idx]), axis=1) / np.mean(metrics_array[method_idx], axis=1) # fold_num * 1
            relative_improve_array.append(relative_improvement)
        relative_improve_array = np.array(relative_improve_array) # 3 * fold_num
        bplot = ax.boxplot(relative_improve_array.T, widths=0.6,
                           medianprops=dict(color='black', linewidth=2, linestyle=':'),
                           showfliers=False, showmeans=False, showcaps=False, patch_artist=True
                           )
        for method_idx, patch in enumerate(bplot['boxes']):
            rgb_value = list(to_rgb(color_list[method_idx+6]))
            rgb_value.append(opacity)
            patch_color = tuple(rgb_value)
            patch.set_facecolor(patch_color)  # 设置填充颜色
        # 为每个中位数显示数值
        for i, line in enumerate(bplot['medians']):
            mean_value = np.mean(relative_improve_array[i])
            ax.text(line.get_xdata()[0], mean_value, f'{mean_value:.2f}%',
                    verticalalignment='center', horizontalalignment='right')

        ax.set_xticks(np.arange(3) + 1)
        ax.set_xticklabels(self.method_list[6:], rotation=45)
        ax.set_xlabel('Mechanism Aiding Method Names')
        ax.set_ylabel(f'Relative Improved Ratio of {metrics_name} (%)')

        if var_name == 'Overall':
            plt.suptitle(f'Relative Improvement w.r.t {metrics_name} Value of Overall Forecast Performance with mechanism aiding')
        else:
            plt.suptitle(f'Relative Improvement w.r.t {metrics_name} Value of {var_name} Forecast Performance with mechanism aiding')

        # 手动创建图例
        legend_elements = [Line2D([0], [0], color=color, lw=2, label=method)
                           for color, method in zip(color_list[6:], self.method_list[6:])]

        ax = fig.add_subplot(gs[:, 2:])  # 使用最后一列绘制图例
        ax.axis('off')  # 隐藏坐标轴
        ax.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        fig_savepath = os.path.join(self.fig_savepath, var_name)
        if not os.path.exists(fig_savepath):
            os.makedirs(fig_savepath)
        plt.savefig(os.path.join(fig_savepath, f'{metrics_name}_improve_boxplot.svg'), dpi=300, format='svg')
        plt.show()

    def singleTestResultRead(self, var_name, fold_idx, test_mat_name, csv_savepath):
        pred_df = pd.read_csv(os.path.join(csv_savepath, [f for f in os.listdir(csv_savepath) if f.startswith(f'preds_{fold_idx}_{test_mat_name}') and f.endswith('.csv')][0]))
        label_df = pd.read_csv(os.path.join(csv_savepath, [f for f in os.listdir(csv_savepath) if f.startswith(f'labels_{fold_idx}_{test_mat_name}') and f.endswith('.csv')][0]))
        if var_name == 'Overall':
            pred_array = np.array(pred_df)
            label_array = np.array(label_df)
        else:
            var_idx = self.variable_list.index(var_name)
            pred_array = np.array(pred_df.iloc[:, var_idx])
            label_array = np.array(label_df.iloc[:, var_idx])
        return pred_array, label_array

    def run(self, var_name='Overall'):
        print(f'===== Plotting {var_name} =====')
        # 初始化r2和rmse记录表
        r2_list, rmse_list = [[[] for fold_idx in range(10)] for method_name in self.method_list], [[[] for fold_idx in range(10)] for method_name in self.method_list]

        # 初始化最大测试集长度
        max_test_len = 10

        for method_idx, method_name in enumerate(self.method_list): # 所有方法
            for fold_idx in range(10):  # 10折
                # 获得存储数据csv的路径
                csv_savepath = os.path.join(self.result_data_savepath, method_name, f'fold_{fold_idx}',
                                            'prediction_csv')
                # 所有测试文件名称
                test_mat_name_list = list(set([f.split('_')[-1] for f in os.listdir(csv_savepath)]))
                test_mat_name_list = sorted(test_mat_name_list) # 对测试文件名称进行排序，保证每次读取的文件顺序一致
                # 更新最大测试集长度
                max_test_len = max(max_test_len, len(test_mat_name_list))
                for test_mat_name in test_mat_name_list: # 所有测试文件
                    # 获得预测和标签数据
                    pred_array, label_array = self.singleTestResultRead(var_name, fold_idx, test_mat_name, csv_savepath)
                    # 计算r2和rmse
                    single_r2 = self.r2Adjust(label_array, pred_array, pred_array.shape[0])
                    # single_r2 = r2_score(label_array, pred_array)
                    single_rmse = RMSE(label_array, pred_array)
                    # 记录r2和rmse
                    r2_list[method_idx][fold_idx].append(single_r2)
                    rmse_list[method_idx][fold_idx].append(single_rmse)

        # 用零值填补欠缺的测试集 # method_num * fold_num * max_test_len = 5 * 10 * 10
        r2_array = np.array([[np.pad(subsublist, (0, max_test_len - len(subsublist)), 'constant') for subsublist in sublist] for sublist in r2_list])
        rmse_array = np.array([[np.pad(subsublist, (0, max_test_len - len(subsublist)), 'constant') for subsublist in sublist] for sublist in rmse_list])

        self.metricsCurvePlot(r2_array, var_name, metrics_name='Adjusted r2')
        self.relativeImproveBoxplot(r2_array, var_name, metrics_name='Adjusted r2')
        self.metricsBoxPlot(rmse_array, var_name, metrics_name='RMSE')
        self.relativeImproveBoxplot(rmse_array, var_name, metrics_name='RMSE')

if __name__ == '__main__':
    # 规定方法列表和变量列表
    method_list = ['LSTM', '1dCNN', 'ConvSparseTran', 'ZonalTransformer', 'TwinTransformer', 'ZT-Transformer', 'MA-ZonalTran', 'MA-TwinTran', 'MA-ZT-Tran'] # left: 2*2, right: 2*2, center: 1
    variable_list = ['TAS', 'GS', 'IVV', 'FPAC', 'CTAC',
                     'PTCH', 'AOAC', 'ROLL', 'DA', 'TH', 'TRK',
                     'WS', 'WD', 'SAT', 'TAT', 'PI', 'PT', 'ALT']

    # 规定工作数据集
    work_folder_name = 'Tail_652_4'

    # 规定训练出的模型以及预测结果csv的存储根目录
    result_data_savepath = "./Result/data"
    print(os.path.abspath(result_data_savepath))
    # 规定绘图存储路径
    fig_savepath = "./Result/figures"
    if not os.path.exists(fig_savepath):
        os.makedirs(fig_savepath)

    # 初始化绘图工具类
    result_plotter = Plotter(method_list, variable_list, result_data_savepath, fig_savepath)

    # 对整体状况打印r2和RMSE的箱线图
    result_plotter.run(var_name='Overall')

    # 对给定变量打印r2和RMSE的箱线图
    for var_name in variable_list:
        result_plotter.run(var_name=var_name)