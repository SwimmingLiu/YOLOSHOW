import matplotlib.pyplot as plt
from PySide6.QtCore import QThread


class PlottingThread(QThread):
    def __init__(self, result_statistic, workpath):
        super().__init__()
        self.result_statistic = result_statistic
        self.workpath = workpath

    def run(self):
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是一个常见的中文黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

        # 计算总数
        total = sum(self.result_statistic.values())
        # 计算每个类别的占比
        percentages = {k: (v / total * 100) for k, v in self.result_statistic.items()}

        # 数据准备
        activities = list(percentages.keys())
        values = list(percentages.values())

        # 创建柱状图
        plt.figure(figsize=(10, 6))  # 设置图形的显示大小
        bars = plt.bar(activities, values, color='skyblue')  # 绘制柱状图

        # 在每个柱子上方添加百分比
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f'{yval:.2f}%', ha='center', va='bottom')

        # 添加标题和标签
        plt.title('Detection results target category statistical proportion')
        plt.xlabel('Target Category')
        plt.ylabel('Percentage (%)')

        # 保存图形到文件
        plt.savefig(self.workpath + r'\config\result.png')
        plt.close()  # 重要：关闭图形，释放内存
