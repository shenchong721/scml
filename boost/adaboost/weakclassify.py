# coding: UTF-8
from __future__ import division
import numpy as np


class WEAKC(object):
    """
    弱分类器的实现
    """
    def __init__(self, x, y):
        """
			X is a N*M matrix
			y is a length M vector
			M is the number of traincase
			this weak classifier is a decision Stump
			it's just a basic example from <统计学习方法>
		"""
        self.X = np.array(x)
        self.y = np.array(y)
        self.N = self.X.shape[0]

    def train(self, W, steps=100):
        """
        训练弱分类器
        :param W: is a N length vector
        :param steps: 训练多少次
        :return:返回训练误差
        """
        dmin = 100000000000.0   # 训练误差
        t_val = 0
        t_point = 0
        t_b = 0
        self.W = np.array(W)
        for i in range(self.N):
            q, err = self.findmin(i, 1, steps)
            if (err < dmin):
                dmin = err
                t_val = q
                t_point = i
                t_b = 1
        for i in range(self.N):
            q, err = self.findmin(i, -1, steps)
            if (err < dmin):
                dmin = err
                t_val = q
                t_point = i
                t_b = -1
        self.t_val = t_val
        self.t_point = t_point
        self.t_b = t_b
        print self.t_val, self.t_point, self.t_b
        return dmin

    def findmin(self, i, b, steps):
        """
        找到使得误差最小的决策树
        :param i:第i个样本
        :param b:b=-1表示样本划分到小于阈值的那一类，b=1表示样本划分到大于阈值的那一类
        :param steps:训练的次数
        :return:决策树的阈值，在该阈值下决策树的误差
        """
        t = 0   # 阈值最开始从0开始
        now = self.predintrain(self.X, i, t, b).transpose()
        err = np.sum((now != self.y) * self.W)
        # print now
        pgd = 0
        buttom = np.min(self.X[i, :])    # 第i个样本中特征值最小的值
        up = np.max(self.X[i, :])        # 第i个样本中特征值最大的值
        mins = 1000000;
        minst = 0
        st = (up - buttom) / steps     # 每个steps 增加的间隔
        for t in np.arange(buttom, up, st):    # 分steps次计算，每个阈值间隔为st
            now = self.predintrain(self.X, i, t, b).transpose()
            # print now.shape,self.W.shape,(now!=self.y).shape,self.y.shape
            err = np.sum((now != self.y) * self.W)
            if err < mins:
                mins = err
                minst = t
        return minst, mins

    def predintrain(self, test_set, i, t, b):
        """
        前向传播，计算数据集的分类结果
        :param test_set:数据集
        :param i:第i个样本
        :param t:决策树的阈值，小于阈值为假，大于阈值为真
        :param b:b=-1表示样本划分到小于阈值的那一类，b=1表示样本划分到大于阈值的那一类
        :return: 数据集的分类结果
        """
        test_set = np.array(test_set).reshape(self.N, -1)  # (N,M)
        gt = np.ones((np.array(test_set).shape[1], 1))  # (M,1)
        # print np.array(test_set[i,:]*b)<t*b
        gt[test_set[i, :] * b < t * b] = -1    # 对于每个维度的特征，当特征值小于阈值时为-1，大于或等于阈值为1
        return gt

    def pred(self, test_set):
        """
        测试模块
        :param test_set:
        :return:
        """
        test_set = np.array(test_set).reshape(self.N, -1)    # （N,M）
        t = np.ones((np.array(test_set).shape[1], 1))       # (M,1)
        t[test_set[self.t_point, :] * self.t_b < self.t_val * self.t_b] = -1
        return t
