#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2017/6/2 14:02

@Author: LiuBangxin
'''

import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = []; labelMat = []
    f = open('testSet.txt')
    for line in f.readlines():
        splitLine = line.strip().split()
        dataMat.append([1.0, float(splitLine[0]), float(splitLine[1])])
        labelMat.append(int(splitLine[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0 / (1+np.exp(-inX))

def gradAscent(dataMatIn, classLabel):
    dataMatrix = np.array(dataMatIn)    # m*n
    labelMat = np.array(classLabel).reshape(-1, 1)
    m, n = dataMatrix.shape
    alpha = 0.001
    maxCycles = 500
    weights = np.zeros([n, 1])
    for i in range(maxCycles):
        h = sigmoid(dataMatrix.dot(weights))
        error = labelMat - h
        weights += alpha*dataMatrix.transpose().dot(error)  # 感知器法则
    return weights

def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    m = len(dataMat)
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(m):
        if labelMat[i] == 1:
            xcord1.append(dataMat[i][1]); ycord1.append(dataMat[i][2])
        else:
            xcord2.append(dataMat[i][1]); ycord2.append(dataMat[i][2])
    import matplotlib.pyplot as plt
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green', marker='o')
    max_x = max(xcord1 + xcord2)
    min_x = min(xcord1 + xcord2)
    x = np.arange(min_x, max_x, 0.01)   # range的step只能为整数
    y = (-weights[0]-weights[1]*x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid()
    plt.show()

def stocGradAscent(dataMatIn, classLabel):
    dataSet = np.array(dataMatIn) # A m*n
    labelSet = np.array(classLabel) #A m
    m, n = dataSet.shape
    alpha = 0.01
    weights = np.zeros([n, 1])  # A n*1
    weightsList = []
    for j in range(100):
        for i in range(m):
            xi = dataSet[i].reshape(1, -1)
            hi = sigmoid(xi.dot(weights))
            error = np.sum(labelSet[i] - hi)
            weights += alpha*error*xi.T
            weightsList.append(weights.ravel().tolist())
    return weights, weightsList

def plotWeoghts(weights):
    weights = np.array(weights)
    fig = plt.figure('w')
    n = len(weights)
    ax = fig.add_subplot(311)
    ax.plot(range(n), weights[:, 0])
    plt.ylabel('x0')
    plt.grid()
    ax = fig.add_subplot(312)
    ax.plot(range(n), weights[:, 1])
    plt.ylabel('x1')
    plt.grid()
    ax = fig.add_subplot(313)
    ax.plot(range(n), weights[:, 2])
    plt.ylabel('x2')
    plt.grid()
    plt.show()

def stocGradAscent1(dataSet, labelSet, numIter=200):
    "改进的随机梯度下降,alpha每次迭代后减小，每次迭代随机选择一个样本点"
    dataSet = np.array(dataSet)
    labelSet = np.array(labelSet)
    m, n = dataSet.shape
    weights = np.zeros(n)
    weightsList = []
    for i in range(numIter):
        indexSeq = range(m)
        for j in range(m):
            alpha = 0.01 + 4 / (1.0+i+j)
            chooseIndex = np.random.randint(0, len(indexSeq))
            index = indexSeq[chooseIndex]
            xi = dataSet[index]
            hi = sigmoid(np.sum(weights*xi))
            error = labelSet[index] - hi
            weights += alpha*error*xi
            weightsList.append(weights.ravel().tolist())
            del indexSeq[chooseIndex]
    return weights, weightsList

def classifyVec(inX, weights):
    y_ = sigmoid(np.sum(inX*weights))
    if y_ > 0.5:
        return 1
    else:
        return 0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainSet = []; trainLabel = []
    for line in frTrain.readlines():
        currLine = line.strip().split()
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainSet.append(lineArr)
        trainLabel.append(float(currLine[-1]))
    weights, _ = stocGradAscent1(trainSet, trainLabel)
    count = 0.0
    numTest = 0.0
    for line in frTest.readlines():
        numTest += 1
        currLine = line.strip().split()
        inX = []
        for i in range(len(currLine)-1):
            inX.append(float(currLine[i]))
        y = float(currLine[-1])

        if y != classifyVec(np.array(inX), weights):
            count += 1
    errorRate = count / numTest
    print 'the error rate of this test is: %f' % errorRate
    return errorRate

def multiTest(numTest=10):
    averageErrorRate = 0.0
    for i in range(numTest):
        averageErrorRate += colicTest()
    print 'after %d iterations the average error rate is: %f' % (numTest, averageErrorRate / numTest)
    return averageErrorRate

if __name__ == '__main__':
    # dataMat, labelMat = loadDataSet()
    # weights, weightsList = stocGradAscent1(dataMat, labelMat, numIter=20)
    # # print weights
    # plotBestFit(weights)
    # plotWeoghts(weightsList)
    # colicTest()
    multiTest()