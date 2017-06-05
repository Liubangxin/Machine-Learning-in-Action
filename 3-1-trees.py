#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/5/31 19:15
# @Author  : LiuBangxin

import math
# 最后一列为label
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    # 统计每个label出现的频数
    for featureVec in dataSet:
        label = featureVec[-1]
        labelCounts[label] = labelCounts.get(label, 0) + 1
    # 计算熵
    entropy = 0
    for v in labelCounts.values():
        prob = v / float(numEntries)
        entropy -= prob * math.log(prob, 2)
    return entropy

def createDateSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featureVec in dataSet:
        if featureVec[axis] == value:
            reducedFeature = featureVec[:axis]    # 去掉当前特征向量
            reducedFeature.extend(featureVec[axis+1:])
            retDataSet.append(reducedFeature)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numEntries = len(dataSet)
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestFeature = -1
    bestInfoGain = 0.0
    for i in range(numFeatures):
        feaList = [example[i] for example in dataSet]
        uniqueVals = set(feaList)
        newEntropy = 0.0
        for val in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, val)
            prob = len(subDataSet) / float(numEntries)
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature, bestInfoGain

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    sorted(classCount.iteritems(), key=lambda item: item[1], reverse=True)
    return classCount[0][0]

def createTree(dataSet, labels):
    labels = labels[:]
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat, _ = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del labels[bestFeat]
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        sublabels = labels[:]   # 将labels复制一份给sublabels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), sublabels)
    return myTree

def classify(inputTree, featLabels, testVec):
    firstFeat = inputTree.keys()[0]
    secondDict = inputTree[firstFeat]
    firstFeatIndex = featLabels.index(firstFeat)
    for key in secondDict.keys():
        if testVec[firstFeatIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                return classify(secondDict[key], featLabels, testVec)
            else:
                return secondDict[key]

def storeTree(inputTree, fileName):
    import pickle
    fw = open(fileName, 'w')
    pickle.dump(inputTree, fw)
    fw.close()

def loadTree(fileName):
    import pickle
    fr = open(fileName, 'r')
    return pickle.load(fr)

if __name__ == '__main__':
    myDat, labels = createDateSet()
    # print myDat
    # print labels
    # entropy = calcShannonEnt(myDat)
    # print entropy
    # print splitDataSet(myDat, 0, 1)
    # print splitDataSet(myDat, 0, 0)
    # print chooseBestFeatureToSplit(myDat)
    # myTree = createTree(myDat, labels)
    # print myTree
    # storeTree(myTree, 'myTree.txt')
    myTree = loadTree('myTree.txt')
    print myTree
    print 'label: ', classify(myTree, labels, [1, 0])