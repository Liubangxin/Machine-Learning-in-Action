#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2017/5/31 20:09

@Author: LiuBangxin
'''

import numpy as np

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def createVocabList(dataSet):
    vocabList = set([])
    for doc in dataSet:
        vocabList = vocabList | set(doc)
    return list(vocabList)

def setWords2Vec(vocabList, doc):
    wordVec = [0]*len(vocabList)
    for word in doc:
        if word in vocabList:
            wordVec[vocabList.index(word)] = 1
        else:
            print 'The word: %s is not in my Vocabulary' % word
    return wordVec

def bagOfWords2VecMN(vocabList, doc):
    returnVec = [0]*len(vocabList)
    for word in doc:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print 'The word: %s is not in my Vocabulary' % word
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    numDocs = len(trainMatrix)
    numFeat = len(trainMatrix[0])
    p1Num = np.ones(numFeat, dtype=float)
    p0Num = np.ones(numFeat, dtype=float)
    pAbusive = sum(trainCategory) / float(numDocs)
    p1Denom = 2.0
    p0Denom = 2.0
    # p1Denom = numFeat
    # p0Denom = numFeat
    for i in range(numDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        if trainCategory[i] == 0:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vec = np.log(p1Num / p1Denom)
    p0Vec = np.log(p0Num / p0Denom)
    return p0Vec, p1Vec, pAbusive

def classifyNB(vec2Classify, p1Vec, p0Vec, pClass1):
    " vec2Classify, p1Vec, p0Vec为numpy数组 "
    p1 = np.sum(vec2Classify * p1Vec)
    p0 = np.sum(vec2Classify * p0Vec)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    dataSet, labels = loadDataSet()
    vocabList = createVocabList(dataSet)
    trainSet = []
    for doc in dataSet:
        trainSet.append(setWords2Vec(vocabList, doc))
    p0, p1, pClass1 = trainNB0(trainSet, labels)
    testEntry = ['love', 'my', 'dalmation']
    testVec = np.array(setWords2Vec(vocabList, testEntry))
    testClassify = classifyNB(testVec, p1, p0, pClass1)
    print testEntry, 'classified as: ', testClassify
    testEntry = ['stupid', 'garbage']
    testVec = np.array(setWords2Vec(vocabList, testEntry))
    testClassify = classifyNB(testVec, p1, p0, pClass1)
    print testEntry, 'classified as: ', testClassify

def textParse(bigString):
    import re
    return [tok.lower() for tok in re.split(r'\W*', bigString) if len(tok)>2]

def spamTest():
    import os
    hamDir = './email/ham/'
    spamDir = './email/spam/'
    hamFiles = os.listdir(hamDir)
    spamFiles = os.listdir(spamDir)
    labels = [0]*len(hamFiles) + [1]*len(spamFiles)
    dataSet = []
    for f in hamFiles:
        dataSet.append(textParse(open(hamDir+f, 'r').read()))
    for f in spamFiles:
        dataSet.append(textParse(open(spamDir+f, 'r').read()))
    vocabList = createVocabList(dataSet)

    # 创建训练集
    testIndex = []
    trainIndex = range(len(labels))
    for i in range(10):
        index = np.random.randint(0, len(trainIndex))
        testIndex.append(index)
        del trainIndex[index]
    trainMatrix = []
    trainCategory = []
    for index in trainIndex:
        trainMatrix.append(setWords2Vec(vocabList, dataSet[index]))
        trainCategory.append(labels[index])
    p0Vec, p1Vec, pClass1 = trainNB0(trainMatrix, trainCategory)
    errorCount = 0
    for i in testIndex:
        testVec = np.array(setWords2Vec(vocabList, dataSet[i]))
        classify = classifyNB(testVec, p1Vec, p0Vec, pClass1)
        if classify != labels[i]:
            errorCount += 1
    print 'the error rate is: ', float(errorCount) / len(testIndex)

if __name__ == '__main__':
    spamTest()
    # dataSet, labels = loadDataSet()
    # vocabList = createVocabList(dataSet)
    # # print vocabList
    # # vec = setWords2Vec(vocabList, doc=dataSet[0])
    # # print vec
    # trainSet = []
    # for doc in dataSet:
    #     trainSet.append(setWords2Vec(vocabList, doc))
    # p0, p1, pA = trainNB0(trainSet, labels)
    # print pA
    # print p0
    # print p1
    # testingNB()
