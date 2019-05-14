#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        test
#-------------------------------------------------------------------------------

import sys
path = 'C:\\Users\\Administrator\\Desktop\\FeatureExtraction\\FeatureExtraction\\libsvm-3.22\\python'
sys.path.append(path)
from svmutil import *
import numpy as np
from skimage import io
import matplotlib.pyplot as plt


def test():
    feature1 = []
    with open('feature1.txt', 'r') as fp:
        for line in fp:
            linearr = line.strip().split('\t')
            row = []
            row.append(int(linearr[0]))
            linearr2 = linearr[1].split(',')
            for i in range(len(linearr2)):
                row.append(int(linearr2[i]))
            feature1.append(row)

    feature1 = np.array(feature1)
    feature1 = feature1[:,1:]

    feature2 = []
    with open('feature2.txt', 'r') as fp:
        for line in fp:
            linearr = line.strip().split('\t')
            row = []
            row.append(int(linearr[0]))
            linearr2 = linearr[1].split(',')
            for i in range(len(linearr2)):
                row.append(int(linearr2[i]))
            feature2.append(row)

    feature2 = np.array(feature2)
    feature2 = feature2[:,1:]

    feature3 = []
    with open('feature3.txt', 'r') as fp:
        for line in fp:
            linearr = line.strip().split('\t')
            row = []
            row.append(int(linearr[0]))
            linearr2 = linearr[1].split(',')
            for i in range(len(linearr2)):
                row.append(int(linearr2[i]))
            feature3.append(row)

    feature3 = np.array(feature3)
    feature3 = feature3[:,1:]

    feature4 = []
    with open('feature4.txt', 'r') as fp:
        for line in fp:
            linearr = line.strip().split('\t')
            row = []
            row.append(int(linearr[0]))
            linearr2 = linearr[1].split(',')
            for i in range(len(linearr2)):
                row.append(int(linearr2[i]))
            feature4.append(row)

    feature4 = np.array(feature4)
    feature4 = feature4[:,1:]

    feature5 = []
    with open('feature5.txt', 'r') as fp:
        for line in fp:
            linearr = line.strip().split('\t')
            row = []
            row.append(int(linearr[0]))
            linearr2 = linearr[1].split(',')
            for i in range(len(linearr2)):
                row.append(int(linearr2[i]))
            feature5.append(row)

    feature5 = np.array(feature5)
    feature5 = feature5[:,1:]

    feature6 = []
    with open('feature6.txt', 'r') as fp:
        for line in fp:
            linearr = line.strip().split('\t')
            row = []
            row.append(int(linearr[0]))
            linearr2 = linearr[1].split(',')
            for i in range(len(linearr2)):
                row.append(int(linearr2[i]))
            feature6.append(row)

    feature6 = np.array(feature6)
    feature6 = feature6[:,1:]

    #feature = np.c_[3*feature1, 4*feature2, 2*feature3, feature4, 4*feature5, 100*feature6]
    feature = feature1
    feat = feature.tolist()

    classification_num = 13 #26
    allclass = [10, 11, 12, 20, 22, 25, 26, 28, 30, 31, 32, 33, 34]# 110 111 112 120 122 125 126 128 130 131 132 133 134];
    indexInfo = ['京', '渝', '鄂', '0',  '2', '5', '6', '8', 'A', 'B', 'C', 'D', 'Q']# '京' '渝' '鄂' '0'  '2' '5' '6' '8' 'A' 'B' 'C' 'D' 'Q'];

    train_num = 800
    selection_index = [];
    with open('selection_index.txt', 'r') as fp:
        line = fp.readline()
        linearr = line.strip().split(' ')
        for i in range(len(linearr)):
            selection_index.append(int(linearr[i]))

    rno = 0
    clast = [] #训练样本集的标签
    clasv = [] #测试样本集的标签
    featt = [] #训练样本集的特征
    featv = [] #测试样本集的特征
    namet = [] #训练样本集的名称
    namev = [] #测试样本集的名称
    with open('Char_Index.txt', 'r') as fp:
        for line in fp:
            if line[0] != '#':
                linearr = line.strip().split('\t')
                if selection_index[rno] == 1:
                    clast.append(int(linearr[1]))
                    namet.append(linearr[2])
                    featt.append(feat[rno])
                else:
                    clasv.append(int(linearr[1]))
                    namev.append(linearr[2])
                    featv.append(feat[rno])
                rno += 1

    model = svm_train(clast, featt, '-t 1 -d 1 -g 0.1 -r 0');
    pred_labels, (ACC, MSE, SCC), pred_values = svm_predict(clasv, featv, model)

    print('ACC = ' + str(ACC))
    err_idx = []
    for i in range(len(clasv)):
        if clasv[i] != pred_labels[i]:
            err_idx.append(i)

    for i in range(len(err_idx)):
        e = allclass.index(clasv[err_idx[i]])
        f = allclass.index(pred_labels[err_idx[i]])
        print(indexInfo[e] + '被误识为' + indexInfo[f])

        imgpath = 'Char_Image/' + namev[err_idx[i]]
        img = io.imread(imgpath)
        plt.figure(i)
        plt.imshow(img)
    plt.show()
    pass

if __name__ == '__main__':
    test()
