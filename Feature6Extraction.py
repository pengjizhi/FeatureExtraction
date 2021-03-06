#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        Feature6Extraction
#-------------------------------------------------------------------------------

import math
import numpy as np
from skimage import io, filters

def Feature6Extraction():
    bcnt = 8*16
    input3 = []
    with open('Char_Index.txt', 'r') as fp:
        for line in fp:
            if line[0] != '#':
                linearr = line.strip().split('\t')
                input3.append(linearr[2])
    with open('feature6.txt','w+') as fp: #打开feature6.txt，用以储存特征1的数据
        for k in range(1000):
            A = io.imread('Char_Image_Binary/' + input3[k])
            t = filters.threshold_otsu(A) #取阈值
            B = (A > t) * 1 #二值化，B为二值化后的图像矩阵，每个元素的值为0或1
            a = B.shape[0]
            b = B.shape[1]
            C = np.zeros(bcnt, dtype=np.int) #定义特征向量
            bh = math.ceil(a/6.0)
            bw = math.ceil(b/6.0)

            for i in range(a):
                ib = math.floor(i/6)
                for j in range(b): #分区区域大小为6*6
                    if B[i,j] == 1:
                        jb = math.floor(j/6)
                        l = ib * bw + jb
                        C[l] += 1 #如果为白点，区域密度加1，最终结果为区域密度

            fp.write(str(k+1) + '\t')
            for i in range(bcnt-1):
                fp.write(str(C[i]) + ',')
            if k < 999:
                fp.write(str(C[bcnt-1]) + '\n')
            else:
                fp.write(str(C[bcnt-1]))
    pass

if __name__ == '__main__':
    Feature6Extraction()
