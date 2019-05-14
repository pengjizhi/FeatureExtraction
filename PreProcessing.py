#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        PreProcessing
#-------------------------------------------------------------------------------

import numpy as np
from skimage import io, color, filters, morphology

def PreProcessing():
    input3 = []
    with open('Char_Index.txt', 'r') as fp:
        for line in fp:
            if line[0] != '#':
                linearr = line.strip().split('\t')
                input3.append(linearr[2])

    bl = 47
    for k in range(1000):
        A = io.imread('Char_Image/' + input3[k], as_gray = True)
        #io.imshow(A)
        #io.show()
        #A = color.rgb2gray(A)
        t = filters.threshold_otsu(A)
        B = (A > t) * 1.0
        a = B.shape[0]
        b = B.shape[1]
        '''
        B = np.zeros((a,b))        
        for i in range(0, a, bl):
            for j in range(0, b, bl):
                mid = A[i:min(i+bl, a), j:min(j+bl, b)]
                t = filters.threshold_otsu(mid)
                mid_t = (mid > t) * 1.0
                B[i:min(i+bl, a), j:min(j+bl, b)] = mid_t

        B = morphology.opening(B, morphology.disk(1))
        '''
        
        if B[0, 0] + B[0, b-1] + B[a-1, 0] + B[a-1, b-1] >= 2:
            for i in range(a):
                for j in range(b):
                    B[i, j] = 1 - B[i, j]
        
        io.imsave('Char_Image_Binary/' + input3[k], B)
    pass

if __name__ == '__main__':
    PreProcessing()
