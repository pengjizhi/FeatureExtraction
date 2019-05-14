#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        PreProcessingToCorrect
#-------------------------------------------------------------------------------

from skimage import io, filters

def PreProcessingToCorrect():
    input = []
    with open('err.txt', 'r') as fp:
        for line in fp:
            input.append(line.strip())
    for k in range(3):
        A = io.imread('Char_Image_Binary/' + input[k])
        t = filters.threshold_otsu(A)
        B = (A > t) * 1.0
        a = B.shape[0]
        b = B.shape[1]
        for i in range(a):
            for j in range(b):
                B[i,j] = 1 - B[i,j]
        io.imsave('Char_Image_Binary/' + input[k], B)
    pass

if __name__ == '__main__':
    PreProcessingToCorrect()
