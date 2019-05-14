#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        main
#-------------------------------------------------------------------------------

from PreProcessing import *
from PreProcessingToCorrect import *
from Feature1Extraction import *
from Feature2Extraction import *
from Feature3Extraction import *
from Feature4Extraction import *
from Feature5Extraction import *

if __name__ == '__main__':
    PreProcessing()
    PreProcessingToCorrect()
    Feature1Extraction()
    Feature2Extraction()
    Feature3Extraction()
    Feature4Extraction()
    Feature5Extraction()

