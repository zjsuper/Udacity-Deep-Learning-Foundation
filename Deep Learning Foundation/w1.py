# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 23:55:52 2017

@author: zjgsw
"""
import numpy as np

# 2 by 2 matrices
w1  = np.array([[1, 2], [3, 4]])
w2  = np.array([[5, 6], [7, 8]])

# flatten
w1_flat = np.reshape(w1, -1)
w2_flat = np.reshape(w2, -1)


print (len(w1_flat))
w = np.concatenate((w1_flat, w2_flat))
# array([1, 2, 3, 4, 5, 6, 7, 8])