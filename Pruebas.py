# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 17:32:15 2020

@author: jorge
"""
import matplotlib.pyplot as plt

x = [1,2,3,4]
y = [1,2,3,4]

for i in range(len(x)):
    plt.figure()
    plt.plot(x[i],y[i],"ro")
    plt.show()