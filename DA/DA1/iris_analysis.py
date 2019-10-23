# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


columns = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv('iris.csv', names = columns)

dataset.head()


#Mean calculation
mean = dataset.mean(axis = 0)
print(mean)

dataset.mean()
#minimum
minimum = dataset.min(axis = 0)
print(minimum)


#maximum
maximum = dataset.max(axis = 0)
print(maximum)


range_list = []
range_list[:-1:] = maximum[:-1:] - minimum[:-1:]
print(range_list)


variance = dataset.var(axis = 0)
print(variance)

std_dev = dataset.std(axis = 0)
print(std_dev)


percentile=dataset.quantile(q=0.5,axis=0)
print(percentile)

histogram_plots = dataset.hist(bins = 100)

boxplot_sepal_length=dataset.boxplot(column=['sepal-length'])
boxplot_sepal_width=dataset.boxplot(column=['sepal-width'])
boxplot_petal_length=dataset.boxplot(column=['petal-length'])
boxplot_petal_width=dataset.boxplot(column=['petal-width'])

bp= dataset.boxplot()
