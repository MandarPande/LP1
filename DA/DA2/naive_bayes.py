# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 07:38:04 2019

@author: mandar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('diabetes.csv')
data.head()

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)

from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred, target_names = ['NO', 'YES'])
print(report)
