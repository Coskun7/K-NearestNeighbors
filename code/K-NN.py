#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 03:14:24 2024

@author: mali
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('/Users/mali/Downloads/archive/column_2C_weka.csv')

y = df.iloc[:,6].values
x = df.iloc[:,:6]

y = [1 if each == 'Abnormal' else 0 for each in y]

M = df[df['class']=='Abnormal']
B = df[df['class']=='Normal']

plt.scatter(M.pelvic_incidence,M.pelvic_radius,color='red',label='Abnormal')
plt.scatter(B.pelvic_incidence,B.pelvic_radius,color='green',label='Normal')
plt.xlabel('Pelvic Incidence')
plt.ylabel('Pelvic Radius')
plt.show()

x = x.to_numpy()
y = np.array(y)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=7)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=4)

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)

knn.score(x_test,y_test)

J_hist=[]
for i in range(1,15):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    J_hist.append(knn.score(x_test,y_test))
    

plt.plot(range(1,15),J_hist)
plt.xlabel('neighbours values')
plt.ylabel('J_hist')
plt.show()
    
    
    
