# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:08:18 2020

@author: RAUSHAN KUMAR
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values


#fitting Svr to dataset
from sklearn.svm import SVR
regressor=SVR(kernel = "rbf")
regressor.fit(x,y)

# Visualising  the svr results
plt.scatter(x,y,color="red")
plt.plot(x,len_reg.predict(x), color = "green")
plt.title("truth or bluff")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()
