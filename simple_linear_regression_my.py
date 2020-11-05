# Simple linear regression

# Data preprocessing


#importing the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("salary_Data.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:, 1].values


#spilliting data into test and train set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/3,random_state = 0)


#featuring scaling
'''from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)'''


# Fitting simple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
#Predicting the Test set return
y_pred = regressor.predict(x_test) 


#visualising the Training set result
plt.scatter(x_train ,y_train,color="red")
plt.plot(x_train, regressor.predict(x_train),color="green")
plt.title("Salary vs Experiance(training set)")
plt.xlabel("Years of expriance")
plt.ylabel("salary")
plt.show()


#visualising the Testing set result
plt.scatter(x_test ,y_test,color="red")
plt.plot(x_train, regressor.predict(x_train),color="green")
plt.title("Salary vs Experiance(training set)")
plt.xlabel("Years of expriance")
plt.ylabel("salary")
plt.show()








