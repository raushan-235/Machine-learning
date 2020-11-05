# Data preprocessing
#importing the library


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("50_Startups.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:, 4].values




# Encoding catogerical data
from sklearn.preprocessing import LabelEncoder

labelencoder_x= LabelEncoder()
x[:,3]=labelencoder_x.fit_transform(x[:,3])


#encoding categorical variable
from sklearn.preprocessing import OneHotEncoder
onehotencoder= OneHotEncoder(categorical_features=[3])
x= onehotencoder.fit_transform(x).toarray()


#avoid the dummy veriable trap
x=x[:,1:]

#Splitting data set 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


#Prediction the test set resulrt
y_pred = regressor.predict(x_test)
print(y_test[9])


# Building the optimal model using backword elemination
import statsmodels.formula.api as sm
x = np.append(arr=np.ones((50, 1)).astype(int), values=x, axis=1)
x_opt = x[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog= y,exog= x_opt).fit()
regressor_OLS.summary()
x_opt = x[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog= y,exog= x_opt).fit()
regressor_OLS.summary()
x_opt = x[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog= y,exog= x_opt).fit()
regressor_OLS.summary()
x_opt = x[:,[0,3,5]]
regressor_OLS = sm.OLS(endog= y,exog= x_opt).fit()
regressor_OLS.summary()
x_opt = x[:,[0,3]]
regressor_OLS = sm.OLS(endog= y,exog= x_opt).fit()
regressor_OLS.summary()



regressor.fit(x_opt[[40,41,42,43,44,45,46,47,48,49],1:],y_train)
y_final =  regressor.predict(x_test)
