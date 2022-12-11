# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 21:20:52 2022

@author: ashwi
"""
import pandas as pd
df = pd.read_csv('Salary_Data.csv')
df.head()
df.shape
df.dtypes

# blanks
df.isnull().sum()

# finding duplicate rows
df.duplicated()
df[df.duplicated()] # hence no duplicates between the rows


# finding duplicate columns
df.columns.duplicated() # hence no duplicates between the column

X = df[["YearsExperience"]]
Y = df['Salary']
df.corr()
#====================================================================
# histogram
df["YearsExperience"].hist()
df["YearsExperience"].skew()
# the YearsExperience is +ve skewness
df["Salary"].hist()
df["Salary"].skew()
# the YearsExperience is +ve skewness
#====================================================================

import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(14,4))
plt.subplot(121)
sns.distplot(df["YearsExperience"])
plt.title("YearsExperience PDF")

plt.subplot(122)
stats.probplot(df["YearsExperience"],dist="norm",plot=plt)
plt.title("YearsExperience QQ plot")
plt.show()
# by this i have seen the probability distribusion function and QQ plot of YearsExperience and distribusion is normal 
#==========================================================================
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(14,4))
plt.subplot(121)
sns.distplot(df["Salary"])
plt.title("Salary") 

plt.subplot(122)
stats.probplot(df["Salary"],dist="norm",plot=plt)
plt.title("Salary QQ plot")
plt.show()
# by this i have seen the probability distribusion function and QQ plot of Salary and distribusion is normal 
#===============================================================
# scatter plot
import matplotlib.pyplot as plt
plt.scatter(X,Y, color='black',alpha=0.5)
plt.show()

#============================================================
# boxplot
df.boxplot(column="YearsExperience",vert=False)

import numpy as np
Q1 = np.percentile(df["YearsExperience"],25)
Q2 = np.percentile(df["YearsExperience"],50)
Q3 = np.percentile(df["YearsExperience"],75)
IQR = Q3 - Q1
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df["YearsExperience"]<LW) | (df["YearsExperience"]>UW)]
len(df[(df["YearsExperience"]<LW) | (df["YearsExperience"]>UW)])
# Therefore in YearsExperience variabel there are Zero outlaiers 

#============================================================

df.boxplot(column="Salary",vert=False)

import numpy as np
Q1 = np.percentile(df["Salary"],25)
Q2 = np.percentile(df["Salary"],50)
Q3 = np.percentile(df["Salary"],75)
IQR = Q3 - Q1
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df["Salary"]<LW) | (df["Salary"]>UW)]
len(df[(df["Salary"]<LW) | (df["Salary"]>UW)])
# Therefore in Salary variabel there are Zero outlaiers 

#===========================================================
# model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)

# Bo
LR.intercept_
# B1
LR.coef_

# predictions
Y_pred = LR.predict(X)
Y_pred

Y_pred =pd.DataFrame(Y_pred)

#scatter plot of actul value and predicted value
import matplotlib.pyplot as plt
plt.scatter(X,Y,color='black')
plt.scatter(X,Y_pred,color='red')
plt.show()

# Matrix
# MSE
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y, Y_pred)
print("mean squred error",mse.round(3))
# RMSE
import numpy as np
print("Root mean squared error", np.sqrt(mse).round(3))

#============================================================
# log Transformation
X_log= np.log(X)

Y_pred_Transform = LR.predict(X_log)
Y_pred_Transform

# matrix
# MSE
from sklearn.metrics import mean_squared_error
mse_1 = mean_squared_error(Y, Y_pred_Transform)
print('Mean squared error',mse.round(3))

# RMSE
import numpy as np
print("Root mean squared error", np.sqrt(mse_1).round(3))

#==========================================================
# Square Root Transformation
X_log1= np.sqrt(X)

Y_pred_Transform1 = LR.predict(X_log1)
Y_pred_Transform1

# matrix
from sklearn.metrics import mean_squared_error
mse_2 = mean_squared_error(Y, Y_pred_Transform1)
print('Mean squared error',mse.round(3))

# RMSE
import numpy as np
print("Root mean squared error", np.sqrt(mse_2).round(3))
#==========================================================
# Cube Root Transformation
X_log2= np.cbrt(X)

Y_pred_Transform2 = LR.predict(X_log2)
Y_pred_Transform2

# matrix
from sklearn.metrics import mean_squared_error
mse_3 = mean_squared_error(Y, Y_pred_Transform2)
print('Mean squared error',mse.round(3))

# RMSE
import numpy as np
print("Root mean squared error", np.sqrt(mse_3).round(3))
#============================================================

import statsmodels.api as sma
X_new = sma.add_constant(X)
X_new
lm2 = sma.OLS(Y,X_new).fit()
lm2.summary()

# therefore by comeparing all the Transformations i find out that without using the Transformation we are getting the best resluts






