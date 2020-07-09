import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm



os.chdir(r"D:\Data Mining\Sample files")
df = pd.read_csv('Life Expectancy Data.csv')
dataset=df.copy()
dataset.head(10)

# REMOVING UNNECESSARY FEATURES
dataset.drop(['GDP','Population','Income composition of resources', 'Schooling','Total expenditure'],axis=1,inplace=True)


# missing values handling
dataset.isnull().sum()
dataset = dataset[(dataset.Country != 'Monaco')&(dataset.Country != 'San Marino')& (dataset.Country != 'South Sudan') & (dataset.Country != 'Sudan') ]#,'San Marino','Sudan')] # South sudan has alot of missing data.
dataset['Alcohol'].fillna(method='ffill',inplace=True)
dataset['Life expectancy '].fillna(method='ffill',inplace=True)
dataset['Adult Mortality'].fillna(method='ffill',inplace=True)

#Canada and France are mislabeled as Developing
dataset.loc[(dataset.Country == 'Canada'), 'Status'] = "Developed"
dataset.loc[(dataset.Country == 'France'), 'Status'] = "Developed"


dataset.fillna({'Hepatitis_B': df.groupby('Country')['Hepatitis_B'].transform(lambda grp: grp.fillna(np.mean(grp))),
                'Polio': df.groupby('Country')['Polio'].transform(lambda grp: grp.fillna(np.mean(grp))),
                'Diphtheria ': df.groupby('Country')['Diphtheria '].transform(lambda grp: grp.fillna(np.mean(grp)))}
               ,inplace=True)
dataset.fillna({'Hepatitis_B':dataset['Hepatitis_B'].mode().iloc[0]},inplace = True) # some countries dont have this value at all
dataset.isnull().sum()

# ENCODING

dataset.Status = dataset.Status.replace({'Developing':0,
                             'Developed':1})

#QQ plot   
stats.probplot(df_out['Life expectancy '], dist="norm", plot=plt)
plt.title('Life Expectancy QQ Plot')
plt.show()
print(stats.shapiro(dataset['Life expectancy ']))

# Outliers identification
before = dataset.describe()
Q1 = dataset.quantile(0.25)
Q3 = dataset.quantile(0.75)
IQR = Q3 - Q1
outliers = ((dataset < (Q1 - 1.5 * IQR)) |(dataset > (Q3 + 1.5 * IQR)))
df_out = dataset[~((dataset < (Q1 - 1.5 * IQR)) |(dataset > (Q3 + 1.5 * IQR))).any(axis=1)]
after = df_out.describe()

 
X= dataset.drop(['Country' ,'Year','Status'],axis=1)

X = sm.add_constant(X)

y = dataset['Life expectancy ']


X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)          

ols = sm.OLS(y_train,X_train) # target, list of independent variable           
lr = ols.fit()
print(lr.summary())                

while (lr.pvalues.max()>0.05):
    X_train.drop(lr.pvalues.idxmax(),axis=1,inplace=True)
    X_test.drop(lr.pvalues.idxmax(),axis=1, inplace = True)
    ols= sm.OLS(y_train,X_train)
    lr = ols.fit()

X_train.columns # remove the constant column

X_train = X2.drop('const', axis=1, inplace=True)
 

from sklearn.linear_model import LinearRegression                 

model = LinearRegression()                 
                 
#train the model on training set
model.fit(X_train,y_train)            

#cross validation
y_pred = model.predict(X_test) 

from sklearn.metrics import mean_absolute_error, mean_squared_error
mean_abs_error = mean_absolute_error(y_test,y_pred)                
mean_sq_error = mean_squared_error(y_test, y_pred)

import math                 
rmse = math.sqrt(mean_sq_error)
print (rmse,mean_abs_error)                 
                 
