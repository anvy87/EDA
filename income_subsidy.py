# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 18:33:21 2020

@author: anvesh
"""


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

#   importing data
data_income = pd.read_csv('income.csv')
data = data_income.copy()

'''  exploratory data analysis '''

#   to check variables' data type
print(data.info())

#   check for missing values
data.isnull()
print(data.isnull().sum()) # no missing value

#   summary of numerical variables
summary_num = data.describe()
print(summary_num)

#   summary of categorical variables
summary_cate = data.describe(include = "O")
print(summary_cate)

#   frequency of each categories
data['JobType'].value_counts()
data['occupation'].value_counts()

#   checking for unique classes
print(np.unique(data['JobType']))
print(np.unique(data['occupation']))
# there exists '?' instead of nan

#   read the data again by including "na_values[" ?"]
data = pd.read_csv('income.csv', na_values=[" ?"])

'''
    Data pre-preocessing
'''
data.isnull().sum()

#   subsetting the rows
missing = data[data.isnull().any(axis=1)]
# axis=1 => to consider at least one column value is missing

'''
    Missing values in JobType = 1809
    Missing values in occupation = 1816
    There are 1809 rows where two specific columns
    i.e. occupation and JobType have missingvalues
    1816-1809= 7 => occupation is unfilled for these 7 rows
    because JobType is Never worked
'''

data2 = data.dropna(axis=0)


#   Relationship between independent variables
correlation = data2.corr()
# none of the variables are correlated

#   Cross Tables & Data Visualization
#extracting the column names
data2.columns

#   Gender proportion table
gender_salstat = pd.crosstab(index   = data2["gender"],
                     columns = data2['SalStat'],
                     margins = True,
                     normalize = 'index')
print(gender_salstat)


'''
    Frequency distribution of 'Salary Status'
'''
SalStat = sns.countplot(data2['SalStat'])

#   Histogram of Age
sns.distplot(data2['age'], bins=10, kde=False)

#   Box Plot - Age vs Salary Status
sns.boxplot('SalStat', 'age', data=data2)
data2.groupby('Salstat')['age'].median()


'''
    *******************************
           LOGISTIC REGRESSION
    *******************************
'''

#   Reindexing the salary status names to 0,1
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

new_data = pd.get_dummies(data2, drop_first=True)

#   Storing the column names
columns_list = list(new_data.columns)
print(columns_list)

#   Separating the input names from data
features = list(set(columns_list)-set(['SalStat']))
print(features)

#   Storing the output values in y
y = new_data['SalStat'].values
print(y)

#   Storing the values from input features
x = new_data[features].values
print(x)

#   Splitting the data intro train and test
train_x, test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state=0)

#   Make an instance of the Model
logistic = LogisticRegression()

#   Fitting the values for x and y
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_

#   Prediction from test data
prediction = logistic.predict(test_x)
print(prediction)

#   Confusion matrix
confusion_matrix = confusion_matrix(test_y, prediction)
print(confusion_matrix)

#   Calculating the accuracy
accuracy_score = accuracy_score(test_y, prediction)
print(accuracy_score)

#   Printing the misclassified values from prediction

print('misclassified samples: %d' % (test_y != prediction).sum())


'''
    *******************
    LOGISTIC REGRESSION - REMOVING INSIGNIFICANT VARIABLES
    *******************
'''

#   Reindexing the salary status names to 0,1
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

cols = ['gender', 'nativecountry', 'race', 'JobType']
new_data = data2.drop(cols, axis=1)

new_data = pd.get_dummies(new_data, drop_first=True)

#   Storing the column names
columns_list = list(new_data.columns)
print(columns_list)

#   Separating the input names from data
features = list(set(columns_list)-set(['SalStat']))
print(features)

#   Storing the output values in y
y = new_data['SalStat'].values
print(y)

#   Storing the values from input features
x = new_data[features].values
print(x)

#   Splitting the data intro train and test
train_x, test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state=0)

#   Make an instance of the Model
logistic = LogisticRegression()

#   Fitting the values for x and y
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_

#   Prediction from test data
prediction = logistic.predict(test_x)
print(prediction)

#   Confusion matrix
confusion_matrix = confusion_matrix(test_y, prediction)
print(confusion_matrix)

#   Calculating the accuracy
accuracy_score = accuracy_score(test_y, prediction)
print(accuracy_score)

#   Printing the misclassified values from prediction

print('misclassified samples: %d' % (test_y != prediction).sum())


'''
    *************
        KNN
    *************
'''

#   importing the library of KNN
from sklearn.neighbors import KNeighborsClassifier

#   import library for plotting
import matplotlib.pyplot as plt

#   Storing the K nearest neighbors classifier
KNN_classifier = KNeighborsClassifier(n_neighbors = 5)

#   Fitting the values for X and Y
KNN_classifier.fit(train_x, train_y)

#   Predicting the test values with model
prediction = KNN_classifier.predict(test_x)

#   Performance metric check
confusion_matrix = confusion_matrix(test_y, prediction)
print("\t", "Predicted values")
print("Original values", "\n", confusion_matrix)

#   Calculating the accuracy
accuracy_score = accuracy_score(test_y, prediction)
print(accuracy_score)

print('Misclassified samples: %d' % (test_y != prediction).sum())

'''
    Effect of K value on classifier
'''
Misclassified_sample = []
#   Calculating error for K values between 1 and 20
for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x, train_y)
    pred_i = knn.predict(test_x)
    Misclassified_sample.append((test_y != pred_i).sum())

print(Misclassified_sample)

