# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 08:18:19 2020

@author: home
"""


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

#   importing data
data_income = pd.read_excel('CreditWorthiness1.xlsx')
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
data['creditScore'].value_counts()

#   checking for unique classes
print(np.unique(data['JobType']))
print(np.unique(data['creditScore']))
# there exists '?' instead of nan

#   read the data again by including "na_values[" ?"]
data = pd.read_excel('creditworthiness1.xlsx', na_values=[" ?"])


'''
    Data pre-preocessing
'''
data.isnull().sum()

#   Relationship between independent variables
correlation = data.corr()

data.columns

age_Cbal = pd.crosstab(index   = data["age"],
                     columns = data['Cbal'],
                     margins = True,
                     normalize = 'index')
print(age_Cbal)


'''
    Frequency distribution of 'Salary Status'
'''
Cbal = sns.countplot(data['Cbal'])
Camt = sns.countplot(data['Camt'])

#   Histogram of Age
sns.distplot(data['age'], bins=10, kde=False)

#   Box Plot - Age vs Cbal
sns.boxplot('Camt', 'Cpur', data=data)
data.groupby('Camt')['Cpur'].median()






