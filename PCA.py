# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 10:03:12 2022

@author: chandu
"""

# Chronic Kidney Disease Prediction 
## Using PCA for Dimension Reduction & Random forest Classifier for Predicting 

### importing required libraries

import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r'C:\Users\chandu\Desktop\Course\Data Science\Machine_Learning_Practice\3_PCA\Chronic_Kidney_Disease\kidney_disease.csv')

### Data is Semi-Structured, Offline, Cross Sectional and Regular Data
df.head()
df.columns

## Checking whether the Data is Balanced vs Imbalanced 
## y is ckd/no-ckd

df['classification'].value_counts() ### Data is Imbalanced 

## Dropping ID Column which is not relavent
df.drop('id', axis = 1, inplace = True)

# Data Preprocessing stage

## Typecasting
df.info() 

## pcv,wc,rc are numerical datapoints but they are read as object type so typecasting for them
df['pcv'] = pd.to_numeric(df['pcv'], errors = 'coerce')
df['wc'] = pd.to_numeric(df['wc'], errors = 'coerce')
df['rc'] = pd.to_numeric(df['rc'], errors = 'coerce')

### Checking for Duplicates
df.duplicated().sum()

### Checking for Missing values
df.isna().sum() 

df['age'] = df['age'].fillna(df['age'].median())
df['bp'] = df['bp'].fillna(df['bp'].median())
df['sg'] = df['sg'].fillna(df['sg'].median())
df['al'] = df['al'].fillna(df['al'].median())
df['su'] = df['su'].fillna(df['su'].mode()[0])
df['rbc'] = df['rbc'].fillna(df['rbc'].mode()[0])
df['pc'] = df['pc'].fillna(df['pc'].mode()[0])
df['pcc'] = df['pcc'].fillna(df['pcc'].mode()[0])
df['ba'] = df['ba'].fillna(df['ba'].mode()[0])
df['bgr'] = df['bgr'].fillna(df['bgr'].median())
df['bu'] = df['bu'].fillna(df['bu'].median())
df['sc'] = df['sc'].fillna(df['sc'].median())
df['sod'] = df['sod'].fillna(df['sod'].median())
df['pot'] = df['pot'].fillna(df['pot'].median())
df['hemo'] = df['hemo'].fillna(df['hemo'].median())
df['pcv'] = df['pcv'].fillna(df['pcv'].median())
df['wc'] = df['wc'].fillna(df['wc'].median())
df['rc'] = df['rc'].fillna(df['rc'].median())
df['htn'] = df['htn'].fillna(df['htn'].mode()[0])
df['dm'] = df['dm'].fillna(df['dm'].mode()[0])
df['cad'] = df['cad'].fillna(df['cad'].mode()[0])
df['appet'] = df['appet'].fillna(df['appet'].mode()[0])
df['pe'] = df['pe'].fillna(df['pe'].mode()[0])
df['ane'] = df['ane'].fillna(df['ane'].mode()[0])

## Converting Non - Numerical to Numerical 
### Since All Non - Numerical Variables are Nominal we can perform One Hot Encoding

## Getting all the Non - Numerical variables to one dataframe
categorical_variables = df[['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']]

Dummy_variables = pd.get_dummies(categorical_variables)

Numerical_variables = df[['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']]

Data = pd.concat([Numerical_variables, Dummy_variables], axis = 1)

### Data Preprocessing Step is Completed 

# Performing PCA - Dimension Reduction without Scaling

from sklearn.decomposition import PCA

pca = PCA()

pca_values_without_Scaling = pca.fit_transform(Data)

columns = list(range(1,39))

pca_df_without_Scaling = pd.DataFrame(pca_values_without_Scaling, columns = columns)

var_ratio_without_Scaling = pca.explained_variance_ratio_

var_without_Scaling = np.cumsum(np.round(var_ratio_without_Scaling, decimals = 4)*100)

var_df_without_Scaling = pd.DataFrame(var_without_Scaling)
var_df_without_Scaling = np.transpose(var_df_without_Scaling)
var_df_without_Scaling = pd.DataFrame(var_df_without_Scaling, columns = columns)

pca_df_without_Scaling = pca_df_without_Scaling.append(var_df_without_Scaling, ignore_index = True)

pca_df_without_Scaling.rename({400: 'Variance'}, axis = 'index', inplace = True) ## The PCA Without Scaling doesnot seems right


# Performing PCA With Scaling
from sklearn.preprocessing import scale

data_norm = pd.DataFrame(scale(Data))

desc_norm = data_norm.describe()

pca_values_with_Scaling = pca.fit_transform(data_norm)

pca_df_with_Scaling = pd.DataFrame(pca_values_with_Scaling, columns = columns)

var_ratio_with_Scaling = pca.explained_variance_ratio_

var_with_Scaling = np.cumsum(np.round(var_ratio_with_Scaling, decimals = 4)*100)

var_df_with_Scaling = pd.DataFrame(var_with_Scaling)

var_df_with_Scaling = np.transpose(var_df_with_Scaling)
var_df_with_Scaling = pd.DataFrame(var_df_with_Scaling, columns = columns)

pca_df_with_Scaling = pca_df_with_Scaling.append(var_df_with_Scaling, ignore_index = True)

pca_df_with_Scaling.rename({400: 'Variance'}, axis = 'index', inplace = True) ## The PCA Without Scaling  seems right

# By taking 18 PCA values we can extract 90% of Data from 38 Features
# Seems like PCA should be happened for Scaled data only and By then only it will be Successfull



