# -*- coding: utf-8 -*-
"""
Breast Cancer Detection Using Machine Learning Project
"""

#Step 1: Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#step 2: Load the data

dataset = pd.read_csv('data.csv')

######################### Analysing the Dataset  #############################

#Using the shape method to Count the number of rows/samples and columns/predictors in that dataset
# The print method is used to display the result
dataset.shape


#Using the info method provides brief description of the columns in the dataset, similar to the describe keyword in sql db
#information like the column name, Column datatype etc...
dataset.info()

#Frequency of Unique Values of the target variable(Technically known as The Dependeent Variable)
print(dataset['diagnosis'].value_counts())


#Understanding the frequency Distribution of data Unique Values of the target variable (Diagnosis)
#I Gave it a label(Count) on Y the Y-axis for easier understanding

sns.countplot(dataset['diagnosis'], label='count')

######################### Analysing the Dataset End  #############################

######################### Dataset Cleaning  #############################


# Check for empty the number of empty(NAN, NaN, na) values in each column
print(dataset.isna().sum())

#Drop the column with the missing values
dataset = dataset.dropna(axis = 1)


#Checking if the missing column has being dropped
print(dataset.shape)

#checking the datatypes of all columns for categorical data
print(dataset.dtypes)

# Encoding the categorical values
from sklearn.preprocessing import LabelEncoder
labelEncoder_Y = LabelEncoder()
dataset.iloc[ : , 1] = labelEncoder_Y.fit_transform(dataset.iloc[: , 1].values)
print(dataset.iloc[: , 1].values)

#Split the dataset to dependent variable (Y) and independent variable (X) dataset
X = dataset.iloc[: , 2:13].values
Y = dataset.iloc[: , 1].values

#Split the dataset into 75%  training set and 25% testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#scale the data (This is called Feature scaling)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


#Creating a functions for the models
def models (X_train, Y_train):
    
    #Logistic Regression
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state = 0)
    log.fit(X_train, Y_train)
    
    #Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    tree.fit(X_train, Y_train)
    
    #Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    forest.fit(X_train, Y_train)
    #Print the models accuracy on the traing data
    print('[0] Logistic Regression Training Accuracy', log.score(X_train, Y_train))
    print('[1] Decision Tree Classifier Training Accuracy', tree.score(X_train, Y_train))
    print('[2] Random Forest Classifier Training Accuracy', forest.score(X_train, Y_train))
    
    return log, tree, forest


#Getting all of the models
model = models(X_train, Y_train)

#Test model accuracy on the test data using Confusion matrix
from sklearn.metrics import classification_report, accuracy_score
for i in range(len(model)):
    print('Model ', i)
    print(classification_report(Y_test, model[i].predict(X_test)))
    print(accuracy_score(Y_test, model[i].predict(X_test)))
    print()
    
    

