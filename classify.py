# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 23:23:15 2016

@author: Gunjan
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

#reading the .csv file in a pandas dataframe
nba = pd.read_csv("NBAstats.csv")    
original_headers = list(nba.columns.values)

#column 'Pos' as class attribute whose value is to be predicted for each record in the dataset
class_column = 'Pos'
#selected some important attributes required for classification and stored them in a list names feature_columns    
feature_columns =['ORB','AST','eFG%','BLK','3P','STL','TOV','TRB','3PA'] 
   
#splitting data into features and class using column selection
nba_feature = nba[feature_columns]
nba_class = nba[class_column]   
   
#splitting dataset into train_set(75%) and test_set(25%) using train_test_split function
train_feature, test_feature, train_class, test_class = \
    train_test_split(nba_feature, nba_class, stratify=nba_class, \
    train_size=0.75, test_size=0.25)     

#fitting the classification model
linearsvm=SVC(C=10,cache_size=900,kernel='rbf',decision_function_shape=None).fit(train_feature, train_class)  
print("Test set accuracy: {:.3f}".format(linearsvm.score(test_feature, test_class)))

train_class_df = pd.DataFrame(train_class,columns=[class_column])    
train_data_df = pd.merge(train_class_df, train_feature, left_index=True, right_index=True)
train_data_df.to_csv('train_data.csv', index=False)

#predicting the value of test_feature 
prediction = linearsvm.predict(test_feature) 
#printing confusion matrix
print("Confusion matrix:")                    
print(pd.crosstab(test_class, prediction,rownames=['True'], colnames=['Predicted'], margins=True))   

#performing cross-validation using cross_val_score method from model_selection package on the data from feature columns and class column,setting the number of folds to 10
scores = cross_val_score(linearsvm, nba_feature,nba_class,cv=10) 
print("Cross-validation scores: {}".format(scores))
#mean of all 10 folds accuracy
print("Average cross-validation score: {:.2f}".format(scores.mean()))

