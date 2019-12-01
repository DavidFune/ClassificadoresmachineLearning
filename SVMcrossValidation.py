# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 15:16:53 2019

Algoritimo de classificação usando  foi SVM, 
a dimensão do problema foi alterada para 2
e foi feita uma normalização dos dados  
e também validação pelo Cross Validation

@author: hunter28
"""
from sklearn import svm
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing



dataBase = pd.read_table('bcw.data',delimiter=',', header=None)
target = dataBase.iloc[:,9:10]
dataBase = dataBase.iloc[:,1:10]

dataBase = SelectKBest(chi2, k=2).fit_transform(dataBase, target)

min_max_scaler = preprocessing.MinMaxScaler()

dataBase = min_max_scaler.fit_transform(dataBase)


clf = svm.SVC(gamma=0.001 , C=1)

scores = cross_val_score(clf, dataBase, target, cv=5)






print("Acurácia: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2),'\n\n')