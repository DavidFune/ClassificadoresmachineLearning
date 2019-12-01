# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 10:47:18 2019

Algoritimo de classificação usando SVM 
com validação pelo Método Holdout

@author: hunter28
"""

from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing


dataBase = pd.read_table('bcw.data',delimiter=',', header=None)
target = dataBase.iloc[:,10:11]
dataBase = dataBase.iloc[:,1:10]

#Caracteristicas do treinamento, gamma= abertura da gluciana, C = regularização
clf = svm.SVC(gamma=0.001 , C=100.)

dataBase = SelectKBest(chi2, k=2).fit_transform(dataBase, target)

min_max_scaler = preprocessing.MinMaxScaler()

dataBase = min_max_scaler.fit_transform(dataBase)

#classificando sem a ultima linha
clf.fit(dataBase,target)


trainS = int(0.7*(len(dataBase)))

train = clf.fit(dataBase[:trainS], target[:trainS])


pred = clf.predict(dataBase[(trainS+1):])



print('Acuracia da classificação: ', accuracy_score(target[(trainS+1):], pred),'\n\n\n')


print('Matriz de Confusão', '\n', confusion_matrix(target[(trainS+1):], pred), "\n\n\n")



print(classification_report(target[(trainS+1):], pred), '\n')

a = dataBase[(trainS+1):,0:1]

b = dataBase[(trainS+1):,1:2] 

plt.scatter(dataBase[(trainS+1):,0:1], dataBase[(trainS+1):,1:2])
plt.show()