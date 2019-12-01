# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 15:05:14 2019

Algoritimo de classificação usando knn 
com validação pelo Método Holdout

@author: David Fune
"""
import pandas as pd
import numpy as np
import matplotlib .pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

dataBase = pd.read_table('bcw.data',delimiter=',', header=None)

#######seleção de colunas relevantes#########
dataBase = dataBase.iloc[:,1:11]

###geração de um vetor com numeros embaralhados do tamanho da base dedos
r = np.random.permutation(len(dataBase))

##aleatoriedade 
base2 = dataBase.iloc[r,:]


treinoK = int (np.floor(len(dataBase)*0.7))
#testeK = int  (np.floor(len(dataBase)*0.3))

## dividindo a base em treino e teste Método Holdout

dataTreino = dataBase[:treinoK]
dataTeste = dataBase[(treinoK+1):]

# substituindo os valores não preenchidos por zero

dataTreino.replace(to_replace='?', value = '0')
dataTeste.replace(to_replace='?', value = '0')

### base de treinamento#####################
aux = dataTreino.iloc[:,9:10]

trueFalse = aux == 4

aux.iloc[trueFalse] = 1

trueFalse = aux == 2

aux.iloc[trueFalse] = -1

### base de teste#####################

aux1 = dataTeste.iloc[:,9:10]

trueFalse = aux1 == 4

aux1.iloc[trueFalse] = 1

trueFalse = aux1 == 2

aux1.iloc[trueFalse] = -1


############## Treinamento e previsão################

classifier = KNeighborsClassifier(n_neighbors=11)

classifier.fit(dataTreino.iloc[:,1:9], aux)

#---------Predição----------------

pred = classifier.predict(dataTeste.iloc[:,1:9])



comparacoN = sum(aux1.iloc[:,0]==-1) - sum(pred==-1)
comparacoP = sum(aux1.iloc[:,0]==1) - sum(pred==1)


#############Matriz de confusão #################################

print(confusion_matrix(aux1, pred), "\n\n\n\n\n")

print(accuracy_score(aux1, pred))

print(classification_report(aux1, pred))
