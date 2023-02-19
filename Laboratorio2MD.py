#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 13:47:23 2023

@author: mito
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
##import seaborn as sns

from pandas_profiling import ProfileReport
pd.options.mode.chained_assignment = None


#Analisis exploratorio
datos = pd.read_csv('breast-cancer-wisconsin.csv')

#profile = ProfileReport(datos, title="Pandas Profiling Report")
#profile. to_file("Reporte antes de la limpieza.html")
indices=[]

#Limpieza de datos
for i in range(len(datos)):
    if(datos.iloc[i]['Bare_Nuclei']=="?"):
       indices.append(i)
datos.drop(indices, axis=0, inplace=True)


datos=datos.drop(['id','cell_shape','marginal_Adhesion','Epithehtial_Cell_Size','Bland_Chromatin' ], axis=1)

#profile = ProfileReport(datos, title="Pandas Profiling Report")
#profile. to_file("Reporte depues de primera la limpieza.html")


#Dividir datos en conjutnos de prueba
X = datos.iloc[:, 0:4].values
y = datos.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_entreno, X_prueba, y_entreno, y_prueba = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.linear_model import LogisticRegression

modeloLog = LogisticRegression(max_iter = 500)
modeloLog.fit(X_entreno,y_entreno)
predicciones = modeloLog.predict(X_prueba)

from sklearn.metrics import classification_report

print(classification_report(y_prueba, predicciones))



