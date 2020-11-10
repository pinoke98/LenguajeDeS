# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 17:09:02 2020

@author: jorge
"""
letras = {1:'A',2:'B'}
import pandas as pd
from joblib import load
import numpy as np

df = pd.read_excel("DatosVal.xlsx")
clf = load("model_clf.pkl")
StandScaler = load("sca_params.pkl")
#print(df)
df_y = np.array(df['Number'])
df_x = np.array(df.drop(['Number'], axis = 1))
df_x_scaled = StandScaler.transform(df_x)
#print(StandScaler.get_params())
#print(df_x)
print("Y:",df_y)
print("Ygorro",clf.predict(df_x_scaled))
    