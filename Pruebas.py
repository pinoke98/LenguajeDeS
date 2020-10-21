# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 17:32:15 2020

@author: jorge
"""
"""
import matplotlib.pyplot as plt

x = [1,2,3,4]
y = [1,2,3,4]

for i in range(len(x)):
    plt.figure()
    plt.plot(x[i],y[i],"ro")
    plt.show()
"""
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report
# from sklearn.neural_network import MLPClassifier
# import pandas as pd

# data  = pd.read_excel("DatosTrain.xlsx")
# data = np.array(data).astype(np.float64)
# data = data.transpose()
# y = data[0]
# x = data[1:18].transpose()

# x = StandardScaler().fit_transform(x)

# # Split the dataset in two equal parts
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# # Set the parameters by cross-validation
# parameters={
# 'learning_rate': ["constant", "invscaling", "adaptive"],
# 'hidden_layer_sizes': [(10,), (20,), (30,)],
# 'alpha': [0.000001],
# 'activation': ["logistic", "relu", "Tanh"]
# }

# scores = ['precision', 'recall']

# print(type(X_train))

# mlp = MLPClassifier(max_iter=100)

# for score in scores:
#     print("# Tuning hyper-parameters for %s" % score)
#     print()

#     clf = GridSearchCV(mlp,parameters,n_jobs=-1,cv=2,scoring='%s_macro'%score)
#     print("D")
#     clf.fit(X_train, y_train)

#     print("Best parameters set found on development set:")
#     print()
#     print(clf.best_params_)
#     print()
#     print("Grid scores on development set:")
#     print()
#     means = clf.cv_results_['mean_test_score']
#     stds = clf.cv_results_['std_test_score']
#     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean, std * 2, params))
#     print()

#     print("Detailed classification report:")
#     print()
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full evaluation set.")
#     print()
#     y_true, y_pred = y_test, clf.predict(X_test)
#     print(classification_report(y_true, y_pred))S
#     print()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_excel("DatosTrain.xlsx")
df = pd.DataFrame(data)
df = df[df['Number'].isin(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21',
                            '22','23','24','25','26'])]

df_y = df['Number']
df_x = df.drop(['Number'], axis = 1)

df_x_scaled = StandardScaler().fit_transform(df_x)

df = pd.DataFrame(df_x_scaled, columns = df_x.columns)
df.insert(0,"Number",df_y,True)
print(df)

def get_class_counts(df):
    grp = df.groupby(['Number'])['Hu0'].nunique()
    return {key: grp[key] for key in list(grp.keys())}

def get_class_proportion(df):
    class_counts = get_class_counts(df)
    return {val[0]: round(val[1]/df.shape[0],4) for val in class_counts.items()}


# y = np.array(y)
# x = np.array(x)
# print(x)
# print("Data set class counts", get_class_counts(df))
# print("Dataset class proportions", get_class_proportion(df))

train,test = train_test_split(df,test_size=0.3,stratify = df['Number'])
train_y = train['Number']
test_y = test['Number']
train_x = train.drop(['Number'],axis = 1)
test_x = test.drop(['Number'],axis = 1)

train_class_proportions = get_class_proportion(train)
test_class_proportions = get_class_proportion(test)

print("Train data class proportions", train_class_proportions)
print("Test data class proportions", test_class_proportions)

parameters={
    'learning_rate': ["constant", "invscaling", "adaptive"],
    'hidden_layer_sizes': [(13),(10),(31),(100),(169),(31,10)],
    'alpha': [0.001,0.0001],
    'activation': ["logistic", "relu", "tanh"],
    'max_iter' : [5000]
    }




