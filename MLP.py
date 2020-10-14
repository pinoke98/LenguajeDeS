# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 12:02:28 2020

@author: jorge
"""
import numpy as np
from time import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def get_class_counts(df):
    grp = df.groupby(['Number'])['Hu0'].nunique()
    return {key: grp[key] for key in list(grp.keys())}

def get_class_proportion(df):
    class_counts = get_class_counts(df)
    return {val[0]: round(val[1]/df.shape[0],4) for val in class_counts.items()}

train_ratio = 0.6
validation_ratio = 0.2
test_ratio = 0.2

data = pd.read_excel("DatosTrain.xlsx")
df = pd.DataFrame(data)
df = df[df['Number'].isin(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21',
                            '22','23','24','25','26'])]

df_y = df['Number']
df_x = df.drop(['Number'], axis = 1)

df_x_scaled = StandardScaler().fit_transform(df_x)

df = pd.DataFrame(df_x_scaled, columns = df_x.columns)
df.insert(0,"Number",df_y,True)
#print(df)


train, test = train_test_split(df, test_size=1-train_ratio, random_state=0,stratify = df['Number'])
val , test = train_test_split(df, test_size=test_ratio/(test_ratio + validation_ratio), random_state=0,stratify = df['Number'])

val_y = val['Number']
train_y = train['Number']
test_y = test['Number']
val_x = val.drop(['Number'], axis = 1)
train_x = train.drop(['Number'], axis = 1)
test_x = test.drop(['Number'], axis = 1)

# train_class_proportions = get_class_proportion(train)
# test_class_proportions = get_class_proportion(test)
# val_class_proportions = get_class_proportion(val)

# train_class_counts = get_class_counts(train)
# test_class_counts = get_class_counts(test)
# val_class_counts = get_class_counts(val)

# print("Train class counts", train_class_counts)
# print("")
# print("Test class counts", test_class_counts)
# print("")
# print("Val class counts", val_class_counts)
# print("")

# print("Train data class proportions", train_class_proportions)
# print("")
# print("Test data class proportions", test_class_proportions)
# print("")
# print("Val data class proportions", val_class_proportions)
# print("")


def Parameters():
    parameters={
    'learning_rate': ["constant", "invscaling", "adaptive"],
    'hidden_layer_sizes': [(13),(10),(31),(100),(169),(31,10)],
    'alpha': [0.001,0.0001],
    'activation': ["logistic", "relu", "tanh"],
    'max_iter' : [5000]
    }
    
    scores = ['precision', 'recall']
    
    best_params = []
    
    print(type(train_x))
    
    mlp = MLPClassifier()
    
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
    
        clf = GridSearchCV(mlp,parameters,n_jobs=-1,cv=2,scoring='%s_macro'%score)
        clf.fit(train_x, train_y)
    
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        best_params.append(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()
    
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = val_y, clf.predict(val_x)
        print(classification_report(y_true, y_pred))
        print()
    return best_params

# def load_xlsx(d):
#     #print(d)
#     d=np.array(d).astype(np.float64)
#     d=d.transpose()
#     y=d[0]
#     x=d[1:13].transpose()
#     print(x)
#     return x,y

if __name__ == '__main__':
    t0 = time()
    parameters = Parameters()
    print(parameters)
    accuracy = 0
    i = 0
    mlp = MLPClassifier(activation = "tanh",alpha = 0.001,hidden_layer_sizes = (31),learning_rate="invscaling",max_iter=10000,tol = 0.00001)
    mlp.fit(train_x,train_y)
    Test_pred = mlp.predict(test_x)
    Test_error = 100 - mlp.score(test_x,test_y)*100
    Train_pred = mlp.predict(train_x)
    Train_error = 100 - mlp.score(train_x,train_y)*100
    Val_pred = mlp.predict(val_x)
    Val_error = 100 - mlp.score(val_x,val_y)*100
    print("Train error: ", Train_error)
    print("Test error: ", Test_error)
    print("Validation error: ", Val_error)
    i+=1
    print ("\n")
    print("done in %0.16fs" % (time() - t0))