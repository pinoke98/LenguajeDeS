# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 08:35:16 2020

@author: jorge
"""

def warn(*args, **kwargs):
    pass
import warnings

from time import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from joblib import dump

def get_class_counts(df):
    grp = df.groupby(['Number'])['Hu0'].nunique()
    return {key: grp[key] for key in list(grp.keys())}

def get_class_proportion(df):
    class_counts = get_class_counts(df)
    return {val[0]: round(val[1]/df.shape[0],4) for val in class_counts.items()}

while(True):
    t0 = time()
    df = pd.read_excel("DatosTrain.xlsx")
    df.info()
    #print(df)
    df_y = df['Number']
    df_x = df.drop(['Number'], axis = 1)
    StandScaler = StandardScaler()
    df_x_scaled = StandScaler.fit_transform(df_x)
    #print(StandScaler.get_params())
    
    df = pd.DataFrame(df_x_scaled, columns = df_x.columns)
    df.insert(0,"Number",df_y,True)
    #print(df)
    train,test = train_test_split(df,test_size=0.25,stratify=df['Number'])
    
    # train_class_proportions = get_class_proportion(train)
    # test_class_proportions = get_class_proportion(test)
    
    # train_class_counts = get_class_counts(train)
    # test_class_counts = get_class_counts(test)
    
    # print("Train class counts", train_class_counts)
    # print("")
    # print("Test class counts", test_class_counts)
    # print("") 
    
    # print("Train data class proportions", train_class_proportions)
    # print("")
    # print("Test data class proportions", test_class_proportions)
    # print("")
    
    train_y = train['Number']
    test_y = test['Number']
    train_x = train.drop(['Number'], axis = 1)
    test_x = test.drop(['Number'], axis = 1)
    
    parameters={
        'hidden_layer_sizes': [(64),(32),(64,32)],
        }
    
    score = 'recall'
    
    mlp = MLPClassifier(learning_rate="invscaling", alpha=0.0001,max_iter=10000,activation="logistic")
    print("# Tuning hyper-parameters for %s" % score)
    print()
    
    clf = GridSearchCV(mlp,parameters,n_jobs=-1,cv=10,scoring='%s_macro'%score)
    clf.fit(train_x, train_y)
    
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("With the precition of:")
    print()
    print(clf.best_score_)
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
    y_true, y_pred = test_y, clf.predict(test_x)
    print(classification_report(y_true, y_pred))
    print()
    Test_pred = clf.predict(test_x)
    # print("Ygorro:",Test_pred)
    Test_error = 100 - clf.score(test_x,test_y)*100
    Train_error = 100 - clf.score(train_x,train_y)*100
    print("Train error: ", Train_error)
    print("Test error: ", Test_error)
    print ("")
    print("done in %0.16fm" % ((time() - t0)/60))
    print("")
    if ( Train_error>=0 and Test_error<15):
        break
dump(StandScaler,'sca_params25.pkl')
dump(clf,'model_clf25.pkl')
