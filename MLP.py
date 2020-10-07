import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
import pandas as pd

data  = pd.read_excel("DatosTrain.xlsx")
data = np.array(data).astype(np.float64)
data = data.transpose()
y = data[0]
x = data[1:18].transpose()

x = StandardScaler().fit_transform(x)

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Set the parameters by cross-validation
parameters={
'learning_rate': ["constant", "invscaling", "adaptive"],
'hidden_layer_sizes': [(10,), (20,), (30,),(40,),(100,),(1000,),(10000,)],
'alpha': [0.000001],
'activation': ["logistic", "relu", "Tanh"]
}

scores = ['precision', 'recall']

print(type(X_train))

mlp = MLPClassifier(max_iter=100)

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(mlp,parameters,n_jobs=-1,cv=2,scoring='%s_macro'%score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
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
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()