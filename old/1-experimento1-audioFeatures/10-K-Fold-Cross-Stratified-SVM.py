import pandas as pd
import numpy as np
import copy as cp
import matplotlib.pyplot as plt

import seaborn as sns
from typing import Tuple
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import decomposition

def cross_val_predict(model, k, X, y) -> Tuple[np.array, np.array, np.array]:

    kfold = StratifiedKFold(n_splits=k)
    model_ = cp.deepcopy(model)
    acc_score = []
    
    no_classes = len(np.unique(y))
    
    actual_classes = np.empty([0], dtype=int)
    predicted_classes = np.empty([0], dtype=int)
    predicted_proba = np.empty([0, no_classes]) 

    for train_ndx, test_ndx in kfold.split(X,y):

        train_X, train_y, test_X, test_y = X[train_ndx], y[train_ndx], X[test_ndx], y[test_ndx]

        actual_classes = np.append(actual_classes, test_y)

        model_.fit(train_X, train_y)
        pred_values = model_.predict(test_X)
        acc = accuracy_score(pred_values , test_y)
        acc_score.append(acc)
        predicted_classes = np.append(predicted_classes, pred_values)
        avg_acc_score = sum(acc_score)/k

        try:
            predicted_proba = np.append(predicted_proba, model_.predict_proba(test_X), axis=0)
        except:
            predicted_proba = np.append(predicted_proba, np.zeros((len(test_X), no_classes), dtype=float), axis=0)

    print('accuracy of each fold - {}'.format(acc_score))
    print('Avg accuracy : {}'.format(avg_acc_score))

    return actual_classes, predicted_classes, predicted_proba

def plot_confusion_matrix(actual_classes : np.array, predicted_classes : np.array, sorted_labels : list):

    matrix = confusion_matrix(actual_classes, predicted_classes, labels=sorted_labels)
    
    plt.figure(figsize=(12.8,6))
    sns.heatmap(matrix, annot=True, xticklabels=sorted_labels, yticklabels=sorted_labels, cmap="Blues", fmt="g")
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
    plt.show()
    plt.savefig('health-confusion-matrix.png')


dataset = pd.read_csv('health_dataset.csv')

# Separate features and target variable
X = dataset.iloc[:, 2:-1].values
y = dataset. iloc [:, 1].values





model = svm.SVC(gamma='scale')
k=10

actual_classes, predicted_classes, _ = cross_val_predict(model, k, X, y)
plot_confusion_matrix(actual_classes, predicted_classes, [0, 1])




