import numpy as np
import copy as cp
import matplotlib.pyplot as plt

import seaborn as sns
from typing import Tuple
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import io



def _trainModel(X,y,foldsNumber:int, model,standarized:bool,dimensionReduction:str,dimensionReductionComponents):
   
    actual_classes, predicted_classes, _, avg,pre,f1,recall,spec = __crossKStratifiedValidation(model, foldsNumber, X, y,standarized,dimensionReduction,dimensionReductionComponents)
    plot = __plotConfusionMatrix(actual_classes, predicted_classes, [0, 1])
    return avg,pre,f1,recall,spec,plot


def _printMostImportantFeatureOfComponent(model,initial_feature_names):
    # number of components
    n_pcs= model.components_.shape[0]

    # get the index of the most important feature on EACH component
    # LIST COMPREHENSION HERE
    most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]


    # get the names
    most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

    # LIST COMPREHENSION HERE AGAIN
    dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}

    # build the dataframe
    df = pd.DataFrame(dic.items())
    return df


# Dado un modelo, un número de capas, un dataset con variables y el listado de variables objetivo
# imprime el accuracy, precision, f1
# Se puede configurar si estandarizar los elementos o aplicarles una reducción de dimensión mediante PCA
# model
# k: número capas
# X: variables
# y: objetivo
# standarized: si se quiere que se estandaricen los datos
# dimensionReduction: 'pca'si se quiere que se aplique reducción de dimensión mediante PCA
# dimensionReduction: 'lda' si se quiere que se aplique reducción de dimensión mediante LDA
def __crossKStratifiedValidation(model, k, X, y,standarized,dimensionReduction, dimensionReductionComponents) -> Tuple[np.array, np.array, np.array]:

    kfold = StratifiedKFold(n_splits=k)
    model_ = cp.deepcopy(model)
    acc_score = []
    pre_score = []
    f_score = []
    r_score = []
    spec_score = []
    no_classes = len(np.unique(y))
    actual_classes = np.empty([0], dtype=int)
    predicted_classes = np.empty([0], dtype=int)
    predicted_proba = np.empty([0, no_classes])
    head = X.head()
    X=X.values
    
 

    for train_ndx, test_ndx in kfold.split(X,y):

        train_X, train_y, test_X, test_y = X[train_ndx], y[train_ndx], X[test_ndx], y[test_ndx]

        actual_classes = np.append(actual_classes, test_y)

        if standarized == True:                                           
            scaler = StandardScaler()
            # Fit on the train set only
            scaler.fit(train_X)
            # Apply to both the train set and the test set. 
            train_X = scaler.transform(train_X)
            test_X = scaler.transform(test_X)
        
        if dimensionReduction == 'pca':
            # Apply PCA
            pca = PCA(n_components=dimensionReductionComponents)
            # Fit on the train set only
            pca.fit(train_X)
            # Apply transform to both the train set and the test set. 
            train_X = pca.transform(train_X)
            test_X = pca.transform(test_X)
           
    
    
        model_.fit(train_X, train_y)
        pred_values = model_.predict(test_X)
        acc = accuracy_score(list(pred_values), list(test_y))
        pre = precision_score(list(pred_values), list(test_y))
        f1 = f1_score(list(pred_values), list(test_y))
        r = recall_score(list(pred_values), list(test_y))
        spec = recall_score(list(pred_values), list(test_y), pos_label=0)
        acc_score.append(acc)
        pre_score.append(pre)
        f_score.append(f1)
        r_score.append(r)
        spec_score.append(spec)
        predicted_classes = np.append(predicted_classes, pred_values)
        avg_acc_score = sum(acc_score)/k
        avg_pre_score = sum (pre_score)/k
        avg_f1_score = sum (f_score)/k
        avg_recall_score = sum(r_score)/k
        avg_spec_score = sum(spec_score)/k


        try:
            predicted_proba = np.append(predicted_proba, model_.predict_proba(test_X), axis=0)
        except:
            predicted_proba = np.append(predicted_proba, np.zeros((len(test_X), no_classes), dtype=float), axis=0)


    print('Avg accuracy : {}'.format(avg_acc_score))
    print('Avg precission : {}'.format(avg_pre_score))
    print('Avg recall : {}'.format(avg_recall_score))
    print('Avg f1 : {}'.format(avg_f1_score))

    return actual_classes, predicted_classes, predicted_proba, avg_acc_score, avg_pre_score, avg_f1_score, avg_recall_score,avg_spec_score

# Imprime la matriz confusión
def __plotConfusionMatrix(actual_classes : np.array, predicted_classes : np.array, sorted_labels : list):

    matrix = confusion_matrix(actual_classes, predicted_classes, labels=sorted_labels)
   

    
    plt.figure(figsize=(12.8,6))
    sns.heatmap(matrix, annot=True, xticklabels=sorted_labels, yticklabels=sorted_labels, cmap="Blues", fmt="g")
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.clf()
    return buf





