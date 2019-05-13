
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate,StratifiedKFold
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from statistics import mean
import csv


# In[9]:


X = pd.read_csv("X.csv",sep = ' ', header=None,dtype=float)
X = X.values
y = pd.read_csv("y_bush_vs_others.csv",sep = ' ', header=None,dtype=float)
y_bush = y.values.ravel()


# # This is for CSV file creation

# In[5]:


# headerOne = ['Parameters', ',n_components',  'kernel', 'degree',  'gamma',  'result1', 'result2', 'result3', 'mean result']
headerOne = ['Parameters', 'n_components', 'result1', 'result2', 'result3', 'mean result']
williamFile = open('bushOutput.csv','a+')
with williamFile:
    writer = csv.DictWriter(williamFile, lineterminator='\n', fieldnames=headerOne)
    writer.writeheader()


# In[14]:


def writeTOBushFile(cv, sbr):
    williamFile = open('bushOutput.csv','a+')
    with williamFile:
        writer = csv.DictWriter(williamFile, lineterminator='\n', fieldnames=headerOne)
    #     writer.writeheader()
        writer.writerow({'Parameters':''})
        writer.writerow({'Parameters':'fit_time','n_components': cv,'result1': sbr.get('fit_time')[0],'result2': sbr.get('fit_time')[1],'result3': sbr.get('fit_time')[2], 'mean result':mean(sbr.get('fit_time'))})
        writer.writerow({'Parameters':'score_time','n_components': cv,'result1': sbr.get('score_time')[0],'result2': sbr.get('score_time')[1],'result3': sbr.get('score_time')[2], 'mean result':mean(sbr.get('score_time'))})
        writer.writerow({'Parameters':'test_f1','n_components': cv,'result1': sbr.get('test_f1')[0],'result2': sbr.get('test_f1')[1],'result3': sbr.get('test_f1')[2], 'mean result':mean(sbr.get('test_f1'))})
        writer.writerow({'Parameters':'test_precision','n_components': cv,'result1': sbr.get('test_precision')[0],'result2': sbr.get('test_precision')[1],'result3': sbr.get('test_precision')[2], 'mean result':mean(sbr.get('test_precision'))})
        writer.writerow({'Parameters':'test_recall','n_components': cv,'result1': sbr.get('test_recall')[0],'result2': sbr.get('test_recall')[1],'result3': sbr.get('test_recall')[2], 'mean result':mean(sbr.get('test_recall'))})


# # Bush SVC code test

# In[13]:


# Fit and transform the full dataset using PCA
n_comps_val = [1,2,4,8,16,32,64,128,256,512, 1024, 2048, 4000]#,2,4,8,16,32,128,256,512
for cv in n_comps_val:
    pca = PCA(n_components = cv)
    X_pca = pca.fit(X).transform(X)
    knn_clf = KNeighborsClassifier(n_neighbors=1)
    bush_stratified_cv_results_ft = cross_validate(knn_clf, X_pca, y_bush, 
                                       cv=StratifiedKFold(n_splits = 3, shuffle=True, random_state = 8854), 
                                       scoring=('f1', 'precision', 'recall'), 
                                       return_train_score=False)
    writeTOBushFile(cv, bush_stratified_cv_results_ft)


# In[19]:


n_comps_val = [1,2,4,8,16,32,64,128,256,512, 1024, 2048,3500, 4000]#,2,4,8,16,32,128,256,512
G_amma = [1e-4, 1e-3, 1e-2, 1e-1, 1]
for g in G_amma:
    for cv in n_comps_val:
    pca = PCA(n_components = cv)
    X_pca = pca.fit(X).transform(X)
    knn_clf = SVC(C=500, kernel='rbf', gamma=g)
    bush_stratified_cv_results_ft = cross_validate(knn_clf, X_pca, y_bush, 
                                       cv=StratifiedKFold(n_splits = 3, shuffle=True, random_state = 8854), 
                                       scoring=('f1', 'precision', 'recall'), 
                                       return_train_score=False)
    writeTOBushFile(cv, bush_stratified_cv_results_ft)
for cv in n_comps_val:
    pca = PCA(n_components = cv)
    X_pca = pca.fit(X).transform(X)
    knn_clf = SVC(C=500, kernel='rbf', gamma='auto')
    bush_stratified_cv_results_ft = cross_validate(knn_clf, X_pca, y_bush, 
                                       cv=StratifiedKFold(n_splits = 3, shuffle=True, random_state = 8854), 
                                       scoring=('f1', 'precision', 'recall'), 
                                       return_train_score=False)
    writeTOBushFile(cv, bush_stratified_cv_results_ft)

