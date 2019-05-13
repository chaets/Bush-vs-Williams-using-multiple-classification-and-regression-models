import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate,StratifiedKFold
from sklearn.svm import SVC
import pickle
from sklearn.decomposition import PCA


X = pd.read_csv("X.csv",sep = ' ', header=None,dtype=float)
X = X.values


n_neighbors=[1,3,5]
# C_set = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5, 1e+6, 1e+7]
C_set = [1,10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500,600, 700, 800, 900, 1000, 1250, 1500, 1750, 2000]
deg = [0,1,2,3,4,5,6]
G_amma = [1e-4, 1e-3, 1e-2, 1e-1, 1]


y = pd.read_csv("y_bush_vs_others.csv",sep = ' ', header=None,dtype=float)
y_bush = y.values.ravel()

n_comps_val = [1,2,4,8,16,32,64,128,256,512, 1024, 2048, 4000]
print("shape", X.shape)
np.sum(y_bush)
print('bush shape', y_bush.shape)





# for cv in n_comps_val:
#     for i in n_neighbors:
#         pca = PCA(n_components=cv)
#         X_pca = pca.fit(X).transform(X)
#         knn = KNeighborsClassifier(i)

#         stratified_cv_results_K = cross_validate(knn, X_pca, y_bush,
#                                                  cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=8854),
#                                                  scoring=('precision', 'recall', 'f1'), return_train_score=False)

#         print('KNN Results: ', stratified_cv_results_K)

# for cv in n_comps_val:
#     for CS in C_set:
#         pca = PCA(n_components=cv)
#         X_pca = pca.fit(X).transform(X)
#         print('For C value in Linear', CS)
#         svc = SVC(C=CS, kernel='linear', degree=3, gamma='auto')
#         stratified_cv_results_L = cross_validate(svc, X_pca, y_bush,
#                                                  cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=8854),
#                                                  scoring=('precision', 'recall', 'f1'), return_train_score=False)
#         print('Variable C in Linear', stratified_cv_results_L)



# for cv in n_comps_val:

#     for CS in C_set:
#         pca = PCA(n_components=cv)
#         X_pca = pca.fit(X).transform(X)
#         print('For C value in Poly', CS)
#         for d in deg:
#             svc = SVC(C=CS, kernel='poly', degree=d, gamma='auto')
#             print('For C value in Poly with degree: ', d)
#             stratified_cv_results_PD = cross_validate(svc, X_pca, y_bush,
#                                                       cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=8854),
#                                                       scoring=('precision', 'recall', 'f1'), return_train_score=False)
#             print('Variable C in Poly', stratified_cv_results_PD)


for cv in n_comps_val:
    pca = PCA(n_components=cv)
    X_pca = pca.fit(X).transform(X)
    for CS in C_set:
        print('For C value in rbf', CS)
        for G in G_amma:
            svc = SVC(C=CS, kernel='rbf', gamma=G)
            print('For C value in rgf with Gamma: ', G)
            stratified_cv_results_RG = cross_validate(svc, X_pca, y_bush,
                                                      cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=8854),
                                                      scoring=('precision', 'recall', 'f1'), return_train_score=False)
            print('Variable C in RBF', stratified_cv_results_RG)

    svc = SVC(C=CS, kernel='rbf', gamma='auto')
    print('For C value in rgf with Gamma: ', G)
    stratified_cv_results_RGA = cross_validate(svc, X_pca, y_bush,
                                           cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=8854),
                                           scoring=('precision', 'recall', 'f1'), return_train_score=False)
    print('Variable C in RBF', stratified_cv_results_RGA)

# print("PICKLING ...")
# # pickle.dump((bush),open('BUSH.pkl','wb'))
# pickle.dump((stratified_cv_results_K, stratified_cv_results_L, stratified_cv_results_PD, stratified_cv_results_RG, stratified_cv_results_RGA),open('BUSH.pkl','wb'))