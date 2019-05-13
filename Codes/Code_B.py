import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate,StratifiedKFold
from sklearn.svm import SVC
import pickle
from sklearn.decomposition import PCA

X = pd.read_csv("X.csv",sep = ' ', header=None,dtype=float)
X = X.values

y = pd.read_csv("y_bush_vs_others.csv",sep = ' ', header=None,dtype=float)
y_bush = y.values.ravel()
print("shape", X.shape)
X_bush= X.reshape(64,64)
print(X_bush)