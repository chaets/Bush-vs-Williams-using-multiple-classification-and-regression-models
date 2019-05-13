
# coding: utf-8

# In[79]:


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


# In[81]:


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self._data = []
        
        self.cnt = 0

    def on_epoch_end(self, epoch, logs={}):
#         print(self.validation_data)
#         val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
#         val_targ = self.validation_data[1]
#         print(val_predict)
#         print('')
#         print(val_targ)
#         _val_f1 = f1_score(np.argmax(val_targ, axis=1),  np.argmax(val_predict, axis=1))
#         _val_recall = recall_score(np.argmax(val_targ, axis=1),  np.argmax(val_predict, axis=1))
#         _val_precision = precision_score(np.argmax(val_targ, axis=1),  np.argmax(val_predict, axis=1))
#         self.val_f1s.append(_val_f1)
#         self.val_recalls.append(_val_recall)
#         self.val_precisions.append(_val_precision)
        
        X_val, y_val = self.validation_data[0], self.validation_data[1]
        y_predict = np.asarray(model.predict(X_val))

        y_val = np.argmax(y_val, axis=1)
        y_predict = np.argmax(y_predict, axis=1)

        self._data.append({
            'val_recall': recall_score(y_val, y_predict),
            'val_precision': precision_score(y_val, y_predict),
            'val_fi_score': f1_score(y_val, y_predict)
        }) 
        
        print (" — val_f1: '%f' — val_precision: '%f' — val_recall '%f'" %(self._data[self.cnt]['val_fi_score'], 
                                                                           self._data[self.cnt]['val_precision'], 
                                                                           self._data[self.cnt]['val_recall']))
        self.cnt += 1

    def get_data(self):
        return self._data

metrics = Metrics()


# In[82]:


X = pd.read_csv("D:\MCS\Sem 3\Machine Learning\Codes\project\Project\Phase_3\X.csv",sep = ' ', header=None,dtype=float)
X = X.values

y = pd.read_csv("D:\MCS\Sem 3\Machine Learning\Codes\project\Project\Phase_3\y_williams_vs_others.csv",sep = ' ', header=None,dtype=float)
y_bush = y.values.ravel()


# In[83]:


print("X shape", X.shape)
print('Y bush shape', y_bush.shape)


# In[84]:


batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28


# In[85]:


x_train, x_test, y_train, y_test = train_test_split(X, y_bush, test_size=1. / 3, random_state=8854)


# In[86]:


x_train = x_train.reshape(-1,64,64,1)
x_test = x_test.reshape(-1,64,64,1)


# In[87]:


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# In[88]:


## convert class vectors to binary class matrices


# In[89]:


y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[90]:



model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64,64,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


# In[91]:


model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])


# In[ ]:


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[metrics])


# In[74]:


# model.fit(x_train[0:56,], y_train[0:56,:], batch_size=28, epochs=epochs, verbose=1, validation_data=(x_test[0:2], y_test[0:2]), callbacks=[metrics])


# In[76]:


metrics._data


# In[52]:


np.argmax(np.array([[1,0,0,0], [0,0,1,0]]), axis=1)


# In[64]:


recall_score([1,1], [1,1], average='binary')


# In[80]:


score = model.evaluate(x_test, y_test, verbose=1)


# In[81]:


print('Test loss:', score[0])
print('Test accuracy:', score[1])

