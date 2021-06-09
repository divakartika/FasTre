# -*- coding: utf-8 -*-
"""
Created on Sat May 22 23:48:33 2021

@author: Diva
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers import Dense

print(tf.__version__)

#%%

df1 = pd.read_csv('WaitDataF1.csv')
df2 = pd.read_csv('WaitDataF2.csv')
df3 = pd.read_csv('WaitDataF3.csv')
df = pd.concat([df1,df2,df3])
df1.loc[df1["Wait"] < 0, "Wait"] = 0

X = df1.drop(['Wait','ThoracicCount','PediatricCount','NeuroCount','CardiacCount'],axis=1)
y = df1['Wait']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/10,random_state=len(X))
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=1/9,random_state=len(X_train))

#%%

normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(X_train))

#%%

model = keras.Sequential([
    normalizer,
    Dense(16, activation='relu'
          ),
    Dense(64, activation='relu'
          ),
    Dense(64, activation='relu'
          ),
    Dense(1, activation='relu')
    ])

#%%

model.compile(optimizer='adam', 
              loss='mae'
              )
print('\n')
model.summary()

#%%

history = model.fit(x=X_train,
                    y=y_train.values,
                    validation_data=(X_val,y_val.values),
                    batch_size=342,
                    epochs=200
                    )

history
#%%

losses = pd.DataFrame(model.history.history)
losses.plot()

#%%


predictions = model.predict(X_val)
mae = mean_absolute_error(y_val,predictions)

rmse = np.sqrt(mean_squared_error(y_val,predictions))

print('MAE =', mae)
print('RMSE =', rmse)

evs = explained_variance_score(y_val,predictions)
print('EVS =',evs)

predictions = model.predict(X_val.head(1))

print('prediction =',predictions)
print('label =', y_val.values[0])

#PREDICT WITH TEST SET
predictions = model.predict(X_test.head(1))

print('test set prediction =',predictions)
print('test set label =', y_test.values[0])

#%%

model.save('wait_est_model.h5')

#%%
model.save('saved_model/my_model')