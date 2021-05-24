# -*- coding: utf-8 -*-
"""
Created on Sat May 22 23:48:33 2021

@author: Diva
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

print(tf.__version__)

#%%


df = pd.read_csv('WaitDataF1.csv')
df = df.sort_index(axis=0,ascending=False,ignore_index=True)

X = df.drop('Wait',
            axis=1)
y = df['Wait']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=3/10)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=2/9)

#%%
#scaler = MinMaxScaler()
#X_train= scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

#%%
model = keras.Sequential([
    Dense(32, activation='relu'),
    Dense(1)
    ])
#%%
model.compile(optimizer='adam', 
              loss='mse',
              metrics=['accuracy','mae', 'mse'])
#%%

#early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

history = model.fit(x=X_train,
                    y=y_train.values,
                    validation_data=(X_val,y_val.values),
                    #batch_size=128,
                    epochs=400
                    #callbacks=[early_stop]
                    )

history
#%%

losses = pd.DataFrame(model.history.history)
losses.plot()

#%%

#PREDICT WITH TEST SET
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test,predictions)

rmse = np.sqrt(mean_squared_error(y_test,predictions))

print('MAE =', mae)
print('RMSE =', rmse)

#%%

model.save('wait_est_model.h5')