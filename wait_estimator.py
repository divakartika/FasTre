# -*- coding: utf-8 -*-
"""
Created on Sat May 22 23:48:33 2021

@author: Diva
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

print(tf.__version__)

#%%


df1 = pd.read_csv('WaitDataF1.csv')
df2 = pd.read_csv('WaitDataF2.csv')
df3 = pd.read_csv('WaitDataF3.csv')
df = pd.concat([df1,df2,df3])
df.loc[df["Wait"] < 0, "Wait"] = 0
#df.drop(df[df['Wait'] <= -180].index, inplace=True)
df.drop(df[df['Wait'] >= 180].index, inplace=True)
#df = df.sort_index(axis=0,ascending=False,ignore_index=True)

X = df.drop('Wait',
            axis=1)
y = df['Wait']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/10,random_state=len(X))
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=1/9,random_state=len(X_train))

#%%
normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(X_train))

#%%
#scaler = MinMaxScaler()
#X_train= scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

#%%
model = keras.Sequential([
    normalizer,
    #tf.keras.layers.Conv1D(filters=32, kernel_size=3,
                      #strides=1, padding="causal",
                      #activation="relu",
                      #input_shape=[None, 1]
                      #),
    #LSTM(64, return_sequences=True),
  #tf.keras.layers.LSTM(64, return_sequences=True),
    # Dense(16, activation='relu', 
           # kernel_regularizer=regularizers.l2(0.001)
           # ),
    Dense(16, activation='relu', 
          kernel_regularizer=regularizers.l2(0.001)
          ),
    #Dropout(0.1),
    Dense(32, activation='relu', 
          kernel_regularizer=regularizers.l2(0.001)
          ),
    #Dropout(0.1),
    Dense(64, activation='relu', 
          kernel_regularizer=regularizers.l2(0.001)
          ),
    #Dropout(0.1),
    Dense(64, activation='relu', 
          kernel_regularizer=regularizers.l2(0.001)
          ),
    #Dropout(0.1),
    # Dense(64, activation='relu', 
           # kernel_regularizer=regularizers.l2(0.001)
           # ),
    #Dropout(0.1),
    #Dense(512, activation='relu', 
          #kernel_regularizer=regularizers.l2(0.001)
     #      ),
    #Dropout(0.1),
    #Dense(512, activation='relu'),
    #Dropout(0.1),
    Dense(1, activation='relu')
    ])
#%%
model.compile(optimizer='adam', 
              loss='mae',
              #metrics=['accuracy']
              )

model.summary()
#%%

#early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

history = model.fit(x=X_train,
                    y=y_train.values,
                    validation_data=(X_val,y_val.values),
                    batch_size=655,
                    epochs=600
                    #callbacks=[early_stop]
                    )

history
#%%

losses = pd.DataFrame(model.history.history)
losses.plot()

#%%

#PREDICT WITH TEST SET
predictions = model.predict(X_val)
mae = mean_absolute_error(y_val,predictions)

rmse = np.sqrt(mean_squared_error(y_val,predictions))

print('MAE =', mae)
print('RMSE =', rmse)

evs = explained_variance_score(y_val,predictions)
print('EVS =',evs)

predictions = model.predict(X_val.head(5))
#mae = mean_absolute_error(y_test[0],predictions)

#rmse = np.sqrt(mean_squared_error(y_test[0],predictions))

print('prediction =',predictions)
print('label =', y_val.values[:5])
#print('MAE =', mae)
#print('RMSE =', rmse)

#%%

#model.save('wait_est_model.h5')