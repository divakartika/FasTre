# -*- coding: utf-8 -*-
"""
Created on Sat May 22 23:48:33 2021

@author: Diva
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

print(tf.__version__)

X = df.drop(#label,
            axis=1)
y = df[#label]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)


scaler = MinMaxScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential([
    Dense(#20, activation='relu', input_shape=[len(train_features[0])]),
    Dense(1)
    ])

model.compile(optimizer='adam', 
              loss='mse',
              metrics=['accuracy','mae', 'mse'])


#early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

model.fit(x=X_train,
          y=y_train.values,
          validation_data=(X_test,y_test.values),
          #batch_size=128,
          epochs=400,
          callbacks=[early_stop]
          )

model.save('wait_est_model.h5')