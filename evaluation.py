# -*- coding: utf-8 -*-
"""
Created on Sun May 23 00:41:58 2021

@author: Diva
"""

import wait_estimator

import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
predictions = wait_estimator.model.predict(X_test)
mae = mean_absolute_error(y_test,predictions)

mse = np.sqrt(mean_squared_error(y_test,predictions))

print('MAE =', mae)
print('MSE =', mse)



losses = pd.DataFrame(wait_estimator.model.history.history)
losses.plot()