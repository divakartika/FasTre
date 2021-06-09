# FasTre: Machine Learning Repository

Dataset source: https://www.kaggle.com/weirdolucifer/medical-facility-operational-data

Preprocessed datasets:
* WaitDataF1.csv
* WaitDataF2.csv
* WaitDataF3.csv

Preprocessing script: waitdata.ipnyb

Training script: wait_estimator.py

Saved model:
* saved_model/my_model (TensorFlow SavedModel)
* wait_est_model.h5 (HDF5)

_Note: the HDF5 probably won't work because of the normalization layer_
