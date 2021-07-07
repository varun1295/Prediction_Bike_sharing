"""
Second module nodes
==================
This module contains nodes that performs data modelling using 2 different ML / DL algorithms and MLFLOW
on the preprocessed bike sharing data set from previous file.
"""

from B1 import data_loading
import math
import pandas as pd
import os
# LSTM
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
tf.random.set_seed(1234)
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import mlflow
import mlflow.pyfunc
import mlflow.keras
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(37)
import random as rn
from urllib.parse import urlparse
rn.seed(125)
os.environ['PYTHONHASHSEED'] = '0'



class data_modeling:

    def eval_metrics(actual, pred):
        """
        This function evaluates the model based on actual and predicted data using the scoring metrics

        :param
        actual: Input the Y values of actual dataframe
        pred: Inputs the Y values of predicted dataframe after fitting the model

        :return
        rmse: Integer value of Root mean squared error for the model
        mae: Integer value of mean absolute error for the model
        r2: R2 score explaining the model variability.
        """

        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2


    def model1(cldf):

        """
        This function is used for modeling the data using LSTM after train-test split of the data. Also, it uses mlflow UI tracking
        to view the all of the plots, metrics,artifacts in mlflow API which is responsible incase of deploying the code.

        :param
        cldf: cleaned and preprocessed dataframe
        :return
        model: returns the LSTM model
        """
        cldf=cldf.set_index('dteday')
        # Splitting the dataset in to 80% train and 20% test
        train_data = math.ceil(len(cldf) * .80)
        test_data = len(cldf) - train_data
        time_steps = 24
        train, test = cldf.iloc[0:train_data], cldf.iloc[(train_data - time_steps):len(cldf)]
        print(cldf.shape, train.shape, test.shape)

        # Scaling temperature,humidity and windspeed
        train_toscale = train[['temp', 'atemp', 'hum', 'windspeed']].to_numpy()
        test_toscale = test[['temp', 'atemp', 'hum', 'windspeed']].to_numpy()

        # Scale using Robustscaler
        scaler = RobustScaler()
        train.loc[:, ['temp', 'atemp', 'hum', 'windspeed']] = scaler.fit_transform(train_toscale)
        test.loc[:, ['temp', 'atemp', 'hum', 'windspeed']] = scaler.fit_transform(test_toscale)

        # Scale the values of the column 'cnt'
        train['cnt'] = scaler.fit_transform(train[['cnt']])
        test['cnt'] = scaler.fit_transform(test[['cnt']])
        x_train = []
        y_train = []
        x_test = []
        y_test = cldf.loc[:, 'cnt'].iloc[train_data:len(cldf)]
        for i in range(len(train) - time_steps):
            x_train.append(train.drop(columns='cnt').iloc[i:i + time_steps].to_numpy())
            y_train.append(train.loc[:, 'cnt'].iloc[i + time_steps])
        for i in range(len(test) - time_steps):
            x_test.append(test.drop(columns='cnt').iloc[i:i + time_steps].to_numpy())
        x_train=np.array(x_train)
        y_train=np.array(y_train)
        x_test=np.array(x_test)
        y_test=np.array(y_test)
        print('Train size:')
        print(x_train.shape, y_train.shape)
        print('Test size:')
        print(x_test.shape, y_test.shape)

        # "MLflow" is a deployment API used for end to end ML Projects.
        with mlflow.start_run():
            model = Sequential()
            model.add(Bidirectional(LSTM(128, input_shape=(x_train.shape[1], x_train.shape[2]))))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))

            # model.summary()
            model.compile(optimizer="adam", loss="mse")
            es = EarlyStopping(monitor='val_loss', mode='min', patience=10,verbose=1)
            history = model.fit(x_train, y_train, epochs=100, batch_size=24, validation_split=0.1,shuffle=True,callbacks=[es]) #callbacks=[es
            y_pred= model.predict(x_test)
            train_pred=model.predict(x_train)
            (rmse1, mae1, r21) = data_modeling.eval_metrics(train_pred, y_train)
            print('Training Mean Absolute Error', mae1)
            print('Training rmse', rmse1)
            print('Training r21', r21)

            # Reverse scaling
            y_pred = scaler.inverse_transform(y_pred)
            print('predicted values:',y_pred)

            # mlflow evaluating and logging metrics in UI
            (rmse, mae, r2) = data_modeling.eval_metrics(y_test,y_pred)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)
            params={'epochs':50, 'batch_size':24,'validation_split':0.1,'optimizer':'adam','loss':'mse'}
            mlflow.log_params(params)
            print('RMSE Value:',rmse)
            print('Mean Absolute Error',mae)
            print('R2_score',r2)
            plt.figure(figsize=(16, 8))
            plt.plot(y_test[1200:1500], label='actual')
            plt.plot(y_pred[1200:1500], label='predicted')
            plt.legend()
            plt.savefig("LSTM Predictions with actual vs predicted.png")
            mlflow.set_tag("LSTM model", "Experiment")
            mlflow.keras.log_model(model, "LSTM") #logging the model in mlflow UI
            mlflow.end_run()

            # mlflow tracking
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(model, "LSTM", registered_model_name="LSTM Prediction")
            else:
                mlflow.keras.log_model(model, "LSTM")

        return model











"""The main code starts here"""

dl = data_loading
cleandf = dl.data_preprocessing(dataset=r'C:\Users\vernn\hour.csv')
dl.data_visualization(cleandf)
if __name__ == '__main__':
    dm = data_modeling
    dm.model1(cleandf)

