"""
Third module nodes
==================
This module contains nodes that performs data modelling using XBG Regression algorithm and MLFLOW
on the preprocessed bike sharing data set.
"""

from B1 import data_loading
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
import os
os.environ['PYTHONHASHSEED'] = '0'
import warnings
warnings.filterwarnings("ignore")
np.random.seed(127)


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

    def model2(cldf):
        """
        This function includes the second model i.e XGBR model to compare the performance and the metrics.lso, it uses mlflow UI tracking
        to view the all of the plots, metrics,artifacts in mlflow API which is responsible incase of deploying the code.
        :param
        cldf: cleaned and preprocessed dataframe

        :return
        xgbmodel: returns the XGBR model
        """
        cldf=cldf.set_index('dteday')
        x_data = cldf.drop('cnt', axis=1)
        y_data = cldf['cnt']
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)
        x_train.shape, x_test.shape, y_train.shape, y_test.shape
        with mlflow.start_run():
            xgbmodel = XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1,
                                    max_depth=5, alpha=10, n_estimators=20)
            xgbmodel.fit(x_train, y_train)
            pred = xgbmodel.predict(x_test)
            resdf = y_test.to_frame()
            resdf['pred_Cnt'] = pred.tolist()
            print('Result', resdf)
            (rmse, mae, r2) = data_modeling.eval_metrics(y_test, pred)
            print('RMSE Value:', rmse)
            print('Mean Absolute Error', mae)
            print('R2_score', r2)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)
            params = {'colsample_bytree': 0.3, 'learning_rate': 0.1, 'max_depth': 5, 'alpha': 10, 'n_estimators': 20}
            mlflow.log_params(params)
            resdf.plot(y=['cnt', 'pred_Cnt'], grid=True, figsize=[18.0, 8.0], label=["actual", "prediction"],
                       color=['b', 'r'])
            plt.ylabel(ylabel='Count of Bikes shares', fontsize=14)
            plt.title('Bike share Prediction vs Actual', fontsize=22)
            plt.savefig("XGB Model Pred vs actual cnt.png")
            mlflow.set_tag("XGBoost", "Experiment")
            mlflow.sklearn.log_model(xgbmodel, "XGBoost")
            mlflow.end_run()
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(xgbmodel, "xgbmodel", registered_model_name="XGBoost")
            else:
                mlflow.sklearn.log_model(xgbmodel, "xgbmodel")

        return xgbmodel





'''Program starts here'''
dl = data_loading
cleandf = dl.data_preprocessing(dataset=r'C:\Users\vernn\hour.csv')
dl.data_visualization(df=cleandf)
if __name__ == '__main__':
    dm = data_modeling
    dm.model2(cleandf)