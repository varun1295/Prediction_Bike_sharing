"""
First module nodes
==================
This module contains nodes that loads and visualizes data
on the Bike Sharing data set.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(127)

class data_loading:

    def data_preprocessing(dataset):
        """
        This function is used for pre-processing dataset that is loaded from the csv.
        :param
        dataset: Inputs the raw dataframe from the excel

        :return
        df: returns the pre-processed cleaned df
        """
        df = pd.read_csv(dataset)
        df.head()
        df.describe()
        df.isnull().sum()
        df= df.drop(['instant'], axis=1)
        df['dteday'] = pd.to_datetime(df['dteday'].apply(str) + ' ' + df['hr'].apply(str) + ':00:00')
        return df

    def data_visualization(df):
        """
        This function is used for visualizing the preprocessed data and doing exploratory data analysis.

        :param: preprocessed pandas dataframe

        :return: plots of the variables.
        """

        # Visualizing the target variable
        plt.figure(figsize=(14, 10))
        plt.title("Count of bike sharing according to dates")
        plt.plot(df['dteday'], df['cnt'])
        #plt.show()
        plt.savefig("Raw data visualization.png")

        # box plot for visualizing outliers
        fig=px.box(df, y="cnt", notched=True,title='Box plot of the count variable')
        #fig.show()
        plt.savefig("Box Plot.png")

        # point plot for hourly utilization
        for column in ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']:
            hist = px.histogram(df, x=column, y='cnt')
            hist.show()
            plt.savefig("Histogram plots for each column.png")
        sns.pointplot(x=df['hr'], y='cnt', data=df);
        plt.title("Hourly Utilization")
        plt.ylabel("Bike Shares", fontsize=12)
        plt.xlabel("Hour", fontsize=12)
        plt.savefig("Hourly Utilization point plot.png", dpi=300, bbox_inches='tight')

        # line plot for hourly utilization
        for c in ['holiday','season','workingday']:
            sns.lineplot(data=df,x='hr',y='cnt',hue=c)
            plt.title('Hourly plot vs count')
            plt.savefig("Hour vs count plot_main features.png",dpi=300, bbox_inches='tight')

        # point plots for humidity vs count
        sns.pointplot(x='hum', y='cnt', data=df)
        plt.title("Amount of bike shares vs humidity", fontsize=25)
        plt.xlabel("Humidity (%)", fontsize=20)
        plt.ylabel('count of bike shares', fontsize=20)
        plt.locator_params(axis='x', nbins=10)
        plt.savefig("Pointplot of humidity vs count.png",dpi=300, bbox_inches='tight')

        # box plots of whole df
        bx=px.box(df, y="cnt")
        bx.show()

        # feature correlation plot
        corrs = abs(df.corr())
        sns.heatmap(corrs, annot=True)
        plt.title("Feature Correlation")
        plt.savefig("Feature_correlation.png", dpi=300, bbox_inches='tight')
        return plt

