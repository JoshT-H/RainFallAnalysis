import pandas as pd
import matplotlib as plt
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.eval_measures import mse, rmse, meanabs


class ARIMAX:
    def __init__(self, df, seasonal, period, train, test):
        self.seasonal = seasonal
        self.period  = period
        self.df = df
        self.train = train
        self.test  = test

    def autoAIMAX(self):
        print(auto_arima(self.df, seasonal = self.seasonal, m = self.period).summary())

    def seasonalAIMAX(self, order, seasonal_order):
        model = SARIMAX(self.train, order = order, seasonal_order =seasonal_order, enfore_invertibility = True)
        results = model.fit() ##############################################
        start = len(self.train)
        end   =  len(self.train) + len(self.test) - 1
        predicition = results.predict(start, end).rename('SARIMA Model')
        return predictions

    def mse_aima(self):
        print(f'The ARIMAX MSE is {meanabs(predictions, test)}')

    def aima_plots(self, prediction):
        self.train.plot()
        predictions.plot(legend = True)
        self.test.plot(legend = True)


if __name__ =='__main__':
    pass
