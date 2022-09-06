import pandas as pd
import matplotlib as plt
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings('ignore')


def met_data_extractor(df, split):
    """
    DataExtractor for Met Office Dataset: Monthly vs. Quarterly vs. Annual
    """
    if  split == "monthly":
        dfMod = df.drop(['year','win', 'spr', 'sum', 'aut', 'ann'], axis = 1)
        x = 0
        year = df.iloc[:,0]
        # Create a Datatime Range for Index
        index = pd.date_range(start = str(year.min()), end = str(year.max()), freq = 'MS')
        dataFrame = pd.DataFrame({'date': index})
        # Create a Series to Iterate through Data
        col = pd.Series()

        while x < len(df):
            monthlyData = pd.Series([ _ for _ in dfMod.iloc[x]])
            col = pd.concat([col, monthlyData], ignore_index = True)
            x += 1
        dataFrame['rainfall']  = col
        dataFrame['date'] = pd.to_datetime(dataFrame['date'])
        dataFrame = dataFrame.set_index('date')
    elif split == "quarterly":
        pass
    elif split == "Annual":
        pass

    return dataFrame.dropna()

def AutoAIMAX(df, seasonal, m):
    return print(auto_arima(df, seasonal = seasonal, m = m).summary())



def traintest_split(df, months):

    train = df.iloc[:len(df)-months]
    test  = df.iloc[len(df)-months:]
    return train, test

if __name__ == '__main__':

    df  = pd.read_csv("Dataset.csv")
    df  = met_data_extractor(df, split = "monthly")

    # Current forcasting has been set to a year
    train, test = traintest_split(df, months = 12);

    # Check if Seasonality Component Exisits
    seasonal = seasonal_decompose(df['rainfall'])
    # seasonal.plot()

    # Obtaining Optimal Model Param.
    AutoAIMAX(df, seasonal = True, m = 15)

    #df.describe()

    #model = SARIMAX(train['rainfall'], order = (2,1,0), seasonal_order = (2,0,1,4), enfore_invertibility = True)
    # result.fit()
    # start = len(train)
    #end   =  len(train) + len(test) -1
    #prediction = results.predict(start, end).rename('SARIMA Model')




    #df.plot()
    #plt.pyplot.show()
