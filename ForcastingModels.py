import pandas as pd
import matplotlib as plt
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.eval_measures import mse, rmse, meanabs
from AIMAX import ARIMAX

warnings.filterwarnings('ignore')

def met_data_extractor(df, split):
    """
    Reformates Dataframe
    """
    # Creating DataFrame based on number of Years
    year = df.iloc[:,0]

    if  split == "monthly":
        dfMod = df.drop(['year','win', 'spr', 'sum', 'aut', 'ann'], axis = 1)
        index = pd.date_range(start = str(year.min()), end = str(year.max()), freq = 'MS')
    elif split == "quarterly":
        dfMod = df.iloc[:,13:17]
    elif split == "annual":
        index = pd.date_range(start = str(year.min()), end = str(year.max()), freq = 'Y')
        dfMod = df.iloc[:,17]

    dataFrame = pd.DataFrame({'date': index})

    # Create a Series to store data from orginal dataframe
    if split != "annual":
        x = 0
        col = pd.Series()
        while x < len(df):
            rainfallData = pd.Series([ _ for _ in dfMod.iloc[x]])
            col = pd.concat([col, rainfallData], ignore_index = True)
            x += 1
        # Inserting original Data into Create Frame
        dataFrame['rainfall']  = col
    else:
        dataFrame['rainfall'] = dfMod

    dataFrame['date'] = pd.to_datetime(dataFrame['date'])
    dataFrame = dataFrame.set_index('date')

    return dataFrame.dropna()


def traintest_split(df, duration):
    train = df.iloc[:len(df)-duration]
    test  = df.iloc[len(df)-duration:]
    return train, test



if __name__ == '__main__':

    df  = pd.read_csv("Dataset.csv")
    df  = met_data_extractor(df, split = "annual")

    # Current forcasting has been set to a year
    train, test = traintest_split(df, duration = 10);

    # Check if Seasonality Component Exisits
    seasonal = seasonal_decompose(df['rainfall'])
    # seasonal.plot()

    # Obtaining Optimal Model Param.
    seasonal = True
    seasonalPeriod = 12
    arimaxObject = ARIMAX(df, seasonal, seasonalPeriod, train['rainfall'], test)

    # Optimal Model for Annual rainfall
    # order = (0, 1, 1)
    # seasonal order = (0, 0, 1, 12)

    # Obtain prediction AIMAX Model
    #predictions = arimaxObject.seasonalAIMAX((0, 1, 1), (0, 0, 1, 12))

    # Evaluationg Predcitions


    # Reccurent Neural Network
    ninput = 2 # Lenght of Training Object i.e., first two points # should be Around 12
    nfeatures = 1 # Predict the Third

    

    

    # df.describe()
    # df.plot()
    plt.pyplot.show()
