from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler



class RNN():
    "Recurent Neural Network Build using Keras Lib"

    def __init__(self,train, ninputs = 12 , nfeatures = 1 ):
        self.ninputs = ninputs
        self.nfeatures = nfeatures # Basically Number of Columsn ???
        self.train = train

    def scale_data(self):
        # Data Scaling for Neural Network
        scaler = MinMaxScaler()
        scaler.fit(self.train)
        scaled_train = scaler.transform(train)
        scaled_test = scaler.transform(test)
        return  scaled_train, scaled_test


    def rnn_model(self, scaled_train, bs = 1, neurons = 150, activation = 'relu', loss = 'mse', optimizer = 'adam', epochs = 25):
        generator = TimeseriesGenerator(scaled_train, scaled_train, length = self.ninputs, batch_size =bs)
        model = Sequential()
        model.add(LTSM(neurons, activation = activation, input_shape=(self.ninputs, self.nfeatures)))
        model.add(Dense(1))
        model.compile(optimizer = optimizer, lose = lose)
        print(model.summary)
        model.fit_generator(train_generator, epochs = ephochs)
        return model

    def plot_lose(self, model):
        loss_dic = model.history.history,keys()
        plt.plot(range(len(loss_dic['loss'])), loss_dic['loss'])

    def test_eval(self, scaled_train, forcast_length):
        test_prediction = []
        first_eval_batch = scaled_train[-self.ninputs:]
        current_batch = first_eval_batch.reshape((1,self.ninputs, self.nfeatures))

        for i in range(len(forcast_length)):
            #  Current prediction into the future
            current_batch = model.predict(current_pred)[0]
            # Store predictions
            test_prediction.append(current_batch)
            # NP indexing Drops first point and Appends new prediction
            current_batch = np.append(current_batch[:,1:,:], [[current_pred]], axis = 1)

        test_prediction = scaler.inverse_transform(test_prediction)



        return test_prediction
