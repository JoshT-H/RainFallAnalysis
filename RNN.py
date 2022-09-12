from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class RNN:
    def __init__(self, train, test, ninputs = 12, nfeatures =  1, bs = 1, neurons = 150, activation = 'relu', loss = 'mse', optimizer = 'adam', epochs = 25):
        self.ninputs = ninputs
        self.nfeatures = nfeatures
        self.bs = bs
        self.neurons = neurons
        self.activation  = activation
        self.loss = loss
        self.optimizer = optimizer
        self.epochs =epochs
        self.train = train
        self.test = test

    def RNNmodel(self):
        # Data Scaling for Neural Network
        scaler = MinMaxScaler()
        scaler.fit(self.train)
        scaled_train = scaler.transform(self.train)
        scaled_test = scaler.transform(self.test)


        train_generator = TimeseriesGenerator(scaled_train, scaled_train, length = self.ninputs, batch_size =self.bs)
        model = Sequential()
        model.add(LSTM(self.neurons, activation = self.activation, input_shape=(self.ninputs, self.nfeatures)))
        model.add(Dense(1))
        model.compile(optimizer = self.optimizer, loss = self.loss)

        print(model.summary)


        model.fit_generator(train_generator, epochs = self.epochs)

        # Plot Model Loss
        loss_dic = model.history.history.keys()
        my_loss = model.history.history['loss']
        plt.pyplot.plot(range(len(my_loss)), my_loss)


        test_prediction = []
        first_eval_batch = scaled_train[-self.ninputs:]
        current_batch = first_eval_batch.reshape((1, self.ninputs, self.nfeatures))

        for i in range(len(self.test)):

            #  Current prediction into the future
            current_pred = model.predict(current_batch)[0]
            # Store predictions
            test_prediction.append(current_pred)
            # NP indexing Drops first point and Appends new prediction
            current_batch = np.append(current_batch[:,1:,:], [[current_pred]], axis = 1)


        true_prediction = scaler.inverse_transform(test_prediction)
        self.test['Predictions'] = true_prediction
        self.test.plot()
        self.train.plot()
        plt.pyplot.show()

if __name__ =='__main__':
    pass
