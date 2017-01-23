'''

This script allows you to train the LSTM Network with the
EDF file of your choice.

The new LSTM model replaces the previous one in media/model/,
and will be the one used at the next file upload.

The trainscore and testscore (RMSE) are also automatically
saved and updated, as you can see in the "Get Predictions"
section.

'''
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from django.core.management.base import BaseCommand
import logging
from project_dreemcare.settings import PROJECT_ROOT
from edf_app.models import Document
from edf_app.edf_reader import *
from edf_app.prediction import create_dataset, update_prediction
# import ipdb


class Command(BaseCommand):

    """ Command to run an algorithm. """

    help = """ Command to run a task. """

    def add_arguments(self, parser):
        """ Parser optional and compulsory arguments. """
        parser.add_argument('--file-id', '-f', type=int, required=True,
                            help="id of the EDF file")
        parser.add_argument('--nb-epoch', '-n', type=int, default=100)

    def handle(self, *args, **kwargs):

        # parameters
        file_id = kwargs['file_id']
        nb_epoch = kwargs['nb_epoch']

        try:
            doc = Document.objects.get(pk=file_id)
        except:
            logging.error(" File id does not exist.")
            return

        # load the dataset
        edf = EDFReader(PROJECT_ROOT + doc.docfile.url)
        EEG_data = edf.get_signal(0)
        dataset = np.array(EEG_data).astype('float32')
        dataset = dataset.reshape(dataset.shape[0], 1)

        # normalize the datasets
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)

        # split into train and test sets
        train_size = int(len(dataset) * 0.67)
        train = dataset[0:train_size, :]
        test = dataset[train_size:len(dataset), :]
        print("(train, test) = ({}, {})\n".format(len(train), len(test)))

        # reshape into X=t and Y=t+1
        look_back = 1
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        # reshape input to be [samples, time steps, features]
        trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1]))
        testX = testX.reshape((testX.shape[0], 1, testX.shape[1]))

        # create and fit the LSTM network
        model = Sequential()
        model.add(LSTM(4, input_dim=look_back))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, nb_epoch=nb_epoch,
                  batch_size=1, verbose=1)
        path = PROJECT_ROOT + '/media/model/lstm_lookback' + \
                              str(look_back) + '.h5'
        model.save(path)

        print("\n\nUpdating Scores...\n")
        # update predictions
        update_prediction()

        print("Success!")
