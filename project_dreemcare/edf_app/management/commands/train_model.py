'''

This script allows you to train the LSTM Network with the
EDF file of your choice.

The new LSTM model replaces the previous one in media/model/,
and will be the one used at the next file upload.

The trainscore and testscore (RMSE) are also automatically
saved and updated, as you can see in the "Get Predictions"
section.

'''
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from django.core.management.base import BaseCommand
import logging
from project_dreemcare.settings import PROJECT_ROOT
from edf_app.models import Document
from edf_app.edf_reader import EDFReader
from edf_app.prediction import (
    update_prediction,
    split_norm_dataset,
    truncate_EEG
)
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
        look_back = 1

        # check if file_id is valid
        try:
            doc = Document.objects.get(pk=file_id)
        except:
            logging.error(" File id does not exist.")
            return

        # load the dataset
        edf = EDFReader(PROJECT_ROOT + doc.docfile.url)
        EEG_signal = edf.get_signal(0)
        EEG_signal = truncate_EEG(EEG_signal)

        # load and format dataset
        dataset = split_norm_dataset(EEG_signal, look_back)
        trainX, trainY = dataset['train']
        testX, testY = dataset['test']

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

        # update predictions
        print("\n\nUpdating Scores...\n")
        update_prediction()

        print("Success!")
