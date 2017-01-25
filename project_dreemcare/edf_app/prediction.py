import numpy as np
import math
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from django.utils import timezone
import logging
from project_dreemcare.settings import PROJECT_ROOT
from edf_app.models import Document, Prediction
from edf_app.edf_reader import EDFReader

'''
We need to use global variable here because otherwise
the second call on load module will make tensorflow crash
and our predictions won't be computed. I couldn't find any
related issue on Github, but it seems to be a bug in tensorflow.
'''

KERAS_MODEL = None


def compute_rmse(EEG_signal, look_back=1):
    '''
    This method perform a non-exhaustive cross-validation,
    evaluating train and test performance by computing
    the Root Mean Square Error.
    '''
    global KERAS_MODEL

    # load and format dataset
    dataset = split_norm_dataset(EEG_signal, look_back)
    trainX, trainY = dataset['train']
    testX, testY = dataset['test']
    scaler = dataset['scaler']

    # load model
    path = PROJECT_ROOT + '/media/model/lstm_lookback' + str(look_back) + '.h5'
    if KERAS_MODEL is None:
        try:
            KERAS_MODEL = load_model(path)
        except Exception as e:
            msg = 'Error loading predictor in ' + path + ': ' + str(e)
            logging.warning(msg)

    # make predictions
    trainPredict = KERAS_MODEL.predict(trainX)
    testPredict = KERAS_MODEL.predict(testX)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))

    return trainScore, testScore


def split_norm_dataset(EEG_signal, look_back):

    # load the dataset
    dataset = np.array(EEG_signal) \
                .reshape(len(EEG_signal), 1) \
                .astype('float32')

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    train = dataset[0:train_size, :]
    test = dataset[train_size:len(dataset), :]
    print("(train, test) = ({}, {})\n".format(len(train), len(test)))

    def create_dataset(dataset, look_back):
        # reshape into X=t and Y=t+1
        dataX, dataY = [], []
        for i in range(0, len(dataset) - look_back):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])

        # reshape input to be [samples, time steps, features]
        dataX = np.array(dataX).reshape((len(dataX), 1, len(dataX[0])))
        dataY = np.array(dataY)
        return dataX, dataY

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    return {"dataset": dataset,
            "train": (trainX, trainY),
            "test": (testX, testY),
            "scaler": scaler
            }


def update_prediction():
    docs = Document.objects.all()
    for doc in docs:
        # load the dataset
        edf = EDFReader(PROJECT_ROOT + doc.docfile.url)
        EEG_data = edf.get_signal(0)

        # make and save new predictions
        rmse_train, rmse_test = compute_rmse(EEG_data)
        new_prediction = Prediction(date_time=timezone.now(),
                                    rmse_train=rmse_train,
                                    rmse_test=rmse_test)
        new_prediction.document = doc
        new_prediction.save()


def truncate_EEG(EEG_signal):
    limiter = min(1000, len(EEG_signal))
    return EEG_signal[0:limiter]
