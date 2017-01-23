# Server for Dreemcare

## Description
The key features of this project are:

* **Manage EDF Files**: Upload an EDF File, anonymize it, and make it available for download. During the upload,the evolution of *the first EEG channel* is predicted with a LSTM Neural Network.
* **Get Prediction Scores**: Display the performance of the Neural Network using the Root Mean Square Error (RMSE) of the train and test sets.
* **Train Model**: Train the LSTM Neural Network with the EDF of your choice.


## Requirements
* `python`
* `virtualenv`

## Installation (through virtualenv)
To set up python 3 virtualenv, install the project dependencies and run the Django app. Enter the following into a terminal:

```shell
git clone https://github.com/Max-Pol/assignement.git
cd assignement/project_dreemcare
virtualenv -p /usr/bin/python3 venv
source venv/bin/activate
pip install -r requirements.txt

python manage.py migrate

python manage.py runserver
```

## Script
### Train_model
```shell
python manage.py train_model -f <file_id> [-n <nb_epoch>]
```
This script allows you to **train the LSTM Neural Network** with the EDF file of your choice.
The new LSTM Neural Network model replaces the previous one in *media/model/*, and will be used for the next file upload.

The **Trainscore** and **Testscore** (RMSE) are also automatically saved and updated, as you can see in
the *Get Prediction Scores* section.
 
*Note:* The `<nb_epoch>` (number of epoch)  is set to 100 by default.


## Notes
**Anonymization** &  **scores calculation** are done during the upload as requested ("à la volée"). 
However, given the dataset size it can take a long time to compute the predictions.

For this reason, I truncate the dataset in *edf_app.views*, for my poor computer to calculate the RMSE
during the upload.
If you have a stronger computing power, just comment the lines concerned.
However, when you run the script `train_model.py`, the RMSE are computed on the whole dataset.