# Server for Dreemcare
Made with Python 3 & Django 1.9.7



Note: anonymization & predictions are done during the upload as requested ("à la volée"). 
However, given the dataset size it can take quite a long time to compute prediction. 
Consequently, I just make the predictions on a portion of the dataset in edf_app/views.py.
If you have a strong computer power, you can comment the lines concerned.
