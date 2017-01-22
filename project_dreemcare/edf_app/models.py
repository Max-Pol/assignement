from __future__ import unicode_literals
from django.db import models


class Document(models.Model):
    docfile = models.FileField(upload_to='edf_files/%Y/%m/%d')


class Prediction(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    date_time = models.DateTimeField()
    rmse_train = models.FloatField(blank=True, null=True)
    rmse_test = models.FloatField(blank=True, null=True)
