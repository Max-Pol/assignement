from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse
from django.utils import timezone
from project_dreemcare.settings import PROJECT_ROOT

from edf_app.models import Document, Prediction
from edf_app.forms import DocumentForm
from edf_app.edf_reader import *
from edf_app.prediction import *


def index(request):
    return render(
        request,
        'edf_app/index.html',
    )


def edf_manager(request):
    # Handle file upload
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = Document(docfile=request.FILES['docfile'])
            newdoc.save()

            # anonymization of EDF
            newedf = EDFReader(PROJECT_ROOT + newdoc.docfile.url)
            newedf.anonymization()

            # make and save predictions
            EEG_signal = newedf.get_signal(2)
            # limit dataset size to reduce computing time. Can be commented.
            if len(EEG_signal) > 10000:
                EEG_signal = EEG_signal[0:10000]

            rmse_train, rmse_test = compute_rmse(EEG_signal)
            new_prediction = Prediction(date_time=timezone.now(),
                                        rmse_train=rmse_train,
                                        rmse_test=rmse_test)
            new_prediction.document = newdoc
            new_prediction.save()

            # Redirect to the document list after POST
            return HttpResponseRedirect(reverse('edf_manager'))
    else:
        form = DocumentForm()  # A empty, unbound form

    # Load documents for the list page
    documents = Document.objects.all()

    # Render list page with the documents and the form
    return render(request,
                  'edf_app/edf_manager.html',
                  {'documents': documents,
                   'form': form,
                   'active_nav_tab': ["active", ""]
                   })


def predictions(request):
    # Load documents data for the list page
    documents_data = []

    documents = Document.objects.all()
    for doc in documents:
        try:
            doc_predictions = Prediction.objects \
                                        .filter(document=doc) \
                                        .order_by('-date_time')
            documents_data.append([doc,
                                  [doc_predictions[0].rmse_train,
                                   doc_predictions[0].rmse_test]])
        except:
            documents_data.append([doc, ["not computed", "not computed"]])

    # Render list page with the documents and the form
    return render(request,
                  'edf_app/predictions.html',
                  {'documents_data': documents_data,
                   'active_nav_tab': ["", "active"]
                   })
