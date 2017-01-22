from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^edf_manager/$', views.edf_manager, name='edf_manager'),
    url(r'^predictions/$', views.predictions, name='predictions')
]
