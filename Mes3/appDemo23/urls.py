from django.urls import path
from . import views
urlpatterns = [
    path('CambioRisa', views.CambioRisa, name='CambioRisa'),
    path('CambiarRisa', views.CambiarRisa, name='CambiarRisa')
]