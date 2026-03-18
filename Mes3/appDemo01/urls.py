from django.urls import path
from . import views
urlpatterns = [
    path('ConsultaRopa', views.ConsultaRopa, name='ConsultaRopa'),
    path('GenerarRopa', views.GenerarRopa, name='GenerarRopa')
]