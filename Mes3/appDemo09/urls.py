from django.urls import path
from . import views
urlpatterns = [
    path('GenerarCaraAlumno', views.GenerarCaraAlumno, name='GenerarCaraAlumno'),
    path('GenerarCaras', views.GenerarCaras, name='GenerarCaras')
]