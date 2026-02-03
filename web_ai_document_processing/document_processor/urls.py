from django.urls import path
from . import views

app_name = "document_processor"  # 👈 THIS IS REQUIRED

urlpatterns = [
    path('upload/', views.index, name='index'),              # Form page at /api/upload/ (assuming include prefix)
    path('process/', views.process_document, name='process_document'),  # API endpoint at /api/process/
]