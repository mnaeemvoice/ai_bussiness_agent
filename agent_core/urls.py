from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),  # Root of agent_core
    path('llm_inference/', views.llm_inference_view, name='llm_inference'),
    path('upload_pdf/', views.upload_pdf_view, name='upload_pdf'),
    path('whatsapp_webhook/', views.whatsapp_webhook_view, name='whatsapp_webhook'),
    path('whatsapp_session/', views.whatsapp_session_view, name='whatsapp_session'),
]