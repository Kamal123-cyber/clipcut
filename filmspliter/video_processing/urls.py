from django.urls import path
from .views import UploadVideoAPIView, SearchFaceAPIView

urlpatterns = [
    path("videos/upload/", UploadVideoAPIView.as_view()),
    path("videos/search-face/", SearchFaceAPIView.as_view()),
]