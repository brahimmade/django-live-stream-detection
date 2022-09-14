from django.urls import path
from . import views
app_name = 'streamvid'

urlpatterns = [
    path('', views.video, name=('video_url')),
]
