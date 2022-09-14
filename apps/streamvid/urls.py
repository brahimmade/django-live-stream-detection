from django.urls import path
from . import views
app_name = 'streamvid'

urlpatterns = [
    path('img_bytes/', views.video, name=('video_image_byte_url')),
]
