from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.HomeView.as_view(), name='home'),
    path('stream/', views.VideoStreamView.as_view(), name='stream'),
    path('upload/', views.UploadView.as_view(), name='upload'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('get_stats/', views.get_current_stats, name='get_stats'),
    path('upload_video/', views.upload_video, name='upload_video'),
    path('start_webcam/', views.start_webcam, name='start_webcam'),
    path('stop_video/', views.stop_video, name='stop_video'),
    path('analytics/', views.AnalyticsView.as_view(), name='analytics'),
    path('get_analytics_data/', views.get_analytics_data, name='get_analytics_data'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
