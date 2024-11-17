from django.db import models
from django.utils import timezone

# Create your models here.

class TrafficData(models.Model):
    timestamp = models.DateTimeField(default=timezone.now)
    video_file = models.FileField(upload_to='videos/')
    processed_video = models.FileField(upload_to='processed_videos/', null=True, blank=True)
    vehicle_count = models.IntegerField(default=0)
    vehicle_types = models.JSONField(default=dict)  # Store vehicle types and their counts
    average_speed = models.FloatField(default=0.0)
    congestion_level = models.CharField(max_length=20, choices=[
        ('LOW', 'Low'),
        ('MEDIUM', 'Medium'),
        ('HIGH', 'High'),
    ])
    incidents = models.JSONField(default=list)  # Store detected incidents

    def __str__(self):
        return f"Traffic Data - {self.timestamp}"

class VehicleDetection(models.Model):
    traffic_data = models.ForeignKey(TrafficData, on_delete=models.CASCADE, related_name='detections')
    timestamp = models.DateTimeField()
    vehicle_type = models.CharField(max_length=50)
    speed = models.FloatField()
    confidence = models.FloatField()
    bbox_coordinates = models.JSONField()  # Store bounding box coordinates

    def __str__(self):
        return f"{self.vehicle_type} at {self.timestamp}"
