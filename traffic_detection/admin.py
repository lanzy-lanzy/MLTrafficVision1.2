from django.contrib import admin
from .models import TrafficData, VehicleDetection

# Register your models here.

@admin.register(TrafficData)
class TrafficDataAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'vehicle_count', 'average_speed', 'congestion_level')
    list_filter = ('congestion_level', 'timestamp')
    search_fields = ('timestamp',)

@admin.register(VehicleDetection)
class VehicleDetectionAdmin(admin.ModelAdmin):
    list_display = ('vehicle_type', 'speed', 'confidence', 'timestamp')
    list_filter = ('vehicle_type', 'timestamp')
    search_fields = ('vehicle_type', 'timestamp')
