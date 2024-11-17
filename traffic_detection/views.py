from django.shortcuts import render, redirect
from django.views.generic import TemplateView, ListView
from django.urls import reverse_lazy
from django.http import StreamingHttpResponse, JsonResponse
from .models import TrafficData, VehicleDetection
import cv2
import torch
from ultralytics import YOLO
from django.conf import settings
import os
import json
import numpy as np
from django.views.decorators.gzip import gzip_page
import tempfile
import time
from .gemini_insights import analyze_traffic_data, get_historical_analysis
from django.views.decorators.http import require_http_methods
from django.utils import timezone
from datetime import timedelta
from django.db.models import Count, Avg

# Global variables
current_stats = {
    'vehicle_count': 0,
    'vehicle_types': {},
    'congestion_level': 'LOW',
    'average_speed': 0
}

# Global video capture object and model
video_capture = None
model = YOLO('yolov5s.pt')
last_detection_time = 0
DETECTION_INTERVAL = 0.1  # Run detection every 100ms

def process_frame(frame):
    global current_stats, last_detection_time
    
    if frame is None:
        return None
    
    current_time = time.time()
    should_detect = current_time - last_detection_time >= DETECTION_INTERVAL
    
    # Only run detection at intervals
    if should_detect:
        # Resize frame for faster processing
        frame_resized = cv2.resize(frame, (640, 640))
        
        # Run YOLOv5 detection
        results = model(frame_resized)
        
        # Reset frame-specific counters
        frame_vehicles = 0
        frame_vehicle_types = {}
        
        # Calculate scale factors
        h, w = frame.shape[:2]
        scale_x = w / 640
        scale_y = h / 640
        
        # Draw detection boxes
        for detection in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = detection
            
            # Scale coordinates back to original size
            x1 = int(float(x1) * scale_x)
            y1 = int(float(y1) * scale_y)
            x2 = int(float(x2) * scale_x)
            y2 = int(float(y2) * scale_y)
            
            class_id = int(cls)
            vehicle_type = model.names[class_id]
            confidence = float(conf)
            
            # Only count vehicles with high confidence
            if vehicle_type in ['car', 'truck', 'bus', 'motorcycle'] and confidence > 0.5:
                frame_vehicles += 1
                frame_vehicle_types[vehicle_type] = frame_vehicle_types.get(vehicle_type, 0) + 1
                
                # Draw rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if confidence > 0.7:  # Only show labels for high confidence detections
                    label = f"{vehicle_type}: {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Update global statistics
        current_stats['vehicle_count'] = frame_vehicles
        current_stats['vehicle_types'] = frame_vehicle_types
        
        # Update congestion level
        if frame_vehicles < 3:
            current_stats['congestion_level'] = 'LOW'
        elif frame_vehicles < 6:
            current_stats['congestion_level'] = 'MEDIUM'
        else:
            current_stats['congestion_level'] = 'HIGH'
            
        last_detection_time = current_time
    
    # Always draw statistics overlay
    stats_text = [
        f"Vehicles: {current_stats['vehicle_count']}",
        f"Congestion: {current_stats['congestion_level']}"
    ]
    
    # Add vehicle type counts
    for v_type, count in current_stats['vehicle_types'].items():
        stats_text.append(f"{v_type}: {count}")
    
    # Draw statistics on frame
    y_position = 30
    for text in stats_text:
        cv2.putText(frame, text, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2)
        y_position += 25
    
    return frame

def get_video_stream():
    global video_capture
    
    while True:
        if video_capture is None or not video_capture.isOpened():
            # If no video is loaded, show a blank frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "No video loaded", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (255, 255, 255), 2)
        else:
            ret, frame = video_capture.read()
            if not ret:
                # Reset video to beginning if it ends
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Process frame with detection
            frame = process_frame(frame)
        
        # Convert frame to JPEG with lower quality for faster streaming
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@gzip_page
def video_feed(request):
    return StreamingHttpResponse(get_video_stream(),
                               content_type='multipart/x-mixed-replace; boundary=frame')

def get_current_stats(request):
    """API endpoint to get current statistics"""
    try:
        # Get insights for the current traffic data
        insights = analyze_traffic_data(current_stats)
        
        # Create a copy of stats to avoid modifying the global variable
        response_data = current_stats.copy()
        response_data['ai_insights'] = insights
        
        return JsonResponse(response_data)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@require_http_methods(["GET"])
def get_analytics_data(request):
    """Endpoint to get real-time analytics data"""
    # Get current stats from the global variable
    data = {
        'timestamp': timezone.now().isoformat(),
        'vehicle_count': current_stats['vehicle_count'],
        'vehicle_types': current_stats['vehicle_types'],
        'congestion_level': current_stats['congestion_level'],
        'average_speed': current_stats['average_speed']
    }
    return JsonResponse(data)

def upload_video(request):
    """Handle video upload"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed'}, status=405)
        
    try:
        if 'video' not in request.FILES:
            return JsonResponse({'error': 'No video file uploaded'}, status=400)

        video_file = request.FILES['video']
        
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            for chunk in video_file.chunks():
                tmp_file.write(chunk)
            temp_path = tmp_file.name
        
        # Update the video source
        global video_capture
        if video_capture is not None:
            video_capture.release()
            
        video_capture = cv2.VideoCapture(temp_path)
        
        if not video_capture.isOpened():
            raise Exception("Failed to open video file")
        
        # Reset statistics
        global current_stats
        current_stats = {
            'vehicle_count': 0,
            'vehicle_types': {},
            'congestion_level': 'Low',
            'average_speed': 0
        }
        
        return JsonResponse({'success': True, 'message': 'Video uploaded successfully'})
        
    except Exception as e:
        if video_capture is not None:
            video_capture.release()
        return JsonResponse({'error': str(e)}, status=500)

class HomeView(TemplateView):
    template_name = 'traffic_detection/home.html'

class VideoStreamView(TemplateView):
    template_name = 'traffic_detection/video_stream.html'

class UploadView(TemplateView):
    template_name = 'traffic_detection/upload.html'

def start_webcam(request):
    """Start webcam capture"""
    global video_capture
    
    try:
        # Close existing video capture if any
        if video_capture is not None:
            video_capture.release()
        
        # Open webcam
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            raise Exception("Failed to open webcam")
        
        # Set buffer size
        video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        return JsonResponse({'success': True})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

def stop_video(request):
    """Stop current video/webcam"""
    global video_capture
    
    if video_capture is not None:
        video_capture.release()
        video_capture = None
    
    return JsonResponse({'success': True})

class AnalyticsView(ListView):
    model = TrafficData
    template_name = 'traffic_detection/analytics.html'
    context_object_name = 'traffic_data'
    ordering = ['-timestamp']
    paginate_by = 50

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Get data from the last hour for initial display
        time_threshold = timezone.now() - timedelta(hours=1)
        traffic_data = TrafficData.objects.filter(timestamp__gte=time_threshold).order_by('timestamp')
        
        # Prepare data for charts
        traffic_data_list = []
        for data in traffic_data:
            traffic_data_list.append({
                'timestamp': data.timestamp.isoformat(),
                'vehicle_count': data.vehicle_count,
                'vehicle_types': data.vehicle_types,
                'congestion_level': data.congestion_level,
                'average_speed': float(data.average_speed)
            })
        
        # Add current stats to the list
        traffic_data_list.append({
            'timestamp': timezone.now().isoformat(),
            'vehicle_count': current_stats['vehicle_count'],
            'vehicle_types': current_stats['vehicle_types'],
            'congestion_level': current_stats['congestion_level'],
            'average_speed': float(current_stats['average_speed'])
        })
        
        # Calculate aggregate statistics
        context['total_vehicles'] = sum(data.vehicle_count for data in traffic_data)
        context['avg_speed'] = traffic_data.aggregate(Avg('average_speed'))['average_speed__avg'] or 0
        
        # Get current congestion level
        context['current_congestion'] = current_stats['congestion_level']
        
        # Add JSON data for charts
        context['traffic_data_json'] = json.dumps(traffic_data_list)
        
        return context
