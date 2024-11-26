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
    'average_speed': 0,
    'speeds': [],  # List to store recent speed measurements
    'category_counts': {
        'motorized': 0,
        'non_motorized': 0,
        'personal_mobility': 0,
        'emergency': 0
    }
}

# Define vehicle categories
VEHICLE_CATEGORIES = {
    'motorized': ['car', 'truck', 'bus', 'motorcycle'],
    'non_motorized': ['bicycle'],
    'personal_mobility': ['bicycle', 'motorcycle'],
    'emergency': ['truck']  # Some emergency vehicles might be classified as trucks
}

# Performance optimization settings
FRAME_SKIP = 1  # Process every frame for better accuracy
MAX_DIMENSION = 800  # Increased resolution for better detection
CONFIDENCE_THRESHOLD = 0.45  # Slightly lower threshold to catch more vehicles
MIN_DETECTION_AREA = 1000  # Minimum area for vehicle detection
IOU_THRESHOLD = 0.5  # Intersection over Union threshold for NMS

# Global video capture object and model
video_capture = None
model = YOLO('yolov8n.pt')
model.conf = CONFIDENCE_THRESHOLD
model.iou = IOU_THRESHOLD
if torch.cuda.is_available():
    model.to('cuda')  # Use GPU if available
last_detection_time = 0
DETECTION_INTERVAL = 0.1  # Run detection every 100ms

def optimize_frame_size(frame):
    """Optimize frame size while maintaining aspect ratio"""
    height, width = frame.shape[:2]
    if max(height, width) > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(frame, (new_width, new_height))
    return frame

def process_frame(frame):
    global current_stats, last_detection_time
    
    if frame is None:
        return None
        
    current_time = time.time()
    
    try:
        # Optimize frame size while maintaining aspect ratio
        original_height, original_width = frame.shape[:2]
        frame_processed = optimize_frame_size(frame.copy())
        processed_height, processed_width = frame_processed.shape[:2]
        
        # Calculate scale factors for coordinate conversion
        scale_x = original_width / processed_width
        scale_y = original_height / processed_height
        
        # Run detection with augmented inference
        results = model(frame_processed, verbose=False, augment=True)  # Enable test time augmentation
        result = results[0]
        
        # Reset counters
        frame_vehicles = 0
        frame_vehicle_types = {}
        frame_category_counts = {cat: 0 for cat in VEHICLE_CATEGORIES.keys()}
        
        # Process detections
        if len(result.boxes) > 0:
            # Get all detections at once
            boxes = result.boxes
            confidences = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            coordinates = boxes.xyxy.cpu().numpy()
            
            # Filter detections
            valid_detections = []
            
            for box, class_id, conf in zip(coordinates, classes, confidences):
                vehicle_type = model.names[int(class_id)]
                
                # Calculate detection area
                x1, y1, x2, y2 = map(int, box)
                area = (x2 - x1) * (y2 - y1)
                
                # Apply size and confidence filters
                if (vehicle_type in sum(VEHICLE_CATEGORIES.values(), []) and 
                    conf > CONFIDENCE_THRESHOLD and 
                    area > MIN_DETECTION_AREA):
                    
                    # Scale coordinates back to original frame size
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    
                    valid_detections.append({
                        'box': (x1, y1, x2, y2),
                        'type': vehicle_type,
                        'conf': conf,
                        'area': area
                    })
                    
                    # Update counters
                    frame_vehicles += 1
                    frame_vehicle_types[vehicle_type] = frame_vehicle_types.get(vehicle_type, 0) + 1
                    
                    # Update category counts
                    for category, types in VEHICLE_CATEGORIES.items():
                        if vehicle_type in types:
                            frame_category_counts[category] += 1
                    
                    # Calculate center point of detection
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Calculate and update speed if we have previous detections
                    if hasattr(process_frame, 'prev_centers'):
                        for prev_center in process_frame.prev_centers:
                            dist = ((center_x - prev_center[0])**2 + (center_y - prev_center[1])**2)**0.5
                            if dist < 100:  # If it's likely the same vehicle
                                speed = calculate_speed(prev_center, (center_x, center_y), time.time() - last_detection_time)
                                current_stats['speeds'].append(speed)
                                # Keep only recent speed measurements
                                current_stats['speeds'] = current_stats['speeds'][-20:]
                    
                    # Store current centers for next frame
                    if not hasattr(process_frame, 'prev_centers'):
                        process_frame.prev_centers = []
                    process_frame.prev_centers = [(center_x, center_y)]
                    
                    # Calculate average speed
                    if current_stats['speeds']:
                        current_stats['average_speed'] = sum(current_stats['speeds']) / len(current_stats['speeds'])
                    
                    # Draw detection box and label with speed
                    color = get_vehicle_color(vehicle_type)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add detection label with speed in yellow
                    speed_text = f"{get_vehicle_icon(vehicle_type)} {vehicle_type}: {current_stats['average_speed']:.1f} km/h"
                    label_size, baseline = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    y1 = max(y1, label_size[1])
                    
                    # Draw label background
                    cv2.rectangle(frame, 
                                (x1, y1 - label_size[1] - baseline),
                                (x1 + label_size[0], y1),
                                (0, 0, 0),  # Black background
                                cv2.FILLED)
                    
                    # Draw label text in yellow
                    cv2.putText(frame, speed_text, (x1, y1 - baseline),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)  # Yellow color
        
        # Update global stats with smoothing
        alpha = 0.7  # Smoothing factor
        for key in current_stats['category_counts'].keys():
            current_stats['category_counts'][key] = int(
                alpha * current_stats['category_counts'][key] + 
                (1 - alpha) * frame_category_counts[key]
            )
        
        current_stats.update({
            'vehicle_count': frame_vehicles,
            'vehicle_types': frame_vehicle_types,
            'congestion_level': get_congestion_level(frame_category_counts)
        })
        
        last_detection_time = current_time
        
    except Exception as e:
        print(f"Error processing frame: {e}")
    
    # Draw overlay with enhanced visibility
    draw_overlay(frame)
    return frame

def draw_overlay(frame):
    """Draw statistics overlay with enhanced visibility"""
    # Create semi-transparent overlay background
    overlay = frame.copy()
    overlay_height = 150
    cv2.rectangle(overlay, (0, 0), (300, overlay_height), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    stats_text = [
        f"[TOTAL] Total Vehicles: {current_stats['vehicle_count']}",
        f"[SPEED] Avg Speed: {current_stats['average_speed']:.1f} km/h",
        f"[STATUS] Congestion: {current_stats['congestion_level']}",
        f"[AUTO] Motorized: {current_stats['category_counts']['motorized']}",
        f"[BIKE] Non-motorized: {current_stats['category_counts']['non_motorized']}"
    ]
    
    # Add vehicle type breakdown
    for v_type, count in current_stats['vehicle_types'].items():
        stats_text.append(f"{v_type.capitalize()}: {count}")
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    padding = 25
    
    for i, text in enumerate(stats_text):
        y_position = 30 + (i * padding)
        # Draw shadow for better visibility
        cv2.putText(frame, text, (11, y_position + 1), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(frame, text, (10, y_position), font, font_scale, (255, 255, 255), thickness)

def get_vehicle_color(vehicle_type):
    """Get color for vehicle type"""
    if vehicle_type in VEHICLE_CATEGORIES['emergency']:
        return (0, 0, 255)  # Red
    elif vehicle_type in VEHICLE_CATEGORIES['non_motorized']:
        return (255, 165, 0)  # Orange
    elif vehicle_type in VEHICLE_CATEGORIES['personal_mobility']:
        return (255, 0, 255)  # Purple
    return (0, 255, 0)  # Green

def get_vehicle_icon(vehicle_type):
    """Return a text-based icon for the vehicle type"""
    icons = {
        'car': '[CAR]',
        'truck': '[TRUCK]',
        'bus': '[BUS]',
        'motorcycle': '[MOTO]',
        'bicycle': '[BIKE]'
    }
    return icons.get(vehicle_type.lower(), '[VEH]')  # Default to [VEH] if type not found

def get_congestion_level(category_counts):
    """Determine congestion level"""
    total_traffic = category_counts['motorized'] + category_counts['non_motorized']
    if total_traffic < 4:
        return 'LOW'
    elif total_traffic < 8:
        return 'MEDIUM'
    return 'HIGH'

def calculate_speed(prev_pos, curr_pos, time_diff):
    """Calculate speed in km/h given two positions and time difference"""
    if time_diff == 0:
        return 0
    # Convert pixel distance to meters (assuming average lane width is 3.5 meters)
    pixel_distance = ((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)**0.5
    meters = pixel_distance * (3.5 / 100)  # Approximate conversion
    speed = (meters / time_diff) * 3.6  # Convert m/s to km/h
    return min(speed, 120)  # Cap at 120 km/h to filter outliers

def get_video_stream():
    global video_capture
    frame_count = 0
    
    while True:
        if video_capture is None or not video_capture.isOpened():
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "No video loaded", (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            ret, frame = video_capture.read()
            if not ret:
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Process only every nth frame
            frame_count += 1
            if frame_count % FRAME_SKIP == 0:
                frame = process_frame(frame)
            else:
                # Just draw overlay for skipped frames
                draw_overlay(frame)
        
        # Optimize JPEG encoding
        ret, buffer = cv2.imencode('.jpg', frame, [
            cv2.IMWRITE_JPEG_QUALITY, 85,
            cv2.IMWRITE_JPEG_OPTIMIZE, 1
        ])
        
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

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
        'average_speed': current_stats['average_speed'],
        'category_counts': current_stats['category_counts']
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
            'average_speed': 0,
            'speeds': [],  # List to store recent speed measurements
            'category_counts': {
                'motorized': 0,
                'non_motorized': 0,
                'personal_mobility': 0,
                'emergency': 0
            }
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
