from typing import Dict, Any
import json
from datetime import datetime

class TrafficInsightsGenerator:
    def __init__(self):
        self.congestion_thresholds = {
            'low': 5,
            'medium': 15,
            'high': 30
        }
    
    def get_congestion_assessment(self, vehicle_count: int) -> str:
        if vehicle_count <= self.congestion_thresholds['low']:
            return "Light traffic flow"
        elif vehicle_count <= self.congestion_thresholds['medium']:
            return "Moderate traffic density"
        else:
            return "Heavy traffic congestion"

    def get_vehicle_distribution_insights(self, vehicle_types: Dict[str, int]) -> list:
        total = sum(vehicle_types.values())
        insights = []
        
        if total == 0:
            return ["No vehicles detected"]
            
        for vehicle_type, count in vehicle_types.items():
            percentage = (count / total) * 100
            if percentage > 30:
                insights.append(f"High proportion of {vehicle_type}s ({percentage:.1f}%)")
            elif percentage > 15:
                insights.append(f"Moderate number of {vehicle_type}s ({percentage:.1f}%)")
        
        return insights

    def get_concerns(self, stats: Dict[str, Any]) -> list:
        concerns = []
        vehicle_count = stats['vehicle_count']
        vehicle_types = stats.get('vehicle_types', {})
        speed_data = stats.get('speed_data', {})
        avg_speed = speed_data.get('average', 0)
        
        # Traffic Volume Concerns
        if vehicle_count > self.congestion_thresholds['high']:
            concerns.append("⚠️ Severe traffic congestion detected - immediate traffic management needed")
        elif vehicle_count > self.congestion_thresholds['medium']:
            concerns.append("⚠️ Moderate traffic buildup - monitor situation closely")
            
        # Vehicle Type Distribution Concerns
        if 'truck' in vehicle_types and vehicle_types['truck'] > 5:
            concerns.append("🚛 High concentration of heavy vehicles may impact traffic flow")
        
        total_vehicles = sum(vehicle_types.values())
        for vehicle_type, count in vehicle_types.items():
            if total_vehicles > 0:
                percentage = (count / total_vehicles) * 100
                if percentage > 40:
                    concerns.append(f"⚠️ Unusually high proportion of {vehicle_type}s ({percentage:.1f}%)")
        
        # Speed-related Concerns
        if avg_speed < 20 and vehicle_count > self.congestion_thresholds['medium']:
            concerns.append("🐢 Very slow traffic movement detected")
        elif avg_speed > 60:
            concerns.append("⚡ Vehicles exceeding recommended speed limits")
            
        # Time-based Concerns
        current_hour = datetime.now().hour
        if current_hour in [7, 8, 9, 16, 17, 18] and vehicle_count > self.congestion_thresholds['medium']:
            concerns.append("⏰ Peak hour traffic congestion")
            
        return concerns

    def get_recommendations(self, stats: Dict[str, Any]) -> list:
        recommendations = []
        vehicle_count = stats['vehicle_count']
        vehicle_types = stats.get('vehicle_types', {})
        speed_data = stats.get('speed_data', {})
        avg_speed = speed_data.get('average', 0)
        
        # Traffic Management Recommendations
        if vehicle_count > self.congestion_thresholds['high']:
            recommendations.append("🚦 Activate adaptive traffic signal timing")
            recommendations.append("📱 Issue traffic alerts to motorists")
            recommendations.append("🔄 Consider temporary lane reassignment")
        elif vehicle_count > self.congestion_thresholds['medium']:
            recommendations.append("👀 Monitor intersection performance")
            recommendations.append("🚗 Consider opening additional lanes")
        
        # Vehicle Type-specific Recommendations
        if 'truck' in vehicle_types and vehicle_types['truck'] > 5:
            recommendations.append("🛣️ Implement temporary truck routing plan")
            recommendations.append("⏰ Consider time-based restrictions for heavy vehicles")
        
        # Speed Management Recommendations
        if avg_speed < 20 and vehicle_count > self.congestion_thresholds['medium']:
            recommendations.append("⚡ Optimize signal timing to improve flow")
            recommendations.append("📢 Deploy traffic officers at key points")
        elif avg_speed > 60:
            recommendations.append("🚓 Increase speed monitoring")
            recommendations.append("⚠️ Display speed warning messages")
        
        # Time-based Recommendations
        current_hour = datetime.now().hour
        if current_hour in [7, 8, 9, 16, 17, 18]:
            recommendations.append("🕒 Implement peak hour traffic measures")
            recommendations.append("🚌 Encourage use of public transportation")
        
        return recommendations

    def analyze_traffic_pattern(self, stats: Dict[str, Any]) -> str:
        vehicle_count = stats['vehicle_count']
        vehicle_types = stats.get('vehicle_types', {})
        speed_data = stats.get('speed_data', {})
        avg_speed = speed_data.get('average', 0)
        
        pattern_insights = []
        
        # Traffic Flow Analysis
        if vehicle_count > self.congestion_thresholds['high']:
            pattern_insights.append("Heavy congestion detected")
        elif vehicle_count > self.congestion_thresholds['medium']:
            pattern_insights.append("Moderate traffic volume")
        else:
            pattern_insights.append("Normal traffic flow")
        
        # Speed Pattern
        if avg_speed < 20:
            pattern_insights.append("slow-moving traffic")
        elif avg_speed > 60:
            pattern_insights.append("high-speed traffic flow")
        else:
            pattern_insights.append("steady flow")
            
        # Vehicle Mix
        dominant_type = max(vehicle_types.items(), key=lambda x: x[1])[0] if vehicle_types else None
        if dominant_type:
            pattern_insights.append(f"predominantly {dominant_type} traffic")
        
        return " with ".join(pattern_insights)

def analyze_traffic_data(stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze traffic data and provide insights
    """
    try:
        insights_generator = TrafficInsightsGenerator()
        
        assessment = insights_generator.get_congestion_assessment(stats['vehicle_count'])
        vehicle_insights = insights_generator.get_vehicle_distribution_insights(stats['vehicle_types'])
        concerns = insights_generator.get_concerns(stats)
        recommendations = insights_generator.get_recommendations(stats)
        traffic_pattern = insights_generator.analyze_traffic_pattern(stats)
        
        insights = {
            'assessment': f"{assessment}. {' '.join(vehicle_insights)}",
            'concerns': concerns,
            'recommendations': recommendations,
            'traffic_pattern': traffic_pattern,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return {
            'success': True,
            'insights': insights
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def get_historical_analysis(stats_history: list) -> Dict[str, Any]:
    """
    Analyze historical traffic patterns
    """
    try:
        total_vehicles = sum(stat['vehicle_count'] for stat in stats_history)
        avg_vehicles = total_vehicles / len(stats_history) if stats_history else 0
        
        analysis = {
            'average_vehicles': avg_vehicles,
            'peak_count': max((stat['vehicle_count'] for stat in stats_history), default=0),
            'trends': []
        }
        
        if avg_vehicles > 20:
            analysis['trends'].append("High average traffic volume")
        elif avg_vehicles > 10:
            analysis['trends'].append("Moderate average traffic volume")
        else:
            analysis['trends'].append("Low average traffic volume")
            
        return {
            'success': True,
            'analysis': analysis
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
