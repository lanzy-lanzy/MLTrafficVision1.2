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
        
        if vehicle_count > self.congestion_thresholds['high']:
            concerns.append("Risk of traffic congestion - consider traffic management measures")
        
        if vehicle_count > self.congestion_thresholds['medium']:
            concerns.append("Increased vehicle density may affect traffic flow")
            
        if 'truck' in stats['vehicle_types'] and stats['vehicle_types']['truck'] > 5:
            concerns.append("High number of trucks may slow traffic flow")
            
        return concerns

    def get_recommendations(self, stats: Dict[str, Any]) -> list:
        recommendations = []
        vehicle_count = stats['vehicle_count']
        
        if vehicle_count > self.congestion_thresholds['high']:
            recommendations.append("Consider implementing traffic control measures")
            recommendations.append("Monitor for potential bottlenecks")
        
        if vehicle_count > self.congestion_thresholds['medium']:
            recommendations.append("Consider activating additional lanes if available")
        
        if 'truck' in stats['vehicle_types'] and stats['vehicle_types']['truck'] > 5:
            recommendations.append("Consider dedicated lanes for heavy vehicles")
            
        return recommendations

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
        
        insights = {
            'assessment': f"{assessment}. {' '.join(vehicle_insights)}",
            'concerns': concerns,
            'recommendations': recommendations,
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
