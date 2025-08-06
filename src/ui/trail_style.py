# trail_style.py
from typing import Union
from src.routes.trail_features import ElevationSegment
from src.routes.statistics_collector import GradientSegmentType, TrailFeatureType

# Color configuration consistent with map_visualization.py
COLOR_MAP = {
    # Base Gradient Segments
    GradientSegmentType.ASCENT: "rgba(220, 80, 80, 0.9)",
    GradientSegmentType.DESCENT: "rgba(80, 130, 220, 0.9)",
    GradientSegmentType.STEEP_ASCENT: "rgba(200, 0, 0, 1.0)",
    GradientSegmentType.STEEP_DESCENT: "rgba(0, 70, 200, 1.0)",
    GradientSegmentType.FLAT: "rgba(150, 150, 150, 0.5)",
    
    # Technical Features
    TrailFeatureType.TECHNICAL_ASCENT: "rgba(180, 20, 20, 0.95)",
    TrailFeatureType.TECHNICAL_DESCENT: "rgba(20, 60, 150, 0.95)",
    
    # Steps
    TrailFeatureType.KICKER: "rgba(255, 50, 50, 1.0)",
    TrailFeatureType.DROP: "rgba(50, 120, 255, 1.0)",
    
    # Special Features
    TrailFeatureType.ROLLER: "rgba(255, 195, 0, 0.95)",
    TrailFeatureType.FLOW_DESCENT: "rgba(100, 220, 100, 0.95)",
    TrailFeatureType.SWITCHBACK: "rgba(255, 165, 0, 0.95)",
}

WIDTH_CONFIG = {
    'shadow': 14,
    'base': 8,
    'highlight': {
        'default': 6,
        # Technical/Dangerous
        'TECHNICAL_ASCENT': 8,
        'TECHNICAL_DESCENT': 8,
        'DROP_SECTION': 8,
        # Steep/Challenging
        'STEEP_ASCENT': 7,
        'STEEP_DESCENT': 7,
        # Special Features
        'SWITCHBACK': 7,
        'STEP_UP': 6.5,
        'STEP_DOWN': 6.5
    }
}

def get_feature_name(feature_type: Union[TrailFeatureType, GradientSegmentType]) -> str:
    """Get display name for a feature"""
    return feature_type.name.replace('_', ' ').title()

def get_feature_description(feature_type: TrailFeatureType) -> str:
    """Get description for a feature"""
    DESCRIPTIONS = {
        'Roller': "Repeated undulations (10-50m wavelength)",
        'Switchback': "Sharp 180° turns changing direction",
        'Technical Descent': "Challenging downhill with obstacles",
        'Technical Ascent': "Challenging uphill requiring skill",
        'Flow Descent': "Smooth, rhythmic downhill section",
        'Kicker': "Very short, steep climb requiring a power burst",
        'Drop': "Sudden steep descent requiring weight shift",
    }
    return DESCRIPTIONS.get(get_feature_name(feature_type), "Trail feature")

def get_feature_color(segment: ElevationSegment) -> str:
    """Get color based on segment features"""
    if segment.feature_type:
        return COLOR_MAP.get(segment.feature_type)
    return "rgba(150, 150, 150, 0.5)"

def get_segment_name(segment: ElevationSegment) -> str:
    """Get display name for segment"""
    if segment.feature_type:
        return segment.feature_type.name.replace('_', ' ').title()
    return segment.gradient_type.name.replace('_', ' ').title()

def get_arrow_size(segment: ElevationSegment) -> int:
    """Smart arrow sizing based on segment characteristics"""
    gradient = abs(segment.avg_gradient())
    
    # Technical features get largest arrows
    if segment.feature_type in [TrailFeatureType.TECHNICAL_ASCENT, 
                              TrailFeatureType.TECHNICAL_DESCENT]:
        return 36
    
    # Steep segments get large arrows
    if gradient > 0.15:
        return 30
    
    # Steps get medium arrows
    if segment.feature_type in [TrailFeatureType.KICKER, TrailFeatureType.DROP]:
        return 26
    
    # Default size for other segments
    return 24

def get_segment_description(segment: ElevationSegment) -> str:
    """Get comprehensive segment description"""
    if segment.feature_type:
        return get_feature_description(segment.feature_type)
    
    gradient = segment.avg_gradient() * 100
    if gradient > 0:
        return f"Gradual ascent ({gradient:.1f}% grade)"
    return f"Gradual descent ({abs(gradient):.1f}% grade)"

def get_gradient_direction(gradient: float) -> str:
    """Determine arrow direction based on gradient"""
    if gradient > 0.05: return '↑'  # Significant ascent
    if gradient < -0.05: return '↓'  # Significant descent
    return '·'  # Flat