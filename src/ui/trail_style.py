# trail_style.py
from typing import Union, List, Dict
from src.routes.profile_analyzer import ProfileSegment, ProfilePoint
from src.routes.trail_features import GradientSegmentType, TrailFeatureType, ShortFeature

# Enhanced color configuration with new feature types and technical score visualization
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

# Enhanced width configuration with technical score consideration
WIDTH_CONFIG = {
    'shadow': 14,
    'base': 8,
    'highlight': {
        'default': 6,
        # Technical/Dangerous
        'TECHNICAL_ASCENT': 8,
        'TECHNICAL_DESCENT': 8,
        'DROP': 8,
        'KICKER': 8,
        # Steep/Challenging
        'STEEP_ASCENT': 7,
        'STEEP_DESCENT': 7,
        # Special Features
        'SWITCHBACK': 7,
        'ROLLER': 6.5,
        'FLOW_DESCENT': 6
    }
}

# Technical score visualization parameters
TECHNICAL_SCORE_COLORS = [
    (0, "rgba(100, 200, 100, 0.7)"),    # Easy
    (2, "rgba(200, 200, 100, 0.7)"),    # Moderate
    (4, "rgba(220, 150, 50, 0.7)"),     # Difficult
    (6, "rgba(220, 80, 80, 0.7)"),      # Very Difficult
    (8, "rgba(180, 20, 20, 0.7)")       # Extreme
]

def get_feature_name(feature_type: Union[TrailFeatureType, GradientSegmentType]) -> str:
    """Get display name for a feature with special cases"""
    name_map = {
        'KICKER': 'Step Up',
        'DROP': 'Step Down',
        'FLOW_DESCENT': 'Flow Section'
    }
    base_name = feature_type.name.replace('_', ' ')
    return name_map.get(feature_type.name, base_name).title()

def get_feature_description(feature_type: TrailFeatureType) -> str:
    """Get enhanced description for a feature including technical aspects"""
    DESCRIPTIONS = {
        'Roller': "Repeated undulations (10-50m wavelength) creating rhythmic challenges",
        'Switchback': "Sharp 180° turns requiring precise weight shifting and braking",
        'Technical Descent': "Challenging downhill with obstacles requiring line choice and control",
        'Technical Ascent': "Demanding uphill requiring power management and technical skill",
        'Flow Descent': "Smooth, rhythmic downhill section allowing for speed and fluid motion",
        'Step Up': "Short, steep climb requiring explosive power (kicker)",
        'Step Down': "Sudden steep descent requiring controlled weight shift (drop)",
    }
    return DESCRIPTIONS.get(get_feature_name(feature_type), "Trail feature")

def get_feature_color(segment: ProfileSegment, technical_score: float = None) -> str:
    """Get color based on segment features with optional technical score overlay"""
    base_color = COLOR_MAP.get(segment.feature_type) if segment.feature_type else COLOR_MAP.get(segment.gradient_type)
    
    if technical_score is not None:
        # Blend with technical score color
        score_color = get_technical_score_color(technical_score)
        return blend_colors(base_color, score_color, 0.3)
    return base_color

def get_technical_score_color(score: float) -> str:
    """Get color representing technical difficulty score"""
    for threshold, color in sorted(TECHNICAL_SCORE_COLORS, reverse=True):
        if score >= threshold:
            return color
    return TECHNICAL_SCORE_COLORS[0][1]

def blend_colors(color1: str, color2: str, ratio: float) -> str:
    """Blend two RGBA colors"""
    # Extract RGBA components
    def parse_color(c):
        parts = c[c.find('(')+1:c.find(')')].split(',')
        return [float(p.strip()) for p in parts]
    
    r1, g1, b1, a1 = parse_color(color1)
    r2, g2, b2, a2 = parse_color(color2)
    
    # Blend components
    r = r1 * (1-ratio) + r2 * ratio
    g = g1 * (1-ratio) + g2 * ratio
    b = b1 * (1-ratio) + b2 * ratio
    a = a1 * (1-ratio) + a2 * ratio
    
    return f"rgba({int(r)}, {int(g)}, {int(b)}, {round(a, 2)})"

def get_segment_name(segment: ProfileSegment) -> str:
    """Get display name for segment considering short features"""
    if segment.feature_type:
        name = get_feature_name(segment.feature_type)
        if segment.short_features:
            feature_types = {sf.feature_type for sf in segment.short_features}
            if len(feature_types) > 1:
                return f"{name} (Technical)"
        return name
    
    # For gradient segments with short features
    if segment.short_features:
        feature_types = {get_feature_name(sf.feature_type) for sf in segment.short_features}
        if len(feature_types) == 1:
            return f"{segment.gradient_type.name.replace('_', ' ').title()} with {next(iter(feature_types))}"
        return f"Technical {segment.gradient_type.name.replace('_', ' ').title()}"
    
    return segment.gradient_type.name.replace('_', ' ').title()

def get_arrow_size(segment: ProfileSegment, profile_points: List[ProfilePoint], technical_score: float = None) -> int:
    """Smart arrow sizing based on segment characteristics and technical score"""
    base_size = 24
    gradient = abs(segment.avg_gradient(profile_points))
    
    # Feature-based sizing
    if segment.feature_type:
        if segment.feature_type in [TrailFeatureType.TECHNICAL_ASCENT, TrailFeatureType.TECHNICAL_DESCENT]:
            base_size = 36
        elif segment.feature_type in [TrailFeatureType.KICKER, TrailFeatureType.DROP]:
            base_size = 28
        elif segment.feature_type == TrailFeatureType.SWITCHBACK:
            base_size = 32
    
    # Gradient-based sizing
    if gradient > 0.15:
        base_size = max(base_size, 30)
    
    # Technical score modifier
    if technical_score is not None:
        base_size += min(12, technical_score * 1.5)
    
    return base_size

def get_segment_description(segment: ProfileSegment, profile_points: List[ProfilePoint], technical_score: float = None) -> str:
    """Get comprehensive segment description with technical details"""
    description_parts = []
    
    # Main feature description
    if segment.feature_type:
        description_parts.append(get_feature_description(segment.feature_type))
    else:
        gradient = segment.avg_gradient(profile_points) * 100
        if gradient > 0:
            description_parts.append(f"Gradual ascent ({gradient:.1f}% grade)")
        else:
            description_parts.append(f"Gradual descent ({abs(gradient):.1f}% grade)")
    
    # Short features description
    if segment.short_features:
        features = [get_feature_name(sf.feature_type) for sf in segment.short_features]
        unique_features = list(set(features))
        if len(unique_features) == 1:
            description_parts.append(f"Contains {len(features)} {unique_features[0]} features")
        else:
            description_parts.append(f"Contains {len(features)} technical features")
    
    # Technical score information
    if technical_score is not None:
        difficulty_level = "Moderate"
        if technical_score > 6:
            difficulty_level = "Very Difficult"
        elif technical_score > 4:
            difficulty_level = "Difficult"
        elif technical_score > 2:
            difficulty_level = "Moderate"
        description_parts.append(f"Technical difficulty: {difficulty_level} ({technical_score:.1f}/10)")
    
    return "\n".join(description_parts)

def get_gradient_direction(segment: ProfileSegment, profile_points: List[ProfilePoint]) -> str:
    """Enhanced arrow direction based on gradient"""
    gradient = segment.avg_gradient(profile_points)
    
    if gradient > 0.10: return '⤊'  # Steep ascent
    if gradient > 0.05: return '↑'  # Ascent
    if gradient < -0.10: return '⤋'  # Steep descent
    if gradient < -0.05: return '↓'  # Descent
    return '·'  # Flat

def get_short_feature_markers(segment: ProfileSegment, points: List[ProfilePoint]) -> List[Dict]:
    """Generate visualization markers for short features in a segment"""
    markers = []
    for sf in segment.short_features:
        start_idx, end_idx = sf.get_absolute_indices(segment.start_index)
        feature_points = points[start_idx:end_idx+1]
        
        markers.append({
            'type': sf.feature_type.name,
            'positions': [(p.lat, p.lon) for p in feature_points],
            'color': COLOR_MAP.get(sf.feature_type),
            'width': WIDTH_CONFIG['highlight'].get(sf.feature_type.name, WIDTH_CONFIG['highlight']['default']),
            'description': f"{get_feature_name(sf.feature_type)}: {sf.length:.1f}m at {sf.max_gradient*100:.1f}%"
        })
    return markers

def get_segment_style(segment: ProfileSegment, points: List[ProfilePoint], technical_score: float = None) -> Dict:
    """Get complete style configuration for a segment"""
    return {
        'name': get_segment_name(segment),
        'color': get_feature_color(segment, technical_score),
        'width': get_segment_width(segment),
        'arrow_size': get_arrow_size(segment, points, technical_score),
        'description': get_segment_description(segment, points, technical_score),
        'short_features': get_short_feature_markers(segment, points),
        'gradient_direction': get_gradient_direction(segment, points)
    }

def get_segment_width(segment: ProfileSegment) -> float:
    """Determine segment line width based on features and gradient"""
    base_width = WIDTH_CONFIG['base']
    
    if segment.feature_type:
        feature_name = segment.feature_type.name
        return WIDTH_CONFIG['highlight'].get(feature_name, WIDTH_CONFIG['highlight']['default'])
    
    gradient_type = segment.gradient_type.name
    return WIDTH_CONFIG['highlight'].get(gradient_type, base_width)