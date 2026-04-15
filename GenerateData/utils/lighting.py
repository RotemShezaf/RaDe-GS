"""
Lighting configurations for Open3D rendering.

This module defines lighting presets for different rendering modes:
- Standard/default lighting (balanced setup)
- Fixed light ID presets (5 variations: 0-4)
- Decoupled appearance groups (10 variations for appearance diversity)
"""

from typing import List, Tuple

# Type alias for light configuration: (name_suffix, color, direction, intensity)
LightConfig = Tuple[str, List[float], List[float], float]


def get_lighting_config(group_id: int, use_decoupled_appearance: bool = False) -> Tuple[List[LightConfig], float]:
    """Get lighting configuration for a specific group.
    
    Args:
        group_id: Lighting group ID
                 -1 = default/standard lighting
                 0-5 = fixed light ID presets (5 = natural sunlight)
                 0-9 = decoupled appearance groups (when use_decoupled_appearance=True)
        use_decoupled_appearance: If True, uses appearance variation mode
    
    Returns:
        Tuple of (light_configs, indirect_intensity)
        where light_configs is a list of (name_suffix, color, direction, intensity)
    """
    if group_id == -1:
        # DEFAULT/STANDARD MODE: Balanced lighting for clear shading
        return STANDARD_LIGHTING
    
    elif group_id in LIGHT_ID_PRESETS and not use_decoupled_appearance:
        # FIXED LIGHT ID MODE: preset configurations (0-5)
        return LIGHT_ID_PRESETS[group_id]
    
    elif use_decoupled_appearance:
        # APPEARANCE VARIATION MODE: Different lighting per group
        group_id = min(group_id, 9)  # Clamp to valid range
        return APPEARANCE_GROUPS[group_id]
    
    else:
        # Fallback to standard
        return STANDARD_LIGHTING


# =============================================================================
# STANDARD LIGHTING (Default/Baseline)
# Natural, balanced lighting with gentle shading from multiple angles
# for good detail capture when rendering from different viewpoints.
# =============================================================================
STANDARD_LIGHTING = (
    [
        # Key light - warm directional from front-top-right (mimics sun at ~45°)
        ("key_light", [1.0, 0.98, 0.95], [-0.3, -0.7, -0.8], 55000),
        
        # Fill lights - softer, from opposite sides to reduce harsh shadows
        ("front_right", [0.98, 0.98, 1.0], [0.7, -0.4, -0.7], 18000),
        ("front_left", [1.0, 0.98, 0.98], [-0.7, -0.4, -0.7], 18000),
        ("top", [1.0, 1.0, 0.98], [0.0, -0.9, -0.3], 14000),
        # Back/rim lights for edge definition from rear viewpoints
        ("back_center", [0.96, 0.98, 1.0], [0.0, -0.3, 0.9], 20000),
        ("back_right", [0.98, 0.98, 1.0], [0.6, -0.3, 0.6], 16000),
        ("back_left", [1.0, 0.98, 0.98], [-0.6, -0.3, 0.6], 16000),
    ],
    30000  # Indirect intensity - moderate ambient for shadow fill
)


# =============================================================================
# FIXED LIGHT ID PRESETS (0-5)
# Each preset provides natural lighting from different primary angles to ensure
# good detail and shading regardless of the rendering viewpoint.
# =============================================================================
LIGHT_ID_PRESETS = {
    0: (  # Front-right key (default balanced, natural daylight)
        [
            ("key_light", [1.0, 0.98, 0.95], [-0.3, -0.7, -0.8], 55000),
            ("front_right", [0.98, 0.98, 1.0], [0.7, -0.4, -0.7], 18000),
            ("front_left", [1.0, 0.98, 0.98], [-0.7, -0.4, -0.7], 18000),
            ("top", [1.0, 1.0, 0.98], [0.0, -0.9, -0.3], 14000),
            ("back_center", [0.96, 0.98, 1.0], [0.0, -0.3, 0.9], 20000),
            ("back_right", [0.98, 0.98, 1.0], [0.6, -0.3, 0.6], 16000),
            ("back_left", [1.0, 0.98, 0.98], [-0.6, -0.3, 0.6], 16000),
        ],
        30000
    ),

    1: (  # Front-left key with warm fill (morning light)
        [
            ("key_light", [1.0, 0.97, 0.92], [-0.5, -0.6, -0.7], 58000),
            ("fill_right", [0.96, 0.98, 1.0], [0.6, -0.4, -0.6], 16000),
            ("fill_left", [1.0, 0.97, 0.94], [-0.6, -0.5, -0.6], 14000),
            ("top", [1.0, 0.99, 0.96], [0.0, -0.85, -0.3], 13000),
            ("back_center", [0.95, 0.97, 1.0], [0.0, -0.35, 0.85], 19000),
            ("back_right", [0.97, 0.98, 1.0], [0.5, -0.35, 0.7], 15000),
            ("back_left", [1.0, 0.97, 0.95], [-0.5, -0.35, 0.7], 15000),
        ],
        28000
    ),

    2: (  # Top-down emphasis with cool tones (overcast daylight)
        [
            ("key_light", [0.96, 0.98, 1.0], [-0.2, -0.8, -0.5], 52000),
            ("fill_right", [0.97, 0.98, 1.0], [0.65, -0.4, -0.55], 18000),
            ("fill_left", [0.98, 1.0, 1.0], [-0.65, -0.4, -0.55], 17000),
            ("top", [0.97, 0.99, 1.0], [0.0, -0.9, -0.2], 16000),
            ("back_center", [0.95, 0.97, 1.0], [0.0, -0.3, 0.88], 20000),
            ("back_right", [0.96, 0.98, 1.0], [0.55, -0.3, 0.65], 16000),
            ("back_left", [0.98, 0.98, 1.0], [-0.55, -0.3, 0.65], 16000),
        ],
        32000
    ),

    3: (  # Side key from right with softer fill (afternoon light)
        [
            ("key_light", [1.0, 0.98, 0.94], [0.6, -0.65, -0.5], 54000),
            ("fill_left", [0.96, 0.98, 1.0], [-0.6, -0.45, -0.6], 17000),
            ("fill_front", [0.98, 0.98, 1.0], [0.0, -0.5, -0.8], 15000),
            ("top", [0.99, 1.0, 0.98], [0.0, -0.88, -0.25], 14000),
            ("back_center", [0.96, 0.97, 1.0], [0.0, -0.3, 0.88], 19000),
            ("back_right", [0.97, 0.98, 1.0], [0.55, -0.3, 0.65], 16000),
            ("back_left", [1.0, 0.98, 0.96], [-0.55, -0.35, 0.65], 16000),
        ],
        29000
    ),

    4: (  # Geometry-extraction: stronger key, lower ambient for shadow detail
        [
            ("key_light", [1.0, 0.99, 0.97], [-0.5, -0.65, -0.6], 65000),
            ("fill_right", [0.96, 0.97, 1.0], [0.6, -0.35, -0.5], 12000),
            ("top", [1.0, 1.0, 0.98], [0.0, -0.92, -0.1], 12000),
            ("rim_back_right", [0.96, 0.97, 1.0], [0.6, -0.2, 0.7], 14000),
            ("rim_back_left", [1.0, 0.97, 0.96], [-0.6, -0.2, 0.7], 14000),
            ("bottom_bounce", [0.97, 0.98, 1.0], [0.0, 0.7, -0.3], 6000),
        ],
        8000  # Low indirect to preserve shadow detail for geometry
    ),

    5: (  # Natural sunlight: warm direct sun with sky-blue ambient fill
        [
            # Direct sun – warm white from upper-right at ~55° elevation
            ("sun", [1.0, 0.96, 0.90], [0.4, -0.82, -0.4], 62000),
            # Sky fill – cool blue diffuse from above (simulates sky dome)
            ("sky_fill", [0.85, 0.92, 1.0], [0.0, -0.95, 0.0], 18000),
            # Bounce fill – warm ground-reflected light from below
            ("ground_bounce", [1.0, 0.95, 0.85], [0.0, 0.6, -0.3], 8000),
            # Secondary fill – softer blue from opposite side for shadow detail
            ("sky_fill_left", [0.88, 0.93, 1.0], [-0.6, -0.5, -0.5], 12000),
            # Back rim – slight warm highlight from behind for edge separation
            ("back_rim", [1.0, 0.96, 0.88], [0.0, -0.25, 0.9], 14000),
            # Side kick – gentle fill from right for 3/4 views
            ("side_kick_right", [0.95, 0.97, 1.0], [0.7, -0.35, -0.5], 10000),
        ],
        22000  # Moderate ambient for natural outdoor feel
    ),
}


# =============================================================================
# APPEARANCE VARIATION GROUPS (0-9)
# =============================================================================
APPEARANCE_GROUPS = {
    0: (  # Bright front lighting (midday sun)
        [
            ("key_light", [1.0, 0.98, 0.95], [-0.2, -0.8, -0.6], 75000),
            ("fill_1", [0.98, 0.98, 1.0], [0.5, -0.5, -0.5], 18000),
            ("fill_2", [1.0, 0.98, 0.98], [-0.5, -0.4, -0.5], 15000),
            ("back", [0.95, 0.98, 1.0], [0.0, -0.3, 0.8], 20000),
        ],
        28000
    ),
    
    1: (  # Warm side lighting (morning/evening)
        [
            ("key_light", [1.0, 0.95, 0.85], [0.7, -0.6, -0.5], 70000),
            ("fill_1", [0.9, 0.95, 1.0], [-0.6, -0.5, -0.5], 12000),
            ("fill_2", [0.95, 0.95, 0.95], [0.0, -0.7, -0.4], 10000),
            ("back", [0.85, 0.9, 1.0], [-0.5, -0.3, 0.7], 25000),
        ],
        22000
    ),
    
    2: (  # Cool diffuse lighting (overcast)
        [
            ("key_light", [0.95, 0.98, 1.0], [-0.4, -0.7, -0.7], 55000),
            ("fill_1", [0.98, 0.98, 1.0], [0.6, -0.5, -0.4], 20000),
            ("fill_2", [0.98, 1.0, 1.0], [-0.5, -0.5, -0.5], 18000),
            ("back", [0.95, 0.98, 1.0], [0.0, -0.2, 0.9], 28000),
        ],
        35000
    ),
    
    3: (  # Dramatic side/back lighting (golden hour)
        [
            ("key_light", [1.0, 0.90, 0.75], [0.8, -0.5, 0.3], 80000),
            ("fill_1", [0.85, 0.90, 1.0], [-0.4, -0.6, -0.6], 15000),
            ("fill_2", [0.95, 0.95, 1.0], [0.0, -0.8, -0.3], 12000),
            ("back", [1.0, 0.95, 0.85], [0.3, -0.3, 0.8], 30000),
        ],
        20000
    ),
    
    4: (  # Balanced neutral (studio lighting)
        [
            ("key_light", [1.0, 1.0, 1.0], [-0.3, -0.7, -0.8], 65000),
            ("fill_1", [0.98, 0.98, 1.0], [0.6, -0.5, -0.5], 15000),
            ("fill_2", [1.0, 0.98, 0.98], [-0.5, -0.5, -0.6], 13000),
            ("back", [0.98, 0.98, 1.0], [0.0, -0.4, 0.8], 24000),
        ],
        26000
    ),
    
    5: (  # Strong top-down (overhead sun)
        [
            ("key_light", [1.0, 0.98, 0.96], [0.0, -0.9, -0.4], 70000),
            ("fill_1", [0.96, 0.98, 1.0], [0.5, -0.4, -0.6], 16000),
            ("fill_2", [0.98, 1.0, 1.0], [-0.5, -0.4, -0.6], 14000),
            ("back", [0.95, 0.96, 1.0], [0.0, -0.2, 0.9], 26000),
        ],
        24000
    ),
    
    6: (  # Soft diffuse (cloudy bright)
        [
            ("key_light", [0.96, 0.98, 1.0], [-0.2, -0.6, -0.8], 58000),
            ("fill_1", [0.98, 1.0, 1.0], [0.6, -0.5, -0.5], 22000),
            ("fill_2", [1.0, 0.98, 0.98], [-0.6, -0.5, -0.5], 20000),
            ("back", [0.96, 0.98, 1.0], [0.0, -0.3, 0.85], 30000),
        ],
        32000
    ),
    
    7: (  # Warm rim lighting (sunset)
        [
            ("key_light", [1.0, 0.88, 0.70], [0.6, -0.4, 0.6], 75000),
            ("fill_1", [0.90, 0.95, 1.0], [-0.5, -0.6, -0.5], 14000),
            ("fill_2", [0.95, 0.98, 1.0], [0.0, -0.7, -0.4], 11000),
            ("back", [1.0, 0.92, 0.78], [-0.4, -0.3, 0.8], 28000),
        ],
        21000
    ),
    
    8: (  # Cool side lighting (twilight)
        [
            ("key_light", [0.85, 0.90, 1.0], [0.7, -0.6, -0.4], 62000),
            ("fill_1", [0.95, 0.98, 1.0], [-0.6, -0.5, -0.6], 17000),
            ("fill_2", [0.98, 0.98, 1.0], [0.0, -0.8, -0.3], 13000),
            ("back", [0.88, 0.92, 1.0], [0.0, -0.2, 0.9], 27000),
        ],
        29000
    ),
    
    9: (  # Dramatic contrast (low-key lighting)
        [
            ("key_light", [1.0, 0.95, 0.90], [0.8, -0.6, -0.2], 85000),
            ("fill_1", [0.80, 0.85, 1.0], [-0.5, -0.6, -0.5], 10000),
            ("fill_2", [0.90, 0.90, 0.95], [0.0, -0.7, -0.5], 8000),
            ("back", [1.0, 0.90, 0.80], [0.4, -0.2, 0.8], 25000),
        ],
        18000
    ),
}
