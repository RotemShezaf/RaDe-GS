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
                 0-4 = fixed light ID presets
                 0-9 = decoupled appearance groups (when use_decoupled_appearance=True)
        use_decoupled_appearance: If True, uses appearance variation mode
    
    Returns:
        Tuple of (light_configs, indirect_intensity)
        where light_configs is a list of (name_suffix, color, direction, intensity)
    """
    if group_id == -1:
        # DEFAULT/STANDARD MODE: Balanced lighting for clear shading
        return STANDARD_LIGHTING
    
    elif group_id in [0, 1, 2, 3, 4] and not use_decoupled_appearance:
        # FIXED LIGHT ID MODE: 5 preset configurations
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
# =============================================================================
STANDARD_LIGHTING = (
    [
        # Key light - strong directional from front-top-right
        ("key_light", [1.0, 1.0, 1.0], [-0.3, -0.7, -0.8], 65000),
        
        # Secondary lights - 6 directions for coverage
        ("front_right", [0.98, 0.98, 1.0], [0.7, -0.4, -0.7], 12000),
        ("front_left", [1.0, 0.98, 0.98], [-0.7, -0.4, -0.7], 12000),
        ("top", [1.0, 1.0, 0.98], [0.0, -0.9, -0.3], 10000),
        ("back_center", [0.96, 0.98, 1.0], [0.0, -0.3, 0.9], 22000),
        ("back_right", [0.98, 0.98, 1.0], [0.6, -0.3, 0.6], 16000),
        ("back_left", [1.0, 0.98, 0.98], [-0.6, -0.3, 0.6], 16000),
    ],
    25000  # Indirect intensity
)


# =============================================================================
# FIXED LIGHT ID PRESETS (0-4)
# =============================================================================
LIGHT_ID_PRESETS = {
    0: STANDARD_LIGHTING,  # Light ID 0 = standard
    
    1: (  # Warmer variant with slightly different key angle
        [
            ("key_light", [1.0, 0.98, 0.93], [-0.4, -0.6, -0.7], 68000),
            ("front_right", [1.0, 0.98, 0.95], [0.6, -0.5, -0.6], 13000),
            ("front_left", [0.98, 0.98, 1.0], [-0.6, -0.5, -0.7], 11000),
            ("top", [1.0, 0.99, 0.96], [0.0, -0.8, -0.4], 11000),
            ("back_center", [0.95, 0.97, 1.0], [0.0, -0.4, 0.85], 23000),
            ("back_right", [0.97, 0.98, 1.0], [0.5, -0.4, 0.7], 17000),
            ("back_left", [1.0, 0.97, 0.95], [-0.5, -0.4, 0.7], 15000),
        ],
        27000
    ),
    
    2: (  # Cooler variant with top-down emphasis
        [
            ("key_light", [0.95, 0.98, 1.0], [-0.2, -0.8, -0.6], 62000),
            ("front_right", [0.96, 0.98, 1.0], [0.7, -0.4, -0.6], 14000),
            ("front_left", [0.98, 1.0, 1.0], [-0.7, -0.4, -0.6], 13000),
            ("top", [0.96, 0.99, 1.0], [0.0, -0.9, -0.2], 12000),
            ("back_center", [0.94, 0.97, 1.0], [0.0, -0.3, 0.9], 24000),
            ("back_right", [0.96, 0.98, 1.0], [0.6, -0.3, 0.7], 18000),
            ("back_left", [0.98, 0.98, 1.0], [-0.6, -0.3, 0.7], 17000),
        ],
        29000
    ),
    
    3: (  # Softer lighting with more ambient
        [
            ("key_light", [1.0, 1.0, 0.98], [-0.3, -0.7, -0.8], 58000),
            ("front_right", [0.98, 0.98, 1.0], [0.7, -0.4, -0.7], 15000),
            ("front_left", [1.0, 0.98, 0.98], [-0.7, -0.4, -0.7], 15000),
            ("top", [0.99, 1.0, 0.98], [0.0, -0.9, -0.3], 13000),
            ("back_center", [0.97, 0.98, 1.0], [0.0, -0.3, 0.9], 25000),
            ("back_right", [0.98, 0.98, 1.0], [0.6, -0.3, 0.6], 19000),
            ("back_left", [1.0, 0.98, 0.98], [-0.6, -0.3, 0.6], 18000),
        ],
        32000
    ),
    
    4: (  # High-contrast dramatic lighting
        [
            ("key_light", [1.0, 0.99, 0.96], [-0.5, -0.6, -0.7], 72000),
            ("front_right", [0.98, 0.98, 1.0], [0.6, -0.5, -0.7], 10000),
            ("front_left", [1.0, 0.98, 0.96], [-0.6, -0.5, -0.8], 9000),
            ("top", [1.0, 1.0, 0.97], [0.0, -0.9, -0.4], 8000),
            ("back_center", [0.95, 0.97, 1.0], [0.0, -0.2, 0.9], 20000),
            ("back_right", [0.97, 0.98, 1.0], [0.7, -0.2, 0.5], 14000),
            ("back_left", [1.0, 0.97, 0.95], [-0.7, -0.2, 0.5], 13000),
        ],
        22000
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
