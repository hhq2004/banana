from pptx.enum.shapes import MSO_SHAPE
from typing import Dict, Optional, Tuple
from .data import ConversionConfig, default_config

# Mapping dictionary: draw.io shape type -> PowerPoint MSO_SHAPE
# Multiple draw.io shape types can map to the same PowerPoint shape
SHAPE_TYPE_MAP: dict[str, MSO_SHAPE] = {
    'ellipse': MSO_SHAPE.OVAL,
    'circle': MSO_SHAPE.OVAL,
    'rect': MSO_SHAPE.RECTANGLE,
    'rectangle': MSO_SHAPE.RECTANGLE,
    'rounded_rectangle': MSO_SHAPE.ROUNDED_RECTANGLE,
    'diamond': MSO_SHAPE.DIAMOND,
    'cloud': MSO_SHAPE.CLOUD,
    'hexagon': MSO_SHAPE.HEXAGON,
    'square': MSO_SHAPE.RECTANGLE,
    'rhombus': MSO_SHAPE.DIAMOND,
    'parallelogram': MSO_SHAPE.PARALLELOGRAM,
    'trapezoid': MSO_SHAPE.TRAPEZOID,
    'cylinder3': MSO_SHAPE.CAN,
    'cylinder': MSO_SHAPE.CAN,
    'document': MSO_SHAPE.FLOWCHART_DOCUMENT,
    'tape': MSO_SHAPE.FLOWCHART_PUNCHED_TAPE,
    'datastorage': MSO_SHAPE.FLOWCHART_STORED_DATA,
    'data_storage': MSO_SHAPE.FLOWCHART_STORED_DATA,
    'data-storage': MSO_SHAPE.FLOWCHART_STORED_DATA,
    'internalstorage': MSO_SHAPE.FLOWCHART_INTERNAL_STORAGE,
    'internal_storage': MSO_SHAPE.FLOWCHART_INTERNAL_STORAGE,
    'internal-storage': MSO_SHAPE.FLOWCHART_INTERNAL_STORAGE,
    'process': MSO_SHAPE.FLOWCHART_PROCESS,
    'predefinedprocess': MSO_SHAPE.FLOWCHART_PREDEFINED_PROCESS,
    'predefined_process': MSO_SHAPE.FLOWCHART_PREDEFINED_PROCESS,
    'predefined-process': MSO_SHAPE.FLOWCHART_PREDEFINED_PROCESS,
    'hexagon': MSO_SHAPE.HEXAGON,
    'pentagon': MSO_SHAPE.REGULAR_PENTAGON,
    'octagon': MSO_SHAPE.OCTAGON,
    'isosceles_triangle': MSO_SHAPE.ISOSCELES_TRIANGLE,
    'right_triangle': MSO_SHAPE.RIGHT_TRIANGLE,
    '4_point_star': MSO_SHAPE.STAR_4_POINT,
    '5_point_star': MSO_SHAPE.STAR_5_POINT,
    '6_point_star': MSO_SHAPE.STAR_6_POINT,
    '8_point_star': MSO_SHAPE.STAR_8_POINT,
    'smiley': MSO_SHAPE.SMILEY_FACE,
}

# Dash pattern mapping (draw.io → DrawingML prstDash)
DASH_PATTERN_MAP: Dict[str, str] = {
    'dashed': 'dash',
    'dash': 'dash',
    'dotted': 'dot',
    'dot': 'dot',
    'dashDot': 'dashDot',
    'dashdot': 'dashDot',
    'dashDotDot': 'lgDashDotDot',
    'dashdotdot': 'lgDashDotDot',
    'longDash': 'lgDash',
    'longdash': 'lgDash',
    'longDashDot': 'lgDashDot',
    'longdashdot': 'lgDashDot',
    'solid': 'solid',
    'none': 'solid',
}


# Arrow type mapping (draw.io → PowerPoint)
ARROW_TYPE_MAP: Dict[str, Tuple[str, str, str]] = {
    # (type, w, len) format
    'classic': ('triangle', 'med', 'med'),
    'block': ('triangle', 'med', 'med'),
    'classicthin': ('triangle', 'sm', 'sm'),
    'blockthin': ('triangle', 'sm', 'sm'),
    # PowerPoint supports an "arrow" line-end type which corresponds to an open arrow head.
    'open': ('arrow', 'med', 'med'),
    'openthin': ('arrow', 'sm', 'sm'),
    'oval': ('oval', 'med', 'med'),
    'diamond': ('diamond', 'med', 'med'),
    'diamondthin': ('diamond', 'sm', 'sm'),
    'cross': ('stealth', 'med', 'med'),
    'crossthin': ('stealth', 'sm', 'sm'),
    'dash': ('triangle', 'sm', 'sm'),
    'dashthin': ('triangle', 'sm', 'sm'),
    'line': ('triangle', 'sm', 'sm'),
    'linethin': ('triangle', 'sm', 'sm'),
    'none': None,
}

DRAWIO_DEFAULT_FONT_FAMILY = "Helvetica"
# EMU conversion constants (conversion from screen pixels to EMU)
EMU_PER_PX = 9525
PT_PER_PX = 0.75  # Assumption: for 96 DPI

# Default skew for parallelogram (MSO_SHAPE.PARALLELOGRAM) (approximately 0.0-0.5)
# Used when converting draw.io → PPTX to match shape appearance with connection point calculation.
PARALLELOGRAM_SKEW: float = 0.2

def map_shape_type_to_pptx(shape_type: str) -> MSO_SHAPE:
    """
    Map draw.io shape type to PowerPoint shape type
    
    Args:
        shape_type: draw.io shape type ('rectangle', 'ellipse', etc.)
    
    Returns:
        MSO_SHAPE enumeration value
    """
    shape_type_lower = (shape_type or "").lower()
    return SHAPE_TYPE_MAP.get(shape_type_lower, MSO_SHAPE.RECTANGLE)


def map_dash_pattern(drawio_dash: Optional[str]) -> Optional[str]:
    """
    Map dash pattern
    
    Args:
        drawio_dash: draw.io dash pattern name
    
    Returns:
        DrawingML prstDash value, or None
    """
    if not drawio_dash:
        return None
    
    return DASH_PATTERN_MAP.get(drawio_dash.lower(), None)


def map_arrow_type(drawio_arrow: Optional[str]) -> Optional[Tuple[str, str, str]]:
    """
    Map arrow type
    
    Args:
        drawio_arrow: draw.io arrow type name
    
    Returns:
        (type, w, len) tuple, or None
    """
    if not drawio_arrow or drawio_arrow.lower() == "none":
        return None
    
    return ARROW_TYPE_MAP.get(drawio_arrow.lower())


def map_arrow_size_px_to_pptx(size_px: Optional[float]) -> Optional[str]:
    """
    Map draw.io marker size (startSize/endSize) in px to PowerPoint's discrete arrow size.

    PowerPoint supports only 'sm' / 'med' / 'lg' for line-end width/length.
    draw.io exposes a numeric marker size (often 6 by default). We approximate by thresholds.
    """
    if size_px is None:
        return None
    try:
        v = float(size_px)
    except Exception:
        return None
    if v <= 6.5:
        return "sm"
    if v <= 10.5:
        return "med"
    return "lg"


def map_arrow_type_with_size(drawio_arrow: Optional[str], size_px: Optional[float]) -> Optional[Tuple[str, str, str]]:
    """
    Map arrow type, optionally overriding (w, len) based on draw.io startSize/endSize.
    """
    base = map_arrow_type(drawio_arrow)
    if not base:
        return None
    override = map_arrow_size_px_to_pptx(size_px)
    if not override:
        return base
    arrow_type, _w, _len = base
    return (arrow_type, override, override)


# Corner radius approximation (rounded → arcSize)
# arcSize is a value from 0-100000, where 100000 is a complete circle
def rounded_to_arc_size(rounded: bool, width: float, height: float) -> Optional[int]:
    """
    Calculate arcSize from rounded flag and size
    
    Args:
        rounded: Whether corner radius is enabled
        width: Shape width (px)
        height: Shape height (px)
    
    Returns:
        arcSize value (0-100000), or None (no corner radius)
    """
    if not rounded:
        return None
    
    # Assume default corner radius (approximately 10% of size)
    radius = min(width, height) * 0.1
    
    # arcSize = (radius / min(width, height)) * 100000
    if min(width, height) > 0:
        arc_size = int((radius / min(width, height)) * 100000)
        return min(arc_size, 100000)
    
    return 50000  # Default value


def map_corner_radius(rounded: bool, width: float, height: float) -> Optional[int]:
    """
    Map corner radius
    
    Args:
        rounded: Whether corner radius is enabled
        width: Shape width (px)
        height: Shape height (px)
    
    Returns:
        arcSize value (0-100000), or None
    """
    return rounded_to_arc_size(rounded, width, height)

def validate_font(font_family: Optional[str]) -> bool:
    """
    Validate if font is available
    
    Args:
        font_family: Font family name
    
    Returns:
        True if available
    """
    # Simple implementation: always return True
    # In actual implementation, check system font list
    return True


def replace_font(font_family: Optional[str], config: Optional[ConversionConfig] = None) -> Optional[str]:
    """
    Replace font (based on configuration)
    
    Args:
        font_family: Original font family name
        config: ConversionConfig instance (uses default_config if None)
    
    Returns:
        Replaced font family name
    """
    if not font_family:
        return None
    
    # Get replacement map from configuration
    config_to_use = config or default_config
    replacements = config_to_use.font_replacements
    
    if font_family in replacements:
        return replacements[font_family]
    
    return font_family