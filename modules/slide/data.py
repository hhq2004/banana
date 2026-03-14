"""
Intermediate model definitions

Defines normalized intermediate representation extracted from draw.io's mxGraph
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, List, Tuple
from pptx.dml.color import RGBColor
import logging

@dataclass
class ConversionConfig:
    """Conversion configuration"""
    # Scale policy: 'fit-to-width', 'fit-to-height', 'contain', 'cover', 'none'
    scale_policy: str = 'none'
    
    # Margin (px)
    margin_x: float = 0.0
    margin_y: float = 0.0
    
    # DPI setting
    dpi: float = 96.0
    
    # Font replacement map
    font_replacements: Dict[str, str] = None
    
    # Whether to warn about unsupported effects
    warn_unsupported_effects: bool = True
    
    # Tolerance for coordinate errors (px)
    coordinate_tolerance: float = 0.1
    
    def __post_init__(self):
        if self.font_replacements is None:
            self.font_replacements = {}

# Default configuration instance
default_config = ConversionConfig()

@dataclass
class ConversionWarning:
    """Warning during conversion"""
    element_id: Optional[str]
    warning_type: str  # 'unsupported_effect', 'coordinate_error', 'font_missing', etc.
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


class ConversionLogger:
    """Logger for conversion process"""
    
    def __init__(self, warn_unsupported: bool = None, config: Optional[ConversionConfig] = None):
        """
        Args:
            warn_unsupported: Whether to warn about unsupported effects (deprecated: use config instead)
            config: ConversionConfig instance (uses default_config if None)
        """
        self.config = config or default_config
        
        # Use config.warn_unsupported_effects if warn_unsupported is not explicitly provided
        if warn_unsupported is None:
            self.warn_unsupported = self.config.warn_unsupported_effects
        else:
            self.warn_unsupported = warn_unsupported
        
        self.warnings: List[ConversionWarning] = []
        self.logger = logging.getLogger('slide')
        
        # Logger configuration (default)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def warn_unsupported_effect(self, element_id: Optional[str], effect_type: str, details: Dict[str, Any] = None):
        """Record warning for unsupported effect"""
        if not self.warn_unsupported:
            return
        
        message = f"Unsupported effect: {effect_type}"
        warning = ConversionWarning(
            element_id=element_id,
            warning_type='unsupported_effect',
            message=message,
            details=details or {}
        )
        self.warnings.append(warning)
        self.logger.warning(f"[{element_id}] {message}")
    
    def warn_coordinate_error(self, element_id: Optional[str], expected: tuple, actual: tuple, tolerance: Optional[float] = None):
        """Record warning for coordinate error"""
        # Use config.coordinate_tolerance if tolerance is not provided
        if tolerance is None:
            tolerance = self.config.coordinate_tolerance
        
        message = f"Coordinate error: expected {expected}, got {actual} (tolerance: {tolerance})"
        warning = ConversionWarning(
            element_id=element_id,
            warning_type='coordinate_error',
            message=message,
            details={
                'expected': expected,
                'actual': actual,
                'tolerance': tolerance
            }
        )
        self.warnings.append(warning)
        self.logger.warning(f"[{element_id}] {message}")
    
    def warn_font_missing(self, element_id: Optional[str], font_family: str, replacement: str = None):
        """Record warning for missing font"""
        message = f"Font not found: {font_family}"
        if replacement:
            message += f" (replaced with {replacement})"
        
        warning = ConversionWarning(
            element_id=element_id,
            warning_type='font_missing',
            message=message,
            details={
                'font_family': font_family,
                'replacement': replacement
            }
        )
        self.warnings.append(warning)
        self.logger.warning(f"[{element_id}] {message}")
    
    def info(self, message: str):
        """Info log"""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Debug log"""
        self.logger.debug(message)
    
    def error(self, message: str):
        """Error log"""
        self.logger.error(message)
    
    def get_warnings(self) -> List[ConversionWarning]:
        """Get warning list"""
        return self.warnings
    
    def clear_warnings(self):
        """Clear warning list"""
        self.warnings.clear()


# Global logger instance
_default_logger = ConversionLogger()


def get_logger() -> ConversionLogger:
    """Get default logger"""
    return _default_logger



@dataclass
class TextRun:
    """Text run (inline formatting unit)"""
    text: str
    font_family: Optional[str] = None
    font_size: Optional[float] = None  # pt
    font_color: Optional[RGBColor] = None
    bold: bool = False
    italic: bool = False
    underline: bool = False
    link: Optional[str] = None  # Hyperlink URL
    is_latex: bool = False  # 新增：是否为 LaTeX 公式
    latex_content: Optional[str] = None  # 新增：原始 LaTeX 内容


@dataclass
class TextParagraph:
    """Text paragraph"""
    runs: List[TextRun] = field(default_factory=list)
    align: Optional[str] = None  # 'left', 'center', 'right'
    vertical_align: Optional[str] = None  # 'top', 'middle', 'bottom'
    spacing_top: Optional[float] = None  # px
    spacing_left: Optional[float] = None  # px
    spacing_bottom: Optional[float] = None  # px
    spacing_right: Optional[float] = None  # px
    line_spacing: Optional[float] = None
    bullet_style: Optional[str] = None  # Bullet style


@dataclass
class Transform:
    """Transformation information (rotation, scale, flip)"""
    rotation: float = 0.0  # degrees
    scale_x: float = 1.0
    scale_y: float = 1.0
    flip_h: bool = False
    flip_v: bool = False
    translate_x: float = 0.0  # px
    translate_y: float = 0.0  # px


@dataclass
class Style:
    """Style information"""
    fill: Optional[Union[RGBColor, str]] = None  # RGBColor, "default", or None
    # draw.io gradient (style keys: gradientColor, gradientDirection). When present, PPTX should use <a:gradFill>.
    # gradient_color accepts RGBColor, "default"/"auto" (theme-like), or None (no gradient).
    gradient_color: Optional[Union[RGBColor, str]] = None
    gradient_direction: Optional[str] = None
    stroke: Optional[RGBColor] = None
    stroke_width: float = 1.0  # px
    dash: Optional[str] = None  # Dash pattern
    opacity: float = 1.0
    corner_radius: Optional[float] = None  # px
    # Text label background (draw.io: labelBackgroundColor). Best-effort mapping to PPTX highlight.
    label_background_color: Optional[RGBColor] = None
    arrow_start: Optional[str] = None  # Arrow type
    arrow_end: Optional[str] = None
    arrow_start_fill: bool = True
    arrow_end_fill: bool = True
    # draw.io style keys: startSize/endSize (unit: px in mxGraph style)
    arrow_start_size_px: Optional[float] = None
    arrow_end_size_px: Optional[float] = None
    has_shadow: bool = False  # Whether shadow is present
    # Text wrapping (draw.io: whiteSpace=wrap/nowrap). True for wrap, False for nowrap.
    # Default is True (wrap) to match draw.io's default behavior.
    word_wrap: bool = True
    # Swimlane/container metadata (draw.io: swimlane; startSize; horizontal)
    is_swimlane: bool = False
    swimlane_start_size: Optional[float] = None  # px
    swimlane_horizontal: Optional[bool] = None
    swimlane_line: Optional[bool] = None
    # Explicitly disable stroke (e.g. draw.io text shapes, strokeColor=none)
    no_stroke: bool = False


@dataclass
class ImageData:
    """Image data"""
    data_uri: Optional[str] = None  # data URI
    file_path: Optional[str] = None  # External file path
    base64_data: Optional[str] = None  # Base64 encoded image data
    format: Optional[str] = None  # Image format (png, jpg, etc.)
    dpi: float = 96.0
    transparent: bool = False


@dataclass
class BaseElement:
    """Base element"""
    id: Optional[str] = None
    x: float = 0.0  # px
    y: float = 0.0  # px
    w: float = 0.0  # px
    h: float = 0.0  # px
    z_index: int = 0
    group_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    transform: Transform = field(default_factory=Transform)
    style: Style = field(default_factory=Style)


@dataclass
class ShapeElement(BaseElement):
    """Shape element"""
    element_type: str = 'shape'  # 'shape', 'rectangle', 'ellipse', 'polygon', etc.
    text: List[TextParagraph] = field(default_factory=list)
    image: Optional[ImageData] = None
    shape_type: str = 'rectangle'  # draw.io shape type
    parent_id: Optional[str] = None  # Parent shape ID (for container children)


@dataclass
class ConnectorElement(BaseElement):
    """Connector element"""
    element_type: str = 'connector'
    source_id: Optional[str] = None
    target_id: Optional[str] = None
    points: List[Tuple[float, float]] = field(default_factory=list)  # [(x, y), ...]
    edge_style: str = 'straight'  # 'straight', 'orthogonal', 'curved'


@dataclass
class TextElement(BaseElement):
    """Text element (standalone text)"""
    element_type: str = 'text'
    text: List[TextParagraph] = field(default_factory=list)


@dataclass
class ImageElement(BaseElement):
    """Image element"""
    element_type: str = 'image'
    image: ImageData = field(default_factory=ImageData)


@dataclass
class GroupElement(BaseElement):
    """Group element"""
    element_type: str = 'group'
    children: List[BaseElement] = field(default_factory=list)


@dataclass
class PolygonElement(BaseElement):
    """Polygon element"""
    element_type: str = 'polygon'
    points: List[Tuple[float, float]] = field(default_factory=list)  # Vertex coordinates


@dataclass
class PathElement(BaseElement):
    """Path element (Bezier curves, etc.)"""
    element_type: str = 'path'
    path_data: str = ""  # SVG path data (M, L, C, Q, etc.)






# Union of element types
Element = Union[
    ShapeElement,
    ConnectorElement,
    TextElement,
    ImageElement,
    GroupElement,
    PolygonElement,
    PathElement,
]
