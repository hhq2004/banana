"""
Slide 模块 - Draw.io 到 PowerPoint 转换

提供 Draw.io XML 文件到 PowerPoint 演示文稿的转换功能
"""

# 核心转换类
from .xml2pptx import DrawioToPptxConverter

# 数据模型
from .data import (
    # 配置和日志
    ConversionConfig,
    ConversionLogger,
    ConversionWarning,
    default_config,
    get_logger,
    
    # 数据结构
    Transform,
    Style,
    TextRun,
    TextParagraph,
    ImageData,
    
    # 元素类型
    BaseElement,
    ShapeElement,
    ConnectorElement,
    TextElement,
    ImageElement,
    GroupElement,
    PolygonElement,
    PathElement,
)

# 加载器和写入器
from .load import (
    DrawIOLoader,
    ColorParser,
    StyleExtractor,
)

from .draw import PPTXWriter

# 映射和转换工具
from .map import (
    # 形状映射
    SHAPE_TYPE_MAP,
    DASH_PATTERN_MAP,
    ARROW_TYPE_MAP,
    
    # 常量
    DRAWIO_DEFAULT_FONT_FAMILY,
    PARALLELOGRAM_SKEW,
    EMU_PER_PX,
    PT_PER_PX,
    
    # 映射函数
    map_shape_type_to_pptx,
    map_dash_pattern,
    map_arrow_type,
    map_arrow_type_with_size,
    map_arrow_size_px_to_pptx,
    rounded_to_arc_size,
    map_corner_radius,
    validate_font,
    replace_font,
)

# 坐标转换工具
from .transform import (
    # 单位转换
    px_to_emu,
    px_to_pt,
    pt_to_emu,
    emu_to_px,
    emu_to_pt,
    pt_to_px,
    scale_font_size_for_pptx,
    
    # 几何变换
    apply_transform,
    apply_group_transform,
    calculate_bounding_box,
    calculate_rotated_bounding_box,
    split_polyline_to_segments,
    catmull_rom_to_bezier,
)



__all__ = [
    # 核心转换类
    'DrawioToPptxConverter',
    
    # 配置和日志
    'ConversionConfig',
    'ConversionLogger',
    'ConversionWarning',
    'default_config',
    'get_logger',
    
    # 数据结构
    'Transform',
    'Style',
    'TextRun',
    'TextParagraph',
    'ImageData',
    
    # 元素类型
    'BaseElement',
    'ShapeElement',
    'ConnectorElement',
    'TextElement',
    'ImageElement',
    'GroupElement',
    'PolygonElement',
    'PathElement',
    
    # 加载器和写入器
    'DrawIOLoader',
    'ColorParser',
    'StyleExtractor',
    'PPTXWriter',
    
    # 形状映射常量
    'SHAPE_TYPE_MAP',
    'DASH_PATTERN_MAP',
    'ARROW_TYPE_MAP',
    'DRAWIO_DEFAULT_FONT_FAMILY',
    'PARALLELOGRAM_SKEW',
    'EMU_PER_PX',
    'PT_PER_PX',
    
    # 映射函数
    'map_shape_type_to_pptx',
    'map_dash_pattern',
    'map_arrow_type',
    'map_arrow_type_with_size',
    'map_arrow_size_px_to_pptx',
    'rounded_to_arc_size',
    'map_corner_radius',
    'validate_font',
    'replace_font',
    
    # 单位转换
    'px_to_emu',
    'px_to_pt',
    'pt_to_emu',
    'emu_to_px',
    'emu_to_pt',
    'pt_to_px',
    'scale_font_size_for_pptx',
    
    # 几何变换
    'apply_transform',
    'apply_group_transform',
    'calculate_bounding_box',
    'calculate_rotated_bounding_box',
    'split_polyline_to_segments',
    'catmull_rom_to_bezier',
    
]

__version__ = '1.0.0'