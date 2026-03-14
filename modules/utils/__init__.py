"""
工具函数模块

包含各处理模块共用的工具函数
"""

from .color_utils import (
    extract_fill_stroke_colors,
    rgb_to_hex,
    hex_to_rgb,
    get_dominant_color,
)

from .xml_utils import (
    create_mxcell,
    create_geometry,
    prettify_xml,
    parse_drawio_xml,
)

from .image_utils import (
    image_to_base64,
    base64_to_image,
    crop_with_padding,
    calculate_iou,
)

from .overlap_utils import (
    detect_overlaps,
    analyze_overlap_relationships,
    resolve_overlapping_elements,
    group_overlapping_elements,
    suggest_xml_structure,
)

from .drawio_library import (
    # 类
    DrawIOLibrary,
    ArrowAttributeDetector,
    # 函数
    match_element_to_drawio,
    detect_arrow_style,
    detect_arrow_attributes,
    get_drawio_style,
    build_style_string,
    build_arrow_style,
    get_all_arrow_head_types,
    get_all_dash_patterns,
    get_all_edge_styles,
    # 常量
    DRAWIO_BASIC_SHAPES,
    DRAWIO_ARROWS,
    DRAWIO_ARROW_HEADS,
    DRAWIO_EDGE_STYLES,
    DRAWIO_DASH_PATTERNS,
    DRAWIO_ARROW_SHAPES,
)

__all__ = [
    # 颜色工具
    'extract_fill_stroke_colors',
    'rgb_to_hex',
    'hex_to_rgb',
    'get_dominant_color',
    
    # XML工具
    'create_mxcell',
    'create_geometry',
    'prettify_xml',
    'parse_drawio_xml',
    
    # 图像工具
    'image_to_base64',
    'base64_to_image',
    'crop_with_padding',
    'calculate_iou',
    
    # 重叠检测工具
    'detect_overlaps',
    'analyze_overlap_relationships',
    'resolve_overlapping_elements',
    'group_overlapping_elements',
    'suggest_xml_structure',
    
    # DrawIO库支持
    'DrawIOLibrary',
    'ArrowAttributeDetector',
    'match_element_to_drawio',
    'detect_arrow_style',
    'detect_arrow_attributes',
    'get_drawio_style',
    'build_style_string',
    'build_arrow_style',
    'get_all_arrow_head_types',
    'get_all_dash_patterns',
    'get_all_edge_styles',
    'DRAWIO_BASIC_SHAPES',
    'DRAWIO_ARROWS',
    'DRAWIO_ARROW_HEADS',
    'DRAWIO_EDGE_STYLES',
    'DRAWIO_DASH_PATTERNS',
    'DRAWIO_ARROW_SHAPES',
]
