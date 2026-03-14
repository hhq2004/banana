"""
图片转可编辑形式的模块化处理框架

模块划分:
    1. sam3_info_extractor   - SAM3提取结构化信息
    2. icon_picture_processor - Icon/Picture非基本图形处理（转base64，生成XML）
    3. basic_shape_processor  - 基本图形处理（取色、生成XML）
    4. other_shape_processor  - 其他图形处理逻辑
    5. arrow_processor        - 箭头处理（生成XML）
    6. xml_merger            - XML合并（只负责收集和排序，不生成样式）
    7. metric_evaluator      - 质量评估
    8. refinement_processor  - 表现不好的区域二次处理
    
    文字处理（另一个小组）:
    - text_processor         - OCR文字识别与XML生成（接口占位）

重构说明：
    - 每个子模块负责生成自己的mxCell XML代码
    - 子模块设置 element.xml_fragment 和 element.layer_level
    - XMLMerger只负责收集、排序、合并，不负责生成样式

使用方式:
    from modules import Sam3InfoExtractor, XMLMerger, TextProcessor
    from modules.data_types import ElementInfo, XMLFragment, LayerLevel
"""

from .base import BaseProcessor, ProcessingContext
from .data_types import (
    ElementInfo, 
    BoundingBox, 
    ProcessingResult, 
    XMLFragment,
    LayerLevel,
    get_layer_level,
)
from .sam3_info_extractor import Sam3InfoExtractor
from .xml_merger import XMLMerger, merge_fragments, merge_shape_and_text

# 图形处理模块（占位）
from .icon_picture_processor import IconPictureProcessor
from .basic_shape_processor import BasicShapeProcessor
from .other_shape_processor import OtherShapeProcessor
from .arrow_processor import ArrowProcessor
from .metric_evaluator import MetricEvaluator
from .refinement_processor import RefinementProcessor
from .enhanced_frame_detector import EnhancedFrameDetector, enhance_frame_detection

# 文字处理模块（另一个小组的接口）
from .text_processor import TextProcessor, create_text_fragments, create_single_text_xml

__all__ = [
    # 基础类
    'BaseProcessor',
    'ProcessingContext',
    
    # 数据类型
    'ElementInfo',
    'BoundingBox',
    'ProcessingResult',
    'XMLFragment',
    'LayerLevel',
    'get_layer_level',
    
    # 核心模块（你负责）
    'Sam3InfoExtractor',      # 任务1
    'XMLMerger',              # 任务6
    
    # 图形处理模块（占位）
    'IconPictureProcessor',   # 任务2
    'BasicShapeProcessor',    # 任务3
    'OtherShapeProcessor',    # 任务4
    'ArrowProcessor',         # 任务5
    'MetricEvaluator',        # 任务7
    'RefinementProcessor',    # 任务8
    'EnhancedFrameDetector',  # 增强框检测
    'enhance_frame_detection',
    
    # 文字处理模块（另一个小组）
    'TextProcessor',
    'create_text_fragments',
    'create_single_text_xml',
    
    # 快捷函数
    'merge_fragments',
    'merge_shape_and_text',
]
