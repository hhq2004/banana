"""
任务4：其他图形处理模块

功能：
    - 处理不使用SAM3的额外图形逻辑
    - 可能包括：特殊形状检测、模板匹配、规则生成等
    - 输出XML和相关信息

负责人：[待分配]
负责任务：任务4 - 其他考虑的图形实现逻辑（不利用SAM3的额外逻辑）

使用示例：
    from modules import OtherShapeProcessor, ProcessingContext
    
    processor = OtherShapeProcessor()
    context = ProcessingContext(image_path="test.png")
    
    result = processor.process(context)

接口说明：
    输入：
        - context.image_path: 原始图片路径
        - context.elements: 已有元素（可选，用于避免重复检测）
        
    输出：
        - 新检测到的ElementInfo列表
        - 这些元素不依赖SAM3检测

TODO: 以下是一些可能的扩展方向，由其他同事实现：
    1. 模板匹配检测特定图标
    2. 霍夫变换检测直线/圆
    3. 轮廓分析检测特殊形状
    4. OCR区域辅助定位
"""

import os
from typing import List, Optional
import cv2
import numpy as np
from PIL import Image

from .base import BaseProcessor, ProcessingContext
from .data_types import ElementInfo, BoundingBox, ProcessingResult


class OtherShapeProcessor(BaseProcessor):
    """
    其他图形处理模块
    
    这是一个占位模块，用于处理SAM3无法检测或需要特殊逻辑的图形。
    
    TODO: 其他同事可以实现以下功能：
        1. 模板匹配：预定义模板库，匹配特定图标
        2. 轮廓检测：使用传统CV方法检测简单形状
        3. 文字区域辅助：根据OCR结果推断图形位置
        4. 规则生成：根据图形排列规律自动补全
    """
    
    def __init__(self, config=None):
        super().__init__(config)
    
    def process(self, context: ProcessingContext) -> ProcessingResult:
        """
        处理入口
        
        Args:
            context: 处理上下文
            
        Returns:
            ProcessingResult
        """
        self._log("开始其他图形处理（当前为占位实现）")
        
        # TODO: 实现额外的图形检测逻辑
        new_elements = []
        
        # 示例：这里可以添加模板匹配、轮廓检测等逻辑
        # new_elements = self._detect_by_template(context.image_path)
        # new_elements += self._detect_by_contour(context.image_path)
        
        # 合并到现有元素
        if new_elements:
            start_id = len(context.elements)
            for i, elem in enumerate(new_elements):
                elem.id = start_id + i
            context.elements.extend(new_elements)
        
        return ProcessingResult(
            success=True,
            elements=context.elements,
            canvas_width=context.canvas_width,
            canvas_height=context.canvas_height,
            metadata={
                'new_elements_count': len(new_elements)
            }
        )
    
    def _detect_by_template(self, image_path: str) -> List[ElementInfo]:
        """
        模板匹配检测
        
        TODO: 实现模板匹配逻辑
        """
        # 占位
        return []
    
    def _detect_by_contour(self, image_path: str) -> List[ElementInfo]:
        """
        轮廓检测
        
        TODO: 实现轮廓检测逻辑
        """
        # 占位
        return []
    
    def _detect_by_hough(self, image_path: str) -> List[ElementInfo]:
        """
        霍夫变换检测直线和圆
        
        TODO: 实现霍夫变换检测
        """
        # 占位
        return []


# ======================== 快捷函数 ========================
def detect_other_shapes(image_path: str,
                        existing_elements: List[ElementInfo] = None) -> List[ElementInfo]:
    """
    快捷函数 - 检测其他图形
    
    Args:
        image_path: 图片路径
        existing_elements: 已有元素（避免重复）
        
    Returns:
        新检测到的元素列表
        
    使用示例:
        new_elements = detect_other_shapes("test.png")
    """
    processor = OtherShapeProcessor()
    context = ProcessingContext(
        image_path=image_path,
        elements=existing_elements or []
    )
    
    result = processor.process(context)
    # 返回新增的元素
    if existing_elements:
        return result.elements[len(existing_elements):]
    return result.elements
