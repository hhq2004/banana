"""
文字处理模块接口

负责人：[文字处理小组]
负责任务：OCR文字识别与XML生成

说明：
    这是给文字处理小组预留的接口。
    文字小组需要实现 process() 方法，返回文字的XML代码。
    
    最终会在 XMLMerger 中与其他模块的XML合并。

输入：
    - context.image_path: 原始图片路径
    - context.canvas_width, canvas_height: 画布尺寸
    - context.elements: 其他模块已识别的元素（可选，用于避免重叠）

输出要求：
    - 设置 element.xml_fragment: 完整的mxCell XML字符串
    - 设置 element.layer_level = LayerLevel.TEXT.value (4)
    - 或者直接返回 XMLFragment 列表

XML格式示例：
    <mxCell id="0" parent="1" vertex="1" value="文字内容" 
            style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=12;fontColor=#000000;">
      <mxGeometry x="100" y="100" width="80" height="20" as="geometry"/>
    </mxCell>

使用示例：
    from modules import TextProcessor, ProcessingContext
    
    processor = TextProcessor()
    context = ProcessingContext(image_path="test.png")
    result = processor.process(context)
    
    # result.elements 或 result.xml_fragments 包含文字的XML
"""

import os
from typing import List, Optional, Dict, Any
from PIL import Image

from .base import BaseProcessor, ProcessingContext
from .data_types import (
    ElementInfo, 
    BoundingBox, 
    ProcessingResult, 
    XMLFragment,
    LayerLevel,
)


# ======================== DrawIO文字样式参考 ========================
TEXT_STYLES = {
    # 基础文本样式
    "default": "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;whiteSpace=wrap;rounded=0;",
    
    # 居中文本
    "center": "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;",
    
    # 标题样式
    "title": "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontStyle=1;fontSize=14;",
    
    # LaTeX公式样式（DrawIO支持）
    "latex": "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;",
}


class TextProcessor(BaseProcessor):
    """
    文字处理模块
    
    ⚠️ 这是接口占位，由文字处理小组实现
    
    职责：
        1. OCR识别图片中的文字
        2. 确定文字的位置和大小
        3. 生成mxCell XML代码
        4. 设置 layer_level = TEXT (4)，确保文字在最上层
    
    XML输出格式：
        <mxCell id="临时ID" parent="1" vertex="1" 
                value="识别的文字内容" 
                style="text;html=1;strokeColor=none;fillColor=none;...">
          <mxGeometry x="x坐标" y="y坐标" width="宽度" height="高度" as="geometry"/>
        </mxCell>
    
    特殊处理：
        - 如果是数学公式，value 应该使用 LaTeX 格式
        - DrawIO 支持在 value 中使用 HTML，如 <b>加粗</b>
    """
    
    def __init__(self, config=None):
        super().__init__(config)
    
    def process(self, context: ProcessingContext) -> ProcessingResult:
        """
        处理入口 - 文字小组需要实现
        
        Args:
            context: 处理上下文
            
        Returns:
            ProcessingResult，包含文字元素的XML
            
        实现要点：
            1. 调用OCR识别文字
            2. 对每个文字区域生成XML
            3. 设置 layer_level = LayerLevel.TEXT.value
        """
        self._log("开始处理文字（当前为占位实现）")
        
        # TODO: 文字小组实现以下逻辑
        # 
        # 1. 调用OCR识别
        # text_regions = self._run_ocr(context.image_path)
        #
        # 2. 对每个文字区域生成XML
        # for region in text_regions:
        #     xml = self._generate_text_xml(region)
        #     fragment = XMLFragment(
        #         element_id=...,
        #         xml_content=xml,
        #         layer_level=LayerLevel.TEXT.value,  # 文字在最上层
        #         bbox=region.bbox,
        #         element_type="text"
        #     )
        #     context.xml_fragments.append(fragment)
        
        return ProcessingResult(
            success=True,
            elements=context.elements,
            xml_fragments=context.xml_fragments,
            canvas_width=context.canvas_width,
            canvas_height=context.canvas_height,
            metadata={
                'text_count': 0,
                'note': '占位实现，待文字小组填充'
            }
        )
    
    def process_from_ocr_result(self, 
                                 ocr_results: List[Dict[str, Any]],
                                 canvas_width: int,
                                 canvas_height: int) -> List[XMLFragment]:
        """
        从OCR结果生成XML片段
        
        这是一个便捷方法，文字小组可以直接调用。
        
        Args:
            ocr_results: OCR结果列表，每个元素应该包含：
                {
                    'text': str,           # 识别的文字
                    'bbox': [x1, y1, x2, y2],  # 边界框
                    'confidence': float,   # 置信度（可选）
                    'font_size': int,      # 字号（可选，默认12）
                    'font_color': str,     # 字体颜色（可选，默认#000000）
                    'align': str,          # 对齐方式（可选，默认left）
                    'is_latex': bool,      # 是否是LaTeX公式（可选）
                }
            canvas_width: 画布宽度
            canvas_height: 画布高度
            
        Returns:
            XMLFragment列表
            
        使用示例:
            processor = TextProcessor()
            
            ocr_results = [
                {'text': '开始', 'bbox': [100, 100, 160, 130]},
                {'text': '结束', 'bbox': [300, 100, 360, 130]},
                {'text': '$f(x) = x^2$', 'bbox': [200, 200, 300, 240], 'is_latex': True},
            ]
            
            fragments = processor.process_from_ocr_result(ocr_results, 800, 600)
            
            # 然后传给XMLMerger
            context.xml_fragments.extend(fragments)
        """
        fragments = []
        
        for i, ocr_item in enumerate(ocr_results):
            text = ocr_item.get('text', '')
            bbox = ocr_item.get('bbox', [0, 0, 100, 20])
            font_size = ocr_item.get('font_size', 12)
            font_color = ocr_item.get('font_color', '#000000')
            align = ocr_item.get('align', 'left')
            is_latex = ocr_item.get('is_latex', False)
            
            # 生成XML
            xml = self._generate_text_xml(
                text=text,
                bbox=bbox,
                font_size=font_size,
                font_color=font_color,
                align=align,
                is_latex=is_latex,
                temp_id=i
            )
            
            x1, y1, x2, y2 = bbox
            fragment = XMLFragment(
                element_id=i,
                xml_content=xml,
                layer_level=LayerLevel.TEXT.value,  # 文字在最上层
                bbox=BoundingBox(x1, y1, x2, y2),
                element_type="text"
            )
            
            fragments.append(fragment)
        
        return fragments
    
    def _generate_text_xml(self,
                           text: str,
                           bbox: List[int],
                           font_size: int = 12,
                           font_color: str = "#000000",
                           align: str = "left",
                           is_latex: bool = False,
                           temp_id: int = 0) -> str:
        """
        生成文字的mxCell XML
        
        Args:
            text: 文字内容
            bbox: [x1, y1, x2, y2]
            font_size: 字号
            font_color: 字体颜色
            align: 对齐方式 (left/center/right)
            is_latex: 是否是LaTeX公式
            temp_id: 临时ID（合并时会被重新分配）
            
        Returns:
            mxCell XML字符串
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # 转义特殊字符
        escaped_text = self._escape_xml(text)
        
        # 如果是LaTeX公式，保持原格式（DrawIO会渲染）
        if is_latex:
            # DrawIO LaTeX 格式: `$公式$` 或在math模式下
            value = escaped_text
        else:
            value = escaped_text
        
        # 构建样式
        style = f"text;html=1;strokeColor=none;fillColor=none;align={align};verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize={font_size};fontColor={font_color};"
        
        xml = f'''<mxCell id="{temp_id}" parent="1" vertex="1" value="{value}" style="{style}">
  <mxGeometry x="{x1}" y="{y1}" width="{width}" height="{height}" as="geometry"/>
</mxCell>'''
        
        return xml
    
    def _escape_xml(self, text: str) -> str:
        """转义XML特殊字符"""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&apos;'))


# ======================== 便捷函数 ========================
def create_text_fragments(ocr_results: List[Dict[str, Any]],
                          canvas_width: int = 800,
                          canvas_height: int = 600) -> List[XMLFragment]:
    """
    快捷函数 - 从OCR结果创建XML片段
    
    Args:
        ocr_results: OCR结果列表
        canvas_width: 画布宽度
        canvas_height: 画布高度
        
    Returns:
        XMLFragment列表
        
    使用示例:
        from modules.text_processor import create_text_fragments
        
        ocr_results = [
            {'text': '开始', 'bbox': [100, 100, 160, 130]},
            {'text': '处理', 'bbox': [200, 100, 260, 130]},
        ]
        
        fragments = create_text_fragments(ocr_results)
        
        # 然后和其他片段一起传给XMLMerger
        all_fragments = shape_fragments + icon_fragments + fragments
        merge_fragments(all_fragments, 800, 600, "output.xml")
    """
    processor = TextProcessor()
    return processor.process_from_ocr_result(ocr_results, canvas_width, canvas_height)


def create_single_text_xml(text: str,
                           x: int,
                           y: int,
                           width: int = None,
                           height: int = None,
                           font_size: int = 12,
                           font_color: str = "#000000",
                           align: str = "left") -> str:
    """
    快捷函数 - 创建单个文字的XML
    
    Args:
        text: 文字内容
        x, y: 位置
        width, height: 尺寸（如果不指定，根据字数估算）
        font_size: 字号
        font_color: 字体颜色
        align: 对齐方式
        
    Returns:
        mxCell XML字符串
        
    使用示例:
        xml = create_single_text_xml("Hello", x=100, y=100)
    """
    # 估算尺寸
    if width is None:
        width = max(len(text) * font_size * 0.6, 20)
    if height is None:
        height = font_size + 8
    
    processor = TextProcessor()
    return processor._generate_text_xml(
        text=text,
        bbox=[x, y, x + int(width), y + int(height)],
        font_size=font_size,
        font_color=font_color,
        align=align
    )
