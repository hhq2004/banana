"""
XML处理工具函数
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, Optional, Any


def create_mxcell(cell_id: int, 
                  style: str,
                  x: int, 
                  y: int, 
                  width: int, 
                  height: int,
                  value: str = "",
                  parent: str = "1") -> ET.Element:
    """
    创建mxCell元素
    
    Args:
        cell_id: 单元格ID
        style: DrawIO样式字符串
        x, y: 位置
        width, height: 尺寸
        value: 单元格值（通常为空或文本内容）
        parent: 父元素ID
        
    Returns:
        ET.Element
    """
    cell = ET.Element("mxCell", {
        "id": str(cell_id),
        "parent": parent,
        "vertex": "1",
        "value": value,
        "style": style
    })
    
    geometry = ET.SubElement(cell, "mxGeometry", {
        "x": str(x),
        "y": str(y),
        "width": str(width),
        "height": str(height),
        "as": "geometry"
    })
    
    return cell


def create_geometry(x: int, y: int, width: int, height: int) -> ET.Element:
    """创建mxGeometry元素"""
    return ET.Element("mxGeometry", {
        "x": str(x),
        "y": str(y),
        "width": str(width),
        "height": str(height),
        "as": "geometry"
    })


def prettify_xml(elem: ET.Element) -> str:
    """
    格式化XML输出（移除版本声明，过滤空行）
    
    Args:
        elem: XML元素
        
    Returns:
        格式化后的XML字符串
    """
    rough_string = ET.tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    
    lines = reparsed.toprettyxml(indent="  ").split('\n')
    return '\n'.join([
        line for line in lines
        if line.strip() and not line.strip().startswith("<?xml")
    ])


def parse_drawio_xml(xml_path: str) -> Optional[Dict[str, Any]]:
    """
    解析DrawIO XML文件
    
    Args:
        xml_path: XML文件路径
        
    Returns:
        解析结果字典，包含 canvas_size 和 cells 列表
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        model = root.find(".//mxGraphModel")
        if model is None:
            return None
        
        result = {
            'canvas_width': int(model.get("pageWidth", 800)),
            'canvas_height': int(model.get("pageHeight", 600)),
            'cells': []
        }
        
        root_elem = model.find("root")
        if root_elem:
            for cell in root_elem:
                cell_id = cell.get("id")
                if cell_id not in ["0", "1"]:
                    geometry = cell.find("mxGeometry")
                    cell_info = {
                        'id': cell_id,
                        'style': cell.get("style", ""),
                        'value': cell.get("value", ""),
                    }
                    if geometry is not None:
                        cell_info['x'] = int(float(geometry.get("x", 0)))
                        cell_info['y'] = int(float(geometry.get("y", 0)))
                        cell_info['width'] = int(float(geometry.get("width", 0)))
                        cell_info['height'] = int(float(geometry.get("height", 0)))
                    result['cells'].append(cell_info)
        
        return result
        
    except Exception as e:
        print(f"解析XML失败: {e}")
        return None
