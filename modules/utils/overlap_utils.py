"""
重叠元素检测和处理工具

提供识别和处理重叠元素的功能，用于：
1. 检测哪些元素存在重叠
2. 分析重叠类型（包含、交叉、层叠）
3. 决定合理的处理策略
4. 生成正确的层叠XML

使用示例:
    from modules.utils.overlap_utils import detect_overlaps, analyze_overlap_relationships
    
    # 检测重叠
    overlaps = detect_overlaps(elements, iou_threshold=0.1)
    
    # 分析关系
    relationships = analyze_overlap_relationships(elements, image_path)
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """计算两个bbox的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    inter_area = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def calculate_containment(box_outer: List[int], box_inner: List[int]) -> float:
    """
    计算box_inner被box_outer包含的比例
    
    返回值范围 [0, 1]：
    - 1.0 表示完全包含
    - 0.0 表示无重叠
    """
    x1 = max(box_outer[0], box_inner[0])
    y1 = max(box_outer[1], box_inner[1])
    x2 = min(box_outer[2], box_inner[2])
    y2 = min(box_outer[3], box_inner[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    inter_area = (x2 - x1) * (y2 - y1)
    inner_area = (box_inner[2] - box_inner[0]) * (box_inner[3] - box_inner[1])
    
    return inter_area / inner_area if inner_area > 0 else 0.0


def detect_overlaps(elements: List[Any], 
                    iou_threshold: float = 0.1) -> List[Dict]:
    """
    检测所有元素之间的重叠关系
    
    Args:
        elements: 元素列表（需要有 bbox 属性或 'bbox' 键）
        iou_threshold: IoU阈值，超过此值认为有重叠
        
    Returns:
        重叠关系列表，每个元素包含：
        - elem_a_id: 第一个元素ID
        - elem_b_id: 第二个元素ID
        - iou: IoU值
        - relationship: 关系类型 (contains, contained, overlaps, stacked)
    """
    overlaps = []
    
    for i, elem_a in enumerate(elements):
        for j, elem_b in enumerate(elements):
            if i >= j:
                continue
            
            # 获取bbox
            if hasattr(elem_a, 'bbox'):
                bbox_a = elem_a.bbox.to_list() if hasattr(elem_a.bbox, 'to_list') else list(elem_a.bbox)
            else:
                bbox_a = elem_a.get('bbox', elem_a)
            
            if hasattr(elem_b, 'bbox'):
                bbox_b = elem_b.bbox.to_list() if hasattr(elem_b.bbox, 'to_list') else list(elem_b.bbox)
            else:
                bbox_b = elem_b.get('bbox', elem_b)
            
            iou = calculate_iou(bbox_a, bbox_b)
            
            if iou < iou_threshold:
                continue
            
            # 分析关系类型
            relationship = analyze_relationship(bbox_a, bbox_b)
            
            # 获取元素ID
            elem_a_id = elem_a.id if hasattr(elem_a, 'id') else i
            elem_b_id = elem_b.id if hasattr(elem_b, 'id') else j
            
            overlaps.append({
                'elem_a_id': elem_a_id,
                'elem_b_id': elem_b_id,
                'elem_a_idx': i,
                'elem_b_idx': j,
                'iou': iou,
                'relationship': relationship,
                'bbox_a': bbox_a,
                'bbox_b': bbox_b
            })
    
    return overlaps


def analyze_relationship(bbox_a: List[int], bbox_b: List[int]) -> str:
    """
    分析两个bbox的关系类型
    
    Returns:
        - 'contains': A完全包含B
        - 'contained': A被B完全包含
        - 'stacked': 层叠关系（一个的中心在另一个内部）
        - 'overlaps': 普通重叠/交叉
    """
    # 检查包含关系
    a_contains_b = (bbox_a[0] <= bbox_b[0] and bbox_a[1] <= bbox_b[1] and
                    bbox_a[2] >= bbox_b[2] and bbox_a[3] >= bbox_b[3])
    b_contains_a = (bbox_b[0] <= bbox_a[0] and bbox_b[1] <= bbox_a[1] and
                    bbox_b[2] >= bbox_a[2] and bbox_b[3] >= bbox_a[3])
    
    if a_contains_b:
        return 'contains'
    if b_contains_a:
        return 'contained'
    
    # 检查是否是层叠关系（一个的中心在另一个内部）
    center_a = ((bbox_a[0] + bbox_a[2]) // 2, (bbox_a[1] + bbox_a[3]) // 2)
    center_b = ((bbox_b[0] + bbox_b[2]) // 2, (bbox_b[1] + bbox_b[3]) // 2)
    
    a_center_in_b = (bbox_b[0] < center_a[0] < bbox_b[2] and 
                     bbox_b[1] < center_a[1] < bbox_b[3])
    b_center_in_a = (bbox_a[0] < center_b[0] < bbox_a[2] and 
                     bbox_a[1] < center_b[1] < bbox_a[3])
    
    if a_center_in_b or b_center_in_a:
        return 'stacked'
    
    return 'overlaps'


def analyze_overlap_relationships(elements: List[Any],
                                   image_path: str,
                                   iou_threshold: float = 0.1) -> List[Dict]:
    """
    深入分析重叠元素的关系，包括图像内容分析
    
    Args:
        elements: 元素列表
        image_path: 原始图片路径
        iou_threshold: IoU阈值
        
    Returns:
        详细的重叠分析结果
    """
    cv2_image = cv2.imread(image_path)
    if cv2_image is None:
        return detect_overlaps(elements, iou_threshold)
    
    overlaps = detect_overlaps(elements, iou_threshold)
    
    for overlap in overlaps:
        # 分析重叠区域的图像特征
        bbox_a = overlap['bbox_a']
        bbox_b = overlap['bbox_b']
        
        # 计算交集区域
        inter_x1 = max(bbox_a[0], bbox_b[0])
        inter_y1 = max(bbox_a[1], bbox_b[1])
        inter_x2 = min(bbox_a[2], bbox_b[2])
        inter_y2 = min(bbox_a[3], bbox_b[3])
        
        if inter_x2 > inter_x1 and inter_y2 > inter_y1:
            # 提取交集区域
            inter_roi = cv2_image[inter_y1:inter_y2, inter_x1:inter_x2]
            
            # 分析交集区域复杂度
            gray = cv2.cvtColor(inter_roi, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            std_dev = np.std(gray)
            
            overlap['inter_complexity'] = {
                'laplacian_var': float(laplacian_var),
                'std_dev': float(std_dev),
                'is_complex': laplacian_var > 500 or std_dev > 40
            }
            
            # 分析是否有明显边界
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = np.count_nonzero(edges) / edges.size
            overlap['inter_complexity']['edge_ratio'] = float(edge_ratio)
            overlap['inter_complexity']['has_clear_boundary'] = edge_ratio > 0.05
    
    return overlaps


def resolve_overlapping_elements(elements: List[Any],
                                  overlaps: List[Dict],
                                  strategy: str = 'layer') -> List[Any]:
    """
    解决重叠元素的处理策略
    
    Args:
        elements: 元素列表
        overlaps: 重叠分析结果
        strategy: 处理策略
            - 'layer': 保持层叠，调整z-index
            - 'merge': 合并重叠元素
            - 'priority': 按优先级只保留一个
            
    Returns:
        处理后的元素列表
    """
    if strategy == 'layer':
        # 层叠策略：不删除，只调整层级
        # 确保包含关系中，外层元素在底层
        for overlap in overlaps:
            if overlap['relationship'] == 'contains':
                # A包含B，A应该在底层
                elem_a_idx = overlap['elem_a_idx']
                elem_b_idx = overlap['elem_b_idx']
                
                if hasattr(elements[elem_a_idx], 'layer_level') and \
                   hasattr(elements[elem_b_idx], 'layer_level'):
                    if elements[elem_a_idx].layer_level >= elements[elem_b_idx].layer_level:
                        # A的层级不应该高于或等于B
                        elements[elem_a_idx].layer_level = elements[elem_b_idx].layer_level - 1
                        elements[elem_a_idx].processing_notes.append(
                            f"层级调整：作为{elements[elem_b_idx].id}的容器"
                        )
        
        return elements
    
    elif strategy == 'merge':
        # 合并策略：将高度重叠的元素合并
        to_remove = set()
        
        for overlap in overlaps:
            if overlap['iou'] > 0.8:  # 高度重叠才合并
                # 保留面积大的，删除面积小的
                area_a = (overlap['bbox_a'][2] - overlap['bbox_a'][0]) * \
                         (overlap['bbox_a'][3] - overlap['bbox_a'][1])
                area_b = (overlap['bbox_b'][2] - overlap['bbox_b'][0]) * \
                         (overlap['bbox_b'][3] - overlap['bbox_b'][1])
                
                if area_a >= area_b:
                    to_remove.add(overlap['elem_b_idx'])
                else:
                    to_remove.add(overlap['elem_a_idx'])
        
        return [elem for i, elem in enumerate(elements) if i not in to_remove]
    
    elif strategy == 'priority':
        # 优先级策略：按score保留高置信度的
        to_remove = set()
        
        for overlap in overlaps:
            elem_a = elements[overlap['elem_a_idx']]
            elem_b = elements[overlap['elem_b_idx']]
            
            score_a = elem_a.score if hasattr(elem_a, 'score') else 0.5
            score_b = elem_b.score if hasattr(elem_b, 'score') else 0.5
            
            if score_a >= score_b:
                to_remove.add(overlap['elem_b_idx'])
            else:
                to_remove.add(overlap['elem_a_idx'])
        
        return [elem for i, elem in enumerate(elements) if i not in to_remove]
    
    return elements


def group_overlapping_elements(elements: List[Any],
                                iou_threshold: float = 0.1) -> List[List[int]]:
    """
    将重叠的元素分组
    
    使用并查集算法将互相重叠的元素分到同一组
    
    Args:
        elements: 元素列表
        iou_threshold: IoU阈值
        
    Returns:
        元素索引分组列表，如 [[0, 3, 5], [1, 2], [4]]
        每组内的元素互相有重叠关系
    """
    n = len(elements)
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # 检测重叠并合并
    overlaps = detect_overlaps(elements, iou_threshold)
    for overlap in overlaps:
        union(overlap['elem_a_idx'], overlap['elem_b_idx'])
    
    # 按根节点分组
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)
    
    return list(groups.values())


def suggest_xml_structure(elements: List[Any],
                           overlaps: List[Dict]) -> List[Dict]:
    """
    根据重叠关系建议XML结构
    
    为重叠元素建议合理的XML层级和父子关系
    
    Args:
        elements: 元素列表
        overlaps: 重叠分析结果
        
    Returns:
        结构建议列表
    """
    suggestions = []
    
    for overlap in overlaps:
        relationship = overlap['relationship']
        elem_a_idx = overlap['elem_a_idx']
        elem_b_idx = overlap['elem_b_idx']
        
        if relationship == 'contains':
            # A包含B：A应该是容器，B是子元素
            suggestions.append({
                'type': 'container_child',
                'container_idx': elem_a_idx,
                'child_idx': elem_b_idx,
                'description': f'元素{elem_a_idx}应作为元素{elem_b_idx}的容器'
            })
        
        elif relationship == 'stacked':
            # 层叠关系：建议保持独立，但调整z-index
            suggestions.append({
                'type': 'stacked_layers',
                'bottom_idx': elem_a_idx,  # 面积大的在底层
                'top_idx': elem_b_idx,
                'description': f'元素{elem_a_idx}和{elem_b_idx}层叠，建议调整层级'
            })
        
        elif relationship == 'overlaps':
            # 普通重叠：可能需要人工检查
            suggestions.append({
                'type': 'partial_overlap',
                'elem_a_idx': elem_a_idx,
                'elem_b_idx': elem_b_idx,
                'iou': overlap['iou'],
                'description': f'元素{elem_a_idx}和{elem_b_idx}部分重叠(IoU={overlap["iou"]:.2f})，建议检查'
            })
    
    return suggestions
