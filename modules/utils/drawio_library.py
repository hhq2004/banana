"""
DrawIO原生元素库支持

这个模块提供DrawIO内置元素的定义和匹配功能。

DrawIO内置了大量可编辑的图形元素，包括：
- 基本形状（矩形、椭圆、菱形等）
- UML元素（类图、活动图、序列图等）
- 流程图元素（开始/结束、判断、处理等）
- 箭头和连接线（完整的箭头属性库）
- 网络拓扑元素
- AWS/Azure/GCP云架构图标
- 通用图标库

如果能识别出图片中的元素是DrawIO原生支持的，就可以用样式代码描述而不是base64图片，
这样生成的元素是可编辑的。

使用示例:
    from modules.utils.drawio_library import DrawIOLibrary, match_element_to_drawio
    from modules.utils.drawio_library import ArrowAttributeDetector, build_arrow_style
    
    # 检查是否匹配内置形状
    match = match_element_to_drawio(element, image_region)
    if match:
        print(f"匹配到: {match['name']}, 样式: {match['style']}")
    
    # 检测箭头属性
    detector = ArrowAttributeDetector()
    attrs = detector.detect_all_attributes(arrow_image, path_points)
    style = build_arrow_style(**attrs)
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy import ndimage
from scipy.signal import find_peaks


# ======================== DrawIO内置形状定义 ========================

DRAWIO_BASIC_SHAPES = {
    # 基础形状
    'rectangle': {
        'style': 'rounded=0;whiteSpace=wrap;html=1;',
        'description': '矩形',
        'features': {'rounded': False, 'aspect_range': (0.2, 5.0)}
    },
    'rounded_rectangle': {
        'style': 'rounded=1;whiteSpace=wrap;html=1;',
        'description': '圆角矩形',
        'features': {'rounded': True, 'aspect_range': (0.2, 5.0)}
    },
    'ellipse': {
        'style': 'ellipse;whiteSpace=wrap;html=1;',
        'description': '椭圆',
        'features': {'shape': 'ellipse', 'aspect_range': (0.3, 3.0)}
    },
    'circle': {
        'style': 'ellipse;whiteSpace=wrap;html=1;aspect=fixed;',
        'description': '圆形',
        'features': {'shape': 'ellipse', 'aspect_range': (0.9, 1.1)}
    },
    'diamond': {
        'style': 'rhombus;whiteSpace=wrap;html=1;',
        'description': '菱形',
        'features': {'shape': 'diamond', 'vertices': 4}
    },
    'triangle': {
        'style': 'triangle;whiteSpace=wrap;html=1;',
        'description': '三角形',
        'features': {'vertices': 3}
    },
    'hexagon': {
        'style': 'shape=hexagon;perimeter=hexagonPerimeter2;whiteSpace=wrap;html=1;fixedSize=1;',
        'description': '六边形',
        'features': {'vertices': 6}
    },
    'parallelogram': {
        'style': 'shape=parallelogram;perimeter=parallelogramPerimeter;whiteSpace=wrap;html=1;fixedSize=1;',
        'description': '平行四边形',
        'features': {'shape': 'parallelogram', 'vertices': 4}
    },
    'trapezoid': {
        'style': 'shape=trapezoid;perimeter=trapezoidPerimeter;whiteSpace=wrap;html=1;fixedSize=1;',
        'description': '梯形',
        'features': {'shape': 'trapezoid', 'vertices': 4}
    },
    'cylinder': {
        'style': 'shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;size=15;',
        'description': '圆柱体（数据库）',
        'features': {'shape': 'cylinder'}
    },
    'cloud': {
        'style': 'ellipse;shape=cloud;whiteSpace=wrap;html=1;',
        'description': '云朵',
        'features': {'shape': 'cloud'}
    },
}

DRAWIO_UML_SHAPES = {
    'actor': {
        'style': 'shape=umlActor;verticalLabelPosition=bottom;verticalAlign=top;html=1;outlineConnect=0;',
        'description': 'UML 参与者',
        'features': {'shape': 'actor'}
    },
    'class': {
        'style': 'swimlane;fontStyle=1;align=center;verticalAlign=top;childLayout=stackLayout;horizontal=1;startSize=26;horizontalStack=0;resizeParent=1;resizeParentMax=0;resizeLast=0;collapsible=1;marginBottom=0;',
        'description': 'UML 类',
        'features': {'shape': 'class'}
    },
    'lifeline': {
        'style': 'shape=umlLifeline;perimeter=lifelinePerimeter;whiteSpace=wrap;html=1;container=1;collapsible=0;recursiveResize=0;outlineConnect=0;',
        'description': 'UML 生命线',
        'features': {'shape': 'lifeline'}
    },
}

DRAWIO_FLOWCHART_SHAPES = {
    'start_end': {
        'style': 'ellipse;whiteSpace=wrap;html=1;',
        'description': '流程图 开始/结束',
        'features': {'shape': 'ellipse', 'aspect_range': (1.5, 3.0)}
    },
    'process': {
        'style': 'rounded=0;whiteSpace=wrap;html=1;',
        'description': '流程图 处理',
        'features': {'rounded': False}
    },
    'decision': {
        'style': 'rhombus;whiteSpace=wrap;html=1;',
        'description': '流程图 判断',
        'features': {'shape': 'diamond'}
    },
    'data': {
        'style': 'shape=parallelogram;perimeter=parallelogramPerimeter;whiteSpace=wrap;html=1;fixedSize=1;',
        'description': '流程图 数据',
        'features': {'shape': 'parallelogram'}
    },
    'document': {
        'style': 'shape=document;whiteSpace=wrap;html=1;boundedLbl=1;',
        'description': '流程图 文档',
        'features': {'shape': 'document'}
    },
    'predefined_process': {
        'style': 'shape=process;whiteSpace=wrap;html=1;backgroundOutline=1;',
        'description': '流程图 预定义处理',
        'features': {'shape': 'process'}
    },
}

# 所有DrawIO内置形状的汇总
ALL_DRAWIO_SHAPES = {
    **DRAWIO_BASIC_SHAPES,
    **DRAWIO_UML_SHAPES,
    **DRAWIO_FLOWCHART_SHAPES,
}


# ======================== 完整的DrawIO箭头属性库 ========================

# 箭头头部类型
DRAWIO_ARROW_HEADS = {
    # ===== 基础箭头 =====
    'none': {
        'description': '无箭头',
        'filled': False,
        'vertices': 0,
        'category': 'basic'
    },
    'classic': {
        'description': '经典三角形箭头',
        'filled': True,
        'vertices': 3,
        'category': 'basic'
    },
    'classicThin': {
        'description': '细经典三角形',
        'filled': True,
        'vertices': 3,
        'category': 'basic'
    },
    'open': {
        'description': '开放V形箭头',
        'filled': False,
        'vertices': 3,
        'category': 'basic'
    },
    'openThin': {
        'description': '细开放V形',
        'filled': False,
        'vertices': 3,
        'category': 'basic'
    },
    'openAsync': {
        'description': '异步开放箭头',
        'filled': False,
        'vertices': 2,
        'category': 'basic'
    },
    'block': {
        'description': '实心块状箭头',
        'filled': True,
        'vertices': 4,
        'category': 'basic'
    },
    'blockThin': {
        'description': '细块状箭头',
        'filled': True,
        'vertices': 4,
        'category': 'basic'
    },
    'async': {
        'description': '异步箭头（半箭头）',
        'filled': True,
        'vertices': 3,
        'category': 'basic'
    },
    
    # ===== 特殊形状箭头 =====
    'oval': {
        'description': '椭圆形',
        'filled': True,
        'vertices': 0,
        'shape': 'ellipse',
        'category': 'special'
    },
    'diamond': {
        'description': '菱形（聚合关系）',
        'filled': True,
        'vertices': 4,
        'shape': 'diamond',
        'category': 'special'
    },
    'diamondThin': {
        'description': '细菱形',
        'filled': True,
        'vertices': 4,
        'shape': 'diamond',
        'category': 'special'
    },
    'circle': {
        'description': '圆形',
        'filled': True,
        'vertices': 0,
        'shape': 'circle',
        'category': 'special'
    },
    'halfCircle': {
        'description': '半圆',
        'filled': False,
        'vertices': 0,
        'shape': 'halfcircle',
        'category': 'special'
    },
    'dash': {
        'description': '短横线',
        'filled': False,
        'vertices': 2,
        'category': 'special'
    },
    'cross': {
        'description': '叉号',
        'filled': False,
        'vertices': 4,
        'category': 'special'
    },
    'circlePlus': {
        'description': '圆内加号',
        'filled': True,
        'vertices': 0,
        'shape': 'circle_plus',
        'category': 'special'
    },
    'baseDash': {
        'description': '基础短线',
        'filled': False,
        'vertices': 2,
        'category': 'special'
    },
    'doubleBlock': {
        'description': '双块状箭头',
        'filled': True,
        'vertices': 8,
        'category': 'special'
    },
    
    # ===== ER图箭头 =====
    'ERone': {
        'description': 'ER图 一',
        'filled': False,
        'vertices': 2,
        'category': 'er'
    },
    'ERmandOne': {
        'description': 'ER图 强制一',
        'filled': False,
        'vertices': 3,
        'category': 'er'
    },
    'ERmany': {
        'description': 'ER图 多（鱼尾）',
        'filled': False,
        'vertices': 3,
        'category': 'er'
    },
    'ERoneToMany': {
        'description': 'ER图 一对多',
        'filled': False,
        'vertices': 4,
        'category': 'er'
    },
    'ERzeroToOne': {
        'description': 'ER图 零或一',
        'filled': False,
        'vertices': 2,
        'category': 'er'
    },
    'ERzeroToMany': {
        'description': 'ER图 零或多',
        'filled': False,
        'vertices': 3,
        'category': 'er'
    },
    
    # ===== UML箭头 =====
    'box': {
        'description': 'UML 方框',
        'filled': True,
        'vertices': 4,
        'category': 'uml'
    },
}

# 连线样式
DRAWIO_EDGE_STYLES = {
    'none': {
        'description': '直线',
        'orthogonal': False,
        'curved': False
    },
    'orthogonalEdgeStyle': {
        'description': '正交折线',
        'orthogonal': True,
        'curved': False
    },
    'elbowEdgeStyle': {
        'description': '肘形连线',
        'orthogonal': True,
        'curved': False
    },
    'entityRelationEdgeStyle': {
        'description': 'ER图连线',
        'orthogonal': True,
        'curved': False
    },
    'segmentEdgeStyle': {
        'description': '分段线',
        'orthogonal': False,
        'curved': False
    },
    'isometricEdgeStyle': {
        'description': '等轴测连线',
        'orthogonal': False,
        'curved': False
    },
}

# 虚线模式
DRAWIO_DASH_PATTERNS = {
    'solid': {
        'pattern': None,
        'dashed': 0,
        'description': '实线'
    },
    'dotted': {
        'pattern': '1 1',
        'dashed': 1,
        'description': '点线'
    },
    'dashed_short': {
        'pattern': '4 4',
        'dashed': 1,
        'description': '短虚线'
    },
    'dashed': {
        'pattern': '8 8',
        'dashed': 1,
        'description': '标准虚线'
    },
    'dash_dot': {
        'pattern': '8 4 1 4',
        'dashed': 1,
        'description': '点划线'
    },
    'dash_dot_dot': {
        'pattern': '8 4 1 4 1 4',
        'dashed': 1,
        'description': '双点划线'
    },
    'dashed_long': {
        'pattern': '12 4',
        'dashed': 1,
        'description': '长虚线'
    },
}

# 特殊箭头形状
DRAWIO_ARROW_SHAPES = {
    'flexArrow': {
        'description': '弹性箭头（胖箭头）',
        'has_width': True,
        'default_width': 10
    },
    'link': {
        'description': '链接线',
        'has_width': False
    },
    'curve': {
        'description': '曲线',
        'has_width': False
    },
}

# 旧的简化箭头定义（保持向后兼容）
DRAWIO_ARROWS = {
    'arrow_classic': {
        'style': 'edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=classic;',
        'description': '经典箭头',
        'end_arrow': 'classic'
    },
    'arrow_open': {
        'style': 'edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=open;',
        'description': '开放箭头',
        'end_arrow': 'open'
    },
    'arrow_block': {
        'style': 'edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;',
        'description': '块状箭头',
        'end_arrow': 'block'
    },
    'arrow_diamond': {
        'style': 'edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=diamond;',
        'description': '菱形箭头（聚合）',
        'end_arrow': 'diamond'
    },
    'arrow_none': {
        'style': 'edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=none;',
        'description': '无箭头线条',
        'end_arrow': 'none'
    },
    'dashed_line': {
        'style': 'edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=none;dashed=1;',
        'description': '虚线',
        'end_arrow': 'none',
        'dashed': True
    },
}


# ======================== DrawIO样式属性定义 ========================

DRAWIO_STYLE_ATTRIBUTES = {
    # 形状相关
    'shape': ['rectangle', 'ellipse', 'rhombus', 'triangle', 'hexagon',
              'cylinder3', 'cloud', 'parallelogram', 'trapezoid', 'document',
              'process', 'umlActor', 'umlLifeline', 'flexArrow'],
    
    # 边框相关
    'strokeColor': str,
    'strokeWidth': int,
    'dashed': int,
    'dashPattern': str,
    
    # 填充相关
    'fillColor': str,
    'gradientColor': str,
    'gradientDirection': ['south', 'north', 'east', 'west'],
    'opacity': int,
    
    # 圆角
    'rounded': int,
    'arcSize': int,
    
    # 箭头
    'endArrow': list(DRAWIO_ARROW_HEADS.keys()),
    'startArrow': list(DRAWIO_ARROW_HEADS.keys()),
    'endFill': int,
    'startFill': int,
    'startSize': float,
    'endSize': float,
    
    # 连线
    'edgeStyle': list(DRAWIO_EDGE_STYLES.keys()),
    'curved': int,
    'orthogonalLoop': int,
    'jettySize': ['auto', int],
    
    # flexArrow特有
    'width': int,
    
    # 文字
    'fontSize': int,
    'fontColor': str,
    'fontFamily': str,
    'fontStyle': int,
    'align': ['left', 'center', 'right'],
    'verticalAlign': ['top', 'middle', 'bottom'],
}


# ======================== 箭头属性检测器 ========================

class ArrowAttributeDetector:
    """
    箭头属性检测器
    
    从图像中检测箭头的各种属性：
    - 虚线检测
    - 箭头头部类型检测
    - 线条宽度检测
    - 曲线检测
    - 正交检测
    """
    
    def __init__(self):
        self.arrow_heads = DRAWIO_ARROW_HEADS
        self.dash_patterns = DRAWIO_DASH_PATTERNS
    
    def detect_all_attributes(self, 
                               image: np.ndarray,
                               path_points: List[List[int]] = None,
                               mask: np.ndarray = None) -> Dict[str, Any]:
        """
        检测箭头的所有属性
        
        Args:
            image: BGR格式的箭头图像
            path_points: 路径点列表 [[x, y], ...]
            mask: 箭头掩码（可选）
            
        Returns:
            检测到的属性字典
        """
        attrs = {
            'end_arrow': 'classic',
            'start_arrow': 'none',
            'edge_style': 'orthogonalEdgeStyle',
            'dashed': False,
            'dash_pattern': None,
            'curved': False,
            'stroke_width': 1,
            'stroke_color': '#000000',
            'start_fill': True,
            'end_fill': True,
        }
        
        if image is None or image.size == 0:
            return attrs
        
        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 1. 检测虚线
        dashed_result = self.detect_dashed_line(gray, path_points)
        attrs['dashed'] = dashed_result['is_dashed']
        attrs['dash_pattern'] = dashed_result.get('pattern')
        
        # 2. 检测线条宽度
        attrs['stroke_width'] = self.detect_stroke_width(gray, mask)
        
        # 3. 检测曲线/正交
        if path_points and len(path_points) >= 2:
            curve_result = self.detect_curve_type(path_points)
            attrs['curved'] = curve_result['is_curved']
            if curve_result['is_orthogonal']:
                attrs['edge_style'] = 'orthogonalEdgeStyle'
            else:
                attrs['edge_style'] = 'none'
        
        # 4. 检测箭头头部类型
        head_result = self.detect_arrow_head(image, path_points)
        if head_result:
            attrs['end_arrow'] = head_result.get('end_type', 'classic')
            attrs['start_arrow'] = head_result.get('start_type', 'none')
            attrs['end_fill'] = head_result.get('end_filled', True)
            attrs['start_fill'] = head_result.get('start_filled', True)
        
        # 5. 检测颜色
        attrs['stroke_color'] = self.detect_stroke_color(image, mask)
        
        return attrs
    
    def detect_dashed_line(self, 
                            gray: np.ndarray,
                            path_points: List[List[int]] = None) -> Dict[str, Any]:
        """
        检测是否为虚线，以及虚线模式
        
        策略：沿路径采样，检测亮度的周期性变化
        
        Returns:
            {
                'is_dashed': bool,
                'pattern': str or None (如 '8 8'),
                'pattern_name': str (如 'dashed')
            }
        """
        result = {
            'is_dashed': False,
            'pattern': None,
            'pattern_name': 'solid'
        }
        
        if gray is None or gray.size == 0:
            return result
        
        h, w = gray.shape
        
        # 如果有路径点，沿路径采样
        if path_points and len(path_points) >= 2:
            samples = self._sample_along_path(gray, path_points)
        else:
            # 否则沿中心线采样
            if w > h:
                # 水平箭头
                samples = gray[h // 2, :]
            else:
                # 垂直箭头
                samples = gray[:, w // 2]
        
        if len(samples) < 20:
            return result
        
        # 二值化采样值
        threshold = np.mean(samples)
        binary_samples = (samples < threshold).astype(int)
        
        # 检测周期性变化（寻找峰值）
        # 计算变化点
        changes = np.diff(binary_samples)
        change_indices = np.where(changes != 0)[0]
        
        if len(change_indices) < 4:
            return result  # 变化太少，不是虚线
        
        # 计算间隔
        intervals = np.diff(change_indices)
        
        if len(intervals) < 2:
            return result
        
        # 检测周期性
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        # 如果间隔比较规则，说明是虚线
        if std_interval < mean_interval * 0.5 and mean_interval > 3:
            result['is_dashed'] = True
            
            # 估算虚线模式
            # 分析线段长度和间隙长度
            on_lengths = []
            off_lengths = []
            
            is_on = binary_samples[0] == 1
            current_length = 0
            
            for i, val in enumerate(binary_samples):
                if (val == 1) == is_on:
                    current_length += 1
                else:
                    if is_on:
                        on_lengths.append(current_length)
                    else:
                        off_lengths.append(current_length)
                    is_on = not is_on
                    current_length = 1
            
            # 最后一段
            if is_on:
                on_lengths.append(current_length)
            else:
                off_lengths.append(current_length)
            
            if on_lengths and off_lengths:
                avg_on = np.mean(on_lengths)
                avg_off = np.mean(off_lengths)
                
                # 匹配最接近的模式
                result['pattern'], result['pattern_name'] = self._match_dash_pattern(avg_on, avg_off)
        
        return result
    
    def _sample_along_path(self, gray: np.ndarray, path_points: List[List[int]]) -> np.ndarray:
        """沿路径采样像素值"""
        samples = []
        h, w = gray.shape
        
        for i in range(len(path_points) - 1):
            p1 = path_points[i]
            p2 = path_points[i + 1]
            
            # 在两点之间采样
            num_samples = max(10, int(np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)))
            
            for t in np.linspace(0, 1, num_samples):
                x = int(p1[0] + t * (p2[0] - p1[0]))
                y = int(p1[1] + t * (p2[1] - p1[1]))
                
                if 0 <= x < w and 0 <= y < h:
                    samples.append(gray[y, x])
        
        return np.array(samples)
    
    def _match_dash_pattern(self, avg_on: float, avg_off: float) -> Tuple[str, str]:
        """匹配最接近的虚线模式"""
        # 计算比例
        ratio = avg_on / avg_off if avg_off > 0 else 1
        
        if avg_on < 3 and avg_off < 3:
            return '1 1', 'dotted'
        elif 0.8 < ratio < 1.2:
            if avg_on < 6:
                return '4 4', 'dashed_short'
            else:
                return '8 8', 'dashed'
        elif ratio > 1.5:
            return '12 4', 'dashed_long'
        else:
            # 可能是点划线
            return '8 4 1 4', 'dash_dot'
    
    def detect_stroke_width(self, 
                            gray: np.ndarray,
                            mask: np.ndarray = None) -> int:
        """
        检测线条宽度
        
        策略：骨架化后计算与边缘的距离
        """
        if gray is None or gray.size == 0:
            return 1
        
        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        if mask is not None:
            binary = cv2.bitwise_and(binary, mask)
        
        # 检查是否有足够的前景像素
        if np.count_nonzero(binary) < 10:
            return 1
        
        # 使用距离变换
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # 获取骨架上的距离值
        try:
            from skimage.morphology import skeletonize
            skeleton = skeletonize(binary > 0)
            skeleton_distances = dist_transform[skeleton]
            
            if len(skeleton_distances) > 0:
                # 线条宽度约为距离的2倍
                width = int(np.median(skeleton_distances) * 2)
                # 限制到 1-2 范围（降低上限以匹配原图）
                return max(1, min(width, 2))
        except:
            pass
        
        # 降级方法：使用最大距离
        max_dist = dist_transform.max()
        # 限制到 1-2 范围
        return max(1, min(int(max_dist * 2), 2))
    
    def detect_curve_type(self, path_points: List[List[int]]) -> Dict[str, Any]:
        """
        检测曲线类型
        
        策略：
        1. 计算路径点的曲率
        2. 检测是否只有水平/垂直段（正交）
        
        Returns:
            {
                'is_curved': bool,
                'is_orthogonal': bool,
                'curvature': float
            }
        """
        result = {
            'is_curved': False,
            'is_orthogonal': False,
            'curvature': 0.0
        }
        
        if not path_points or len(path_points) < 3:
            return result
        
        points = np.array(path_points)
        
        # 计算角度变化
        angles = []
        for i in range(1, len(points) - 1):
            v1 = points[i] - points[i-1]
            v2 = points[i+1] - points[i]
            
            # 计算角度
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 0 and norm2 > 0:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.degrees(np.arccos(cos_angle))
                angles.append(angle)
        
        if not angles:
            return result
        
        # 检测正交性
        # 正交路径的角度应该是90度或180度
        is_orthogonal = all(
            abs(a - 90) < 15 or abs(a - 180) < 15 or a < 15
            for a in angles
        )
        result['is_orthogonal'] = is_orthogonal
        
        # 检测曲线
        # 如果角度变化平滑，说明是曲线
        if len(angles) >= 3:
            angle_changes = np.diff(angles)
            smoothness = np.std(angle_changes)
            
            # 平滑的角度变化 + 存在非直角角度 = 曲线
            if smoothness < 30 and any(15 < a < 75 or 105 < a < 165 for a in angles):
                result['is_curved'] = True
        
        # 计算平均曲率
        result['curvature'] = np.mean(angles) if angles else 0
        
        return result
    
    def detect_arrow_head(self,
                          image: np.ndarray,
                          path_points: List[List[int]] = None) -> Optional[Dict]:
        """
        检测箭头头部类型
        
        策略：
        1. 提取端点区域
        2. 轮廓分析（三角形/菱形/圆形等）
        
        Returns:
            {
                'end_type': str,
                'start_type': str,
                'end_filled': bool,
                'start_filled': bool
            }
        """
        result = {
            'end_type': 'classic',
            'start_type': 'none',
            'end_filled': True,
            'start_filled': True
        }
        
        if image is None or image.size == 0:
            return result
        
        h, w = image.shape[:2]
        
        # 确定端点位置
        if path_points and len(path_points) >= 2:
            start_point = path_points[0]
            end_point = path_points[-1]
        else:
            # 假设水平箭头
            start_point = [0, h // 2]
            end_point = [w, h // 2]
        
        # 提取终点区域
        end_region = self._extract_endpoint_region(image, end_point, is_end=True)
        if end_region is not None and end_region.size > 0:
            end_type, end_filled = self._analyze_arrow_head(end_region)
            result['end_type'] = end_type
            result['end_filled'] = end_filled
        
        # 提取起点区域
        start_region = self._extract_endpoint_region(image, start_point, is_end=False)
        if start_region is not None and start_region.size > 0:
            start_type, start_filled = self._analyze_arrow_head(start_region)
            # 如果起点有箭头形状
            if start_type != 'none':
                result['start_type'] = start_type
                result['start_filled'] = start_filled
        
        return result
    
    def _extract_endpoint_region(self, 
                                  image: np.ndarray,
                                  point: List[int],
                                  is_end: bool,
                                  size: int = 30) -> Optional[np.ndarray]:
        """提取端点区域"""
        h, w = image.shape[:2]
        x, y = int(point[0]), int(point[1])
        
        # 根据是起点还是终点调整提取区域
        half_size = size // 2
        
        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(w, x + half_size)
        y2 = min(h, y + half_size)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        return image[y1:y2, x1:x2]
    
    def _analyze_arrow_head(self, region: np.ndarray) -> Tuple[str, bool]:
        """
        分析箭头头部形状
        
        Returns:
            (arrow_type, is_filled)
        """
        if region is None or region.size == 0:
            return 'none', True
        
        # 转灰度
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region
        
        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 'none', True
        
        # 取最大轮廓
        main_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(main_contour)
        
        if area < 20:
            return 'none', True
        
        # 多边形近似
        peri = cv2.arcLength(main_contour, True)
        approx = cv2.approxPolyDP(main_contour, 0.04 * peri, True)
        vertices = len(approx)
        
        # 计算填充率
        x, y, rw, rh = cv2.boundingRect(main_contour)
        bbox_area = rw * rh
        fill_ratio = area / bbox_area if bbox_area > 0 else 0
        
        # 检测是否为空心（只有轮廓）
        # 检查内部是否有空洞
        inner_contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        is_hollow = len(inner_contours) > len(contours)
        is_filled = not is_hollow and fill_ratio > 0.4
        
        # 根据顶点数和形状判断类型
        if vertices == 3:
            # 三角形
            if is_filled:
                return 'classic', True
            else:
                return 'open', False
        
        elif vertices == 4:
            # 检查是否是菱形
            if 0.4 < fill_ratio < 0.6:
                return 'diamond', is_filled
            else:
                return 'block', is_filled
        
        elif vertices > 6:
            # 可能是圆形
            circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0
            if circularity > 0.7:
                return 'oval', is_filled
        
        elif vertices == 2:
            # 短线
            return 'dash', False
        
        # 默认
        return 'classic', is_filled
    
    def detect_stroke_color(self,
                            image: np.ndarray,
                            mask: np.ndarray = None) -> str:
        """
        检测线条颜色
        
        Returns:
            十六进制颜色字符串
        """
        if image is None or image.size == 0:
            return '#000000'
        
        if len(image.shape) == 2:
            # 灰度图
            return '#000000'
        
        # 提取前景像素
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        if mask is not None:
            binary = cv2.bitwise_and(binary, mask)
        
        # 获取前景像素的颜色
        foreground_pixels = image[binary > 0]
        
        if len(foreground_pixels) == 0:
            return '#000000'
        
        # 取中值颜色
        median_color = np.median(foreground_pixels, axis=0).astype(int)
        
        # BGR -> RGB -> Hex
        r, g, b = median_color[2], median_color[1], median_color[0]
        return "#{:02x}{:02x}{:02x}".format(r, g, b)


# ======================== 形状识别函数 ========================

class DrawIOLibrary:
    """DrawIO元素库管理器"""
    
    def __init__(self, icons_dir: str = None):
        """
        初始化
        
        Args:
            icons_dir: 自定义图标目录（可选，用于图标模板匹配）
        """
        self.shapes = ALL_DRAWIO_SHAPES
        self.arrows = DRAWIO_ARROWS
        self.arrow_heads = DRAWIO_ARROW_HEADS
        self.edge_styles = DRAWIO_EDGE_STYLES
        self.dash_patterns = DRAWIO_DASH_PATTERNS
        self.icons_dir = icons_dir
        self._icon_templates = None
    
    def get_shape_style(self, shape_name: str) -> Optional[str]:
        """获取形状的DrawIO样式"""
        shape = self.shapes.get(shape_name.lower())
        return shape['style'] if shape else None
    
    def get_arrow_style(self, arrow_type: str) -> Optional[str]:
        """获取箭头的DrawIO样式"""
        arrow = self.arrows.get(arrow_type.lower())
        return arrow['style'] if arrow else None
    
    def get_arrow_head_info(self, head_type: str) -> Optional[Dict]:
        """获取箭头头部信息"""
        return self.arrow_heads.get(head_type)
    
    def list_shapes(self) -> List[str]:
        """列出所有支持的形状"""
        return list(self.shapes.keys())
    
    def list_arrows(self) -> List[str]:
        """列出所有支持的箭头类型"""
        return list(self.arrows.keys())
    
    def list_arrow_heads(self) -> List[str]:
        """列出所有支持的箭头头部类型"""
        return list(self.arrow_heads.keys())
    
    def list_edge_styles(self) -> List[str]:
        """列出所有支持的连线样式"""
        return list(self.edge_styles.keys())
    
    def list_dash_patterns(self) -> List[str]:
        """列出所有支持的虚线模式"""
        return list(self.dash_patterns.keys())
    
    def match_shape(self, 
                    contour: np.ndarray,
                    aspect_ratio: float = None) -> Optional[Dict]:
        """
        根据轮廓匹配形状
        
        Args:
            contour: OpenCV轮廓
            aspect_ratio: 宽高比（可选）
            
        Returns:
            匹配结果或None
        """
        if contour is None or len(contour) < 3:
            return None
        
        # 多边形近似
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        vertices = len(approx)
        
        # 计算宽高比
        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / h if h > 0 else 1
        
        # 计算轮廓面积与边界框面积的比值（用于判断形状）
        contour_area = cv2.contourArea(contour)
        bbox_area = w * h
        area_ratio = contour_area / bbox_area if bbox_area > 0 else 0
        
        # 匹配规则
        matches = []
        
        # 圆形/椭圆
        if 0.7 < area_ratio < 0.85:
            if 0.9 < aspect < 1.1:
                matches.append(('circle', 0.9))
            else:
                matches.append(('ellipse', 0.8))
        
        # 矩形
        elif vertices == 4:
            if area_ratio > 0.95:
                matches.append(('rectangle', 0.85))
            elif 0.85 < area_ratio < 0.95:
                matches.append(('rounded_rectangle', 0.8))
        
        # 菱形
        if vertices == 4 and 0.45 < area_ratio < 0.55:
            matches.append(('diamond', 0.85))
        
        # 三角形
        if vertices == 3:
            matches.append(('triangle', 0.9))
        
        # 六边形
        if vertices == 6:
            matches.append(('hexagon', 0.85))
        
        # 平行四边形/梯形
        if vertices == 4 and 0.6 < area_ratio < 0.85:
            matches.append(('parallelogram', 0.7))
        
        # 返回最佳匹配
        if matches:
            matches.sort(key=lambda x: x[1], reverse=True)
            best_name, score = matches[0]
            return {
                'name': best_name,
                'score': score,
                'style': self.shapes[best_name]['style'],
                'description': self.shapes[best_name]['description']
            }
        
        return None


# ======================== 箭头样式构建函数 ========================

def build_arrow_style(
    end_arrow: str = 'classic',
    start_arrow: str = 'none',
    edge_style: str = 'orthogonalEdgeStyle',
    dashed: bool = False,
    dash_pattern: str = None,
    curved: bool = False,
    rounded: bool = False,
    stroke_width: int = 1,
    stroke_color: str = '#000000',
    start_fill: bool = True,
    end_fill: bool = True,
    start_size: float = None,
    end_size: float = None,
    shape: str = None,
    width: int = None,
    curve_type: str = None,
    **kwargs
) -> str:
    """
    构建DrawIO箭头样式字符串
    
    Args:
        end_arrow: 终点箭头类型（见 DRAWIO_ARROW_HEADS）
        start_arrow: 起点箭头类型
        edge_style: 连线样式（见 DRAWIO_EDGE_STYLES）
        dashed: 是否虚线
        dash_pattern: 虚线模式（如 '8 8'）
        curved: 是否曲线
        rounded: 是否圆角转弯
        stroke_width: 线条宽度
        stroke_color: 线条颜色
        start_fill: 起点箭头是否填充
        end_fill: 终点箭头是否填充
        start_size: 起点箭头大小
        end_size: 终点箭头大小
        shape: 特殊形状（如 'flexArrow'）
        width: flexArrow的宽度
        curve_type: 曲线类型（'sharp'/'rounded'/'curved'），会覆盖curved和rounded参数
        **kwargs: 额外属性
        
    Returns:
        完整的样式字符串
    """
    # 处理 curve_type 参数
    if curve_type:
        if curve_type == 'curved':
            curved = True
            rounded = False
        elif curve_type == 'rounded':
            curved = False
            rounded = True
        elif curve_type == 'sharp':
            curved = False
            rounded = False
    
    style_parts = ['html=1']
    
    # 连线样式
    if edge_style and edge_style != 'none':
        style_parts.append(f'edgeStyle={edge_style}')
    
    # 特殊形状
    if shape:
        style_parts.append(f'shape={shape}')
        if width and shape == 'flexArrow':
            style_parts.append(f'width={width}')
    
    # 箭头
    style_parts.append(f'endArrow={end_arrow}')
    style_parts.append(f'startArrow={start_arrow}')
    
    # 箭头填充
    if not start_fill:
        style_parts.append('startFill=0')
    if not end_fill:
        style_parts.append('endFill=0')
    
    # 箭头大小
    if start_size is not None:
        style_parts.append(f'startSize={start_size}')
    if end_size is not None:
        style_parts.append(f'endSize={end_size}')
    
    # 虚线
    if dashed:
        style_parts.append('dashed=1')
        if dash_pattern:
            style_parts.append(f'dashPattern={dash_pattern}')
    
    # 曲线/圆角
    if curved:
        style_parts.append('curved=1')
    if rounded:
        style_parts.append('rounded=1')
    else:
        style_parts.append('rounded=0')
    
    # 线条
    style_parts.append(f'strokeWidth={stroke_width}')
    style_parts.append(f'strokeColor={stroke_color}')
    
    # 正交循环
    style_parts.append('orthogonalLoop=1')
    style_parts.append('jettySize=auto')
    
    # 额外属性
    for key, value in kwargs.items():
        if value is not None:
            style_parts.append(f'{key}={value}')
    
    return ';'.join(style_parts) + ';'


def build_style_string(base_style: str, **attributes) -> str:
    """
    构建DrawIO样式字符串
    
    Args:
        base_style: 基础样式
        **attributes: 要添加的属性
            例如: fillColor='#ff0000', strokeWidth=2
            
    Returns:
        完整的样式字符串
    """
    style = base_style.rstrip(';') + ';'
    
    for key, value in attributes.items():
        if value is not None:
            style += f"{key}={value};"
    
    return style


# ======================== 元素匹配函数 ========================

def match_element_to_drawio(element: Any,
                            image: np.ndarray = None,
                            mask: np.ndarray = None) -> Optional[Dict]:
    """
    将检测到的元素匹配到DrawIO原生形状
    
    Args:
        element: 检测到的元素（需要有 bbox 和可选的 element_type）
        image: 原始图像（可选，用于更精确的匹配）
        mask: 元素掩码（可选，用于形状分析）
        
    Returns:
        匹配结果字典或None
    """
    library = DrawIOLibrary()
    
    # 如果已有类型，直接查找样式
    if hasattr(element, 'element_type'):
        elem_type = element.element_type.lower().replace(' ', '_')
        
        # 尝试直接匹配
        style = library.get_shape_style(elem_type)
        if style:
            return {
                'name': elem_type,
                'style': style,
                'match_type': 'direct'
            }
    
    # 如果有mask，分析形状
    if mask is not None and np.count_nonzero(mask) > 0:
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                        cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            match = library.match_shape(main_contour)
            if match:
                match['match_type'] = 'contour_analysis'
                return match
    
    return None


def detect_arrow_style(arrow_head_image: np.ndarray) -> Optional[str]:
    """
    检测箭头头部样式（简化版）
    
    Args:
        arrow_head_image: 箭头头部区域图像
        
    Returns:
        箭头样式名称 ('classic', 'open', 'block', 'diamond', 'none')
    """
    detector = ArrowAttributeDetector()
    result = detector._analyze_arrow_head(arrow_head_image)
    return result[0] if result else 'classic'


def detect_arrow_attributes(image: np.ndarray,
                            path_points: List[List[int]] = None,
                            mask: np.ndarray = None) -> Dict[str, Any]:
    """
    从图像中检测完整的箭头属性
    
    Args:
        image: 箭头图像（BGR格式）
        path_points: 路径点列表
        mask: 箭头掩码
        
    Returns:
        属性字典，可直接传给 build_arrow_style()
    """
    detector = ArrowAttributeDetector()
    return detector.detect_all_attributes(image, path_points, mask)


# ======================== 快捷函数 ========================

def get_drawio_style(element_type: str,
                      fill_color: str = None,
                      stroke_color: str = None,
                      stroke_width: int = None) -> str:
    """
    获取元素的完整DrawIO样式
    
    Args:
        element_type: 元素类型
        fill_color: 填充颜色（十六进制）
        stroke_color: 边框颜色（十六进制）
        stroke_width: 边框宽度
        
    Returns:
        完整的DrawIO样式字符串
    """
    library = DrawIOLibrary()
    base_style = library.get_shape_style(element_type)
    
    if base_style is None:
        base_style = 'rounded=0;whiteSpace=wrap;html=1;'
    
    return build_style_string(
        base_style,
        fillColor=fill_color,
        strokeColor=stroke_color,
        strokeWidth=stroke_width
    )


def get_all_arrow_head_types() -> Dict[str, Dict]:
    """获取所有箭头头部类型及其信息"""
    return DRAWIO_ARROW_HEADS.copy()


def get_all_dash_patterns() -> Dict[str, Dict]:
    """获取所有虚线模式及其信息"""
    return DRAWIO_DASH_PATTERNS.copy()


def get_all_edge_styles() -> Dict[str, Dict]:
    """获取所有连线样式及其信息"""
    return DRAWIO_EDGE_STYLES.copy()
