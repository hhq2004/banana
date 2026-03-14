"""
坐标处理器模块

功能：
    将 OCR 返回的多边形坐标转换为 draw.io 的几何坐标。
    支持横排文字和竖排文字（旋转角度检测）。

坐标系统：
    - OCR 返回：四边形顶点坐标 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    - draw.io 需要：左上角坐标 (x, y) + 宽高 (width, height) + 旋转角度

转换原理：
    1. 计算四边形的中心点
    2. 计算两条边的长度，确定宽度和高度
    3. 计算旋转角度（通过第一条边的倾斜方向）
    4. draw.io 以中心点为旋转轴，所以坐标需要调整

             P0 -------- P1
              |          |
              |  center  |
              |          |
             P3 -------- P2
"""

import math
from dataclasses import dataclass


@dataclass
class NormalizedCoords:
    """
    归一化后的坐标
    
    Attributes:
        x: 左上角 X 坐标
        y: 左上角 Y 坐标
        width: 宽度
        height: 高度
        baseline_y: 基线 Y 坐标（用于文字对齐）
        rotation: 旋转角度（度，顺时针为正）
    """
    x: float
    y: float
    width: float
    height: float
    baseline_y: float
    rotation: float


class CoordProcessor:
    """
    坐标处理器
    
    将源图像坐标转换为目标画布坐标，支持缩放。
    
    使用示例：
        processor = CoordProcessor(source_width=1920, source_height=1080)
        geometry = processor.polygon_to_geometry(polygon)
        # geometry = {"x": 100, "y": 200, "width": 80, "height": 20, "rotation": 0}
    """
    
    def __init__(self, source_width: int, source_height: int,
                 canvas_width: int = None, canvas_height: int = None):
        """
        初始化坐标处理器
        
        Args:
            source_width: 源图像宽度（像素）
            source_height: 源图像高度（像素）
            canvas_width: 目标画布宽度，默认与源图像相同
            canvas_height: 目标画布高度，默认与源图像相同
        """
        self.source_width = source_width
        self.source_height = source_height
        self.canvas_width = canvas_width if canvas_width is not None else source_width
        self.canvas_height = canvas_height if canvas_height is not None else source_height
        
        # 计算缩放比例
        self.scale_x = self.canvas_width / source_width
        self.scale_y = self.canvas_height / source_height
        # 使用统一缩放，保持宽高比
        self.uniform_scale = min(self.scale_x, self.scale_y)
    
    def normalize_polygon(self, polygon: list[tuple[float, float]]) -> NormalizedCoords:
        """
        将多边形坐标归一化到目标画布
        
        Args:
            polygon: 四边形顶点坐标 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
            
        Returns:
            NormalizedCoords: 归一化后的坐标
        """
        if len(polygon) < 4:
            return NormalizedCoords(0, 0, 0, 0, 0, 0)
        
        # 缩放坐标
        normalized_points = [
            (p[0] * self.uniform_scale, p[1] * self.uniform_scale)
            for p in polygon
        ]
        
        p0, p1, p2, p3 = normalized_points[:4]
        
        # 计算旋转角度（通过上边的倾斜方向）
        rotation = self._calculate_rotation(p0, p1)
        
        # 计算中心点
        center_x = sum(p[0] for p in normalized_points) / 4
        center_y = sum(p[1] for p in normalized_points) / 4
        
        # 计算边长
        edge_top = math.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)     # 上边（宽度方向）
        edge_left = math.sqrt((p3[0] - p0[0])**2 + (p3[1] - p0[1])**2)    # 左边（高度方向）
        
        # 判断是否竖排（旋转角度接近 ±90°）
        is_vertical = abs(abs(rotation) - 90) < 15
        
        if is_vertical:
            width = edge_top
            height = edge_left
        else:
            width = edge_top
            height = edge_left
        
        # 计算左上角坐标（draw.io 从左上角定位）
        x = center_x - width / 2
        y = center_y - height / 2
        
        # 计算基线位置
        baseline_y = (p2[1] + p3[1]) / 2
        
        return NormalizedCoords(
            x=x, y=y, width=width, height=height,
            baseline_y=baseline_y, rotation=rotation
        )
    
    def _calculate_rotation(self, p0: tuple, p1: tuple) -> float:
        """
        计算旋转角度
        
        通过上边（P0 到 P1）的方向向量计算旋转角度。
        水平向右为 0°，顺时针旋转为正角度。
        
        Args:
            p0: 左上角坐标
            p1: 右上角坐标
            
        Returns:
            旋转角度（度）
        """
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        
        if dx == 0:
            return 90.0 if dy > 0 else -90.0
        
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        
        # 小角度忽略（避免微小误差）
        if abs(angle_deg) < 2:
            return 0.0
        
        return round(angle_deg, 1)
    
    def polygon_to_geometry(self, polygon: list[tuple[float, float]]) -> dict:
        """
        将多边形转换为 draw.io geometry 格式
        
        Args:
            polygon: 四边形顶点坐标
            
        Returns:
            dict: {"x", "y", "width", "height", "baseline_y", "rotation"}
        """
        coords = self.normalize_polygon(polygon)
        
        return {
            "x": round(coords.x, 2),
            "y": round(coords.y, 2),
            "width": round(coords.width, 2),
            "height": round(coords.height, 2),
            "baseline_y": round(coords.baseline_y, 2),
            "rotation": coords.rotation
        }
if __name__ == "__main__":
    # 测试代码
    processor = CoordProcessor(source_width=2000, source_height=1500)
    
    # 测试横排文本框
    test_polygon = [
        (100, 200),   # 左上
        (300, 200),   # 右上
        (300, 250),   # 右下
        (100, 250)    # 左下
    ]
    
    result = processor.normalize_polygon(test_polygon)
    print(f"横排文本归一化结果:")
    print(f"  位置: ({result.x:.2f}, {result.y:.2f})")
    print(f"  尺寸: {result.width:.2f} x {result.height:.2f}")
    print(f"  旋转: {result.rotation}°")
    
    geometry = processor.polygon_to_geometry(test_polygon)
    print(f"  Geometry: {geometry}")