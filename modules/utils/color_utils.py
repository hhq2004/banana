"""
颜色处理工具函数
"""

import numpy as np
import cv2
from typing import Tuple, List


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """RGB转十六进制颜色"""
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """十六进制转RGB"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def extract_fill_stroke_colors(image: np.ndarray, 
                                bbox: List[int]) -> Tuple[str, str]:
    """
    提取填充色和描边色
    
    Args:
        image: BGR格式的OpenCV图像
        bbox: [x1, y1, x2, y2]
        
    Returns:
        (fill_color_hex, stroke_color_hex)
    """
    x1, y1, x2, y2 = map(int, bbox)
    h_box, w_box = y2 - y1, x2 - x1
    
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return "#ffffff", "#000000"
    
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    # 提取填充色（内部区域中值）
    margin_x = max(5, int(w_box * 0.2))
    margin_y = max(5, int(h_box * 0.2))
    
    inner_roi = roi_rgb[margin_y:h_box-margin_y, margin_x:w_box-margin_x]
    if inner_roi.size == 0:
        inner_roi = roi_rgb
    
    inner_pixels = inner_roi.reshape(-1, 3)
    fill_rgb = tuple(np.median(inner_pixels, axis=0).astype(int))
    
    # 提取描边色（边缘区域暗色）
    border_width = max(2, int(min(w_box, h_box) * 0.1))
    
    top = roi_rgb[:border_width, :]
    bottom = roi_rgb[h_box-border_width:, :]
    left = roi_rgb[:, :border_width]
    right = roi_rgb[:, w_box-border_width:]
    
    border_pixels = np.concatenate([
        top.reshape(-1, 3),
        bottom.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3)
    ], axis=0)
    
    luminance = np.dot(border_pixels, [0.299, 0.587, 0.114])
    dark_threshold = np.percentile(luminance, 25)
    darker_pixels = border_pixels[luminance <= dark_threshold]
    
    if len(darker_pixels) > 0:
        stroke_rgb = tuple(np.mean(darker_pixels, axis=0).astype(int))
    else:
        stroke_rgb = tuple(np.mean(border_pixels, axis=0).astype(int))
    
    return rgb_to_hex(fill_rgb), rgb_to_hex(stroke_rgb)


def get_dominant_color(image: np.ndarray, 
                       bbox: List[int] = None,
                       n_clusters: int = 3) -> str:
    """
    使用K-Means提取主色调
    
    Args:
        image: BGR格式图像
        bbox: 可选的区域 [x1, y1, x2, y2]
        n_clusters: 聚类数量
        
    Returns:
        主色调的十六进制表示
    """
    if bbox:
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]
    else:
        roi = image
    
    if roi.size == 0:
        return "#ffffff"
    
    # 转为RGB
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pixels = roi_rgb.reshape(-1, 3).astype(np.float32)
    
    # K-Means聚类
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    
    # 找最大的簇
    label_counts = np.bincount(labels.flatten())
    dominant_idx = np.argmax(label_counts)
    dominant_color = centers[dominant_idx].astype(int)
    
    return rgb_to_hex(tuple(dominant_color))
