"""
图像处理工具函数
"""

import io
import base64
from typing import List, Tuple, Optional
from PIL import Image
import numpy as np
from typing import Optional

def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    将PIL图像转换为base64字符串
    
    Args:
        image: PIL图像
        format: 图像格式
        
    Returns:
        base64编码字符串
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_image(b64_str: str, target_mode: Optional[str] = None) -> Image.Image:
    """Decode a base64 string back into an image, preserving mode when requested."""
    img_bytes = base64.b64decode(b64_str)
    buffer = io.BytesIO(img_bytes)
    with Image.open(buffer) as img:
        restored = img.copy()

    if target_mode and restored.mode != target_mode:
        restored = restored.convert(target_mode)

    return restored

def crop_with_padding(image: Image.Image, 
                      bbox: List[int], 
                      padding: int = 5) -> Tuple[Image.Image, List[int]]:
    """
    带padding的图像裁剪
    
    Args:
        image: PIL图像
        bbox: [x1, y1, x2, y2]
        padding: 边缘padding
        
    Returns:
        (裁剪后的图像, 新的bbox)
    """
    img_w, img_h = image.size
    x1, y1, x2, y2 = bbox
    
    # 添加padding并限制边界
    x1_p = max(0, x1 - padding)
    y1_p = max(0, y1 - padding)
    x2_p = min(img_w, x2 + padding)
    y2_p = min(img_h, y2 + padding)
    
    cropped = image.crop((x1_p, y1_p, x2_p, y2_p))
    new_bbox = [x1_p, y1_p, x2_p, y2_p]
    
    return cropped, new_bbox


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """
    计算两个边界框的IoU
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        IoU值 (0-1)
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def calculate_image_complexity(image: np.ndarray) -> Tuple[float, float]:
    """
    计算图像复杂度
    
    Args:
        image: BGR格式OpenCV图像
        
    Returns:
        (拉普拉斯方差, 标准差)
    """
    import cv2
    
    if image.size == 0:
        return 0.0, 0.0
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    std_dev = np.std(gray)
    
    return laplacian_var, std_dev
