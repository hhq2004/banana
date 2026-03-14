"""
Pix2Text OCR 模块

功能：
    使用 Pix2Text 识别图像中的数学公式。
    作为 Azure OCR 的补充，专门处理 LaTeX 公式识别。

安装：
    pip install pix2text
    
    注意：首次使用会自动下载模型文件（约 1GB），需要网络连接。
    GPU 加速需要安装 onnxruntime-gpu。

返回格式：
    - text: 公式内容（自动添加 $ 符号）
    - polygon: 四边形坐标
    - block_type: 类型（formula, text, isolated, embedding）
    - is_latex: 是否为公式

使用场景：
    当 Azure OCR 无法识别数学公式时，使用 Pix2Text 补充识别。
    在 main.py 中使用 --pix2text 参数启用。
"""

import math
from pathlib import Path
from dataclasses import dataclass, field

from pix2text import Pix2Text


@dataclass
class Pix2TextBlock:
    """
    Pix2Text 识别块
    
    Attributes:
        text: 文字/公式内容
        polygon: 四边形坐标
        block_type: 类型（text/formula/isolated/embedding）
        font_size_px: 字号（像素）
        is_latex: 是否为 LaTeX 公式
    """
    text: str
    polygon: list[tuple[float, float]]
    block_type: str
    font_size_px: float
    is_latex: bool = False


@dataclass
class Pix2TextResult:
    """Pix2Text 识别结果"""
    image_width: int
    image_height: int
    blocks: list[Pix2TextBlock] = field(default_factory=list)


class Pix2TextOCR:
    """
    Pix2Text OCR 客户端
    
    专门用于识别数学公式，返回 LaTeX 格式。
    
    使用示例：
        ocr = Pix2TextOCR()
        result = ocr.analyze_image("input.png")
        for block in result.blocks:
            if block.is_latex:
                print(f"公式: {block.text}")
    """
    
    # def __init__(self, device: str = 'cuda', languages: tuple = ('en',)):
    #     """
    #     初始化 Pix2Text
        
    #     Args:
    #         device: 计算设备（cuda 使用 GPU3）
    #         languages: 文字语言（影响非公式部分的识别）
    #     """
    #     print(f"   Pix2Text 使用设备: {device}")
        
    #     # 降低 MFD 检测阈值，提高公式检测率
    #     self.p2t = Pix2Text.from_config(
    #         device=device,
    #         text_config={'device': device, 'languages': languages},
    #         formula_config={'device': device},
    #         # MFD 配置：降低置信度阈值
    #         mfd_config={
    #             'device': device,
    #             'conf_threshold': 0.15,  # 默认 0.25，降低以检测更多公式
    #             'iou_threshold': 0.45,   # 默认 0.45
    #         },
    #     )

    def __init__(self, device: str = 'cuda', languages: tuple = ('en',)):
        """
        初始化 Pix2Text
        
        Args:
            device: 计算设备（cuda 使用 GPU3）
            languages: 文字语言（影响非公式部分的识别）
        """
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        from config.read_config import load_config

        print(f"   Pix2Text 使用设备: {device}")

        # 从配置文件读取模型路径
        config = load_config()
        p2t_config = config.get('text', {}).get('pix2text', {})
        
        import os
        if p2t_config.get('cnocr_home'):
            os.environ['CNOCR_HOME'] = p2t_config['cnocr_home']
        if p2t_config.get('cnstd_home'):
            os.environ['CNSTD_HOME'] = p2t_config['cnstd_home']

        formula_cfg = {'device': device}
        mfd_cfg = {
            'device': device,
            'conf_threshold': 0.15,
            'iou_threshold': 0.45,
        }

        # 如果配置了显式模型路径，使用它们（避免运行时自动下载）
        if p2t_config.get('formula_model_dir'):
            formula_cfg['model_dir'] = p2t_config['formula_model_dir']
        if p2t_config.get('mfd_model_path'):
            mfd_cfg['model_path'] = p2t_config['mfd_model_path']

        self.p2t = Pix2Text.from_config(
            device=device,
            text_config={'device': device, 'languages': languages},
            formula_config=formula_cfg,
            mfd_config=mfd_cfg,
        )
    
    def analyze_image(self, image_path: str) -> Pix2TextResult:
        """
        分析图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            Pix2TextResult: 识别结果（包含文字和公式）
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        # 获取图像尺寸
        from PIL import Image
        with Image.open(image_path) as img:
            image_width, image_height = img.size
        
        # 识别（使用较低的 resized_shape 提高检测率）
        # result = self.p2t.recognize(str(image_path), resized_shape=1600, return_text=False)
        # 暂时只使用 recognize_formula 针对特定区域识别，全图识别容易产生混乱
        # 如果需要全图识别，请恢复上方代码
        
        result = self.p2t.recognize(str(image_path), resized_shape=1600, return_text=False)
        
        # 解析结果
        blocks = []
        type_counts = {}
        
        print(f"   Pix2Text 原始识别结果 (共 {len(result)} 项):")
        
        for i, item in enumerate(result):
            block_type = item.get('type', 'text')
            text = item.get('text', '')
            position = item.get('position', [])
            
            # 打印原始识别内容
            print(f"      {i+1}. [{block_type}] \"{text}\"")
            
            # 统计类型
            type_counts[block_type] = type_counts.get(block_type, 0) + 1
            
            polygon = self._convert_position(position)
            font_size_px = self._estimate_font_size(polygon)
            
            # 判断是否为公式（formula, isolated, embedding 都是公式类型）
            is_latex = (block_type in ['formula', 'isolated', 'embedding'])
            
            # 为公式添加 $ 符号
            if is_latex and text and not text.startswith('$'):
                text = f"${text}$"
            
            block = Pix2TextBlock(
                text=text,
                polygon=polygon,
                block_type=block_type,
                font_size_px=font_size_px,
                is_latex=is_latex
            )
            blocks.append(block)
        
        # 打印类型统计
        print(f"   Pix2Text 块类型统计: {type_counts}")
        
        return Pix2TextResult(image_width=image_width, image_height=image_height, blocks=blocks)

    def recognize_region(self, image_path: str, polygon: list, save_debug_crop: bool = False) -> str:
        """
        识别特定区域的公式
        
        Args:
            image_path: 图片路径
            polygon: [x1, y1, x2, y2...] or [(x,y), ...]
            save_debug_crop: 是否保存调试裁剪图
            
        Returns:
            LaTex 字符串，如果识别失败或置信度低返回 None
        """
        from PIL import Image, ImageEnhance, ImageFilter
        import numpy as np
        
        try:
            # 读取图片
            img = Image.open(image_path).convert('RGB')
            
            # 处理 Polygon
            if not polygon:
                return None
            
            # 转换为 bbox (minx, miny, maxx, maxy)
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            
            # 计算原始 bbox 尺寸
            orig_width = max(xs) - min(xs)
            orig_height = max(ys) - min(ys)
            
            # 动态 padding：根据文字高度调整，确保不会切到下标/上标
            # 但避免过大导致包含其他行
            base_padding = 5  # 基础 padding
            vertical_padding = max(base_padding, int(orig_height * 0.15))  # 垂直 padding 15% (降低避免包含其他行)
            horizontal_padding = max(base_padding, int(orig_width * 0.1))  # 水平 padding 10%
            
            bbox = (
                max(0, min(xs) - horizontal_padding),
                max(0, min(ys) - vertical_padding),
                min(img.width, max(xs) + horizontal_padding),
                min(img.height, max(ys) + vertical_padding)
            )
            
            # 裁剪
            crop = img.crop(bbox)
            
            # 图像预处理：提高识别准确率
            # 1. 锐化图像（增强边缘，有助于识别小字符）
            crop = crop.filter(ImageFilter.SHARPEN)
            
            # 2. 增强对比度（有助于区分字符和背景）
            enhancer = ImageEnhance.Contrast(crop)
            crop = enhancer.enhance(1.3)  # 增强 30%
            
            # 3. 如果图像太小，进行放大（确保有足够分辨率识别下标）
            min_size = 64  # 最小边长
            w, h = crop.size
            if min(w, h) < min_size:
                scale = min_size / min(w, h)
                new_size = (int(w * scale), int(h * scale))
                crop = crop.resize(new_size, Image.Resampling.LANCZOS)
            
            # 调试：保存裁剪图
            if save_debug_crop:
                import time
                debug_dir = Path("output/debug_crops")
                debug_dir.mkdir(exist_ok=True, parents=True)
                timestamp = int(time.time() * 1000)
                crop.save(debug_dir / f"crop_{timestamp}.png")
                print(f"   [DEBUG] 保存裁剪图: crop_{timestamp}.png (size: {crop.size})")
            
            # 识别公式
            # recognize_formula 仅识别公式，不进行检测
            result = self.p2t.recognize_formula(crop)
            
            if isinstance(result, str):
                return result
            return None
            
        except Exception as e:
            print(f"   Region recognition error: {e}")
            return None

    
    def _convert_position(self, position) -> list[tuple[float, float]]:
        """转换坐标格式（numpy 数组 → Python 列表）"""
        if position is None or len(position) == 0:
            return [(0, 0), (0, 0), (0, 0), (0, 0)]
        
        # 处理 numpy 数组
        if hasattr(position, 'tolist'):
            position = position.tolist()
        
        points = []
        for p in position:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                points.append((float(p[0]), float(p[1])))
            else:
                points.append((0, 0))
        
        while len(points) < 4:
            points.append((0, 0))
        
        return points[:4]
    
    def _estimate_font_size(self, polygon: list[tuple[float, float]]) -> float:
        """估算字号（取短边长度）"""
        if len(polygon) < 4:
            return 12.0
        
        p0, p1, p2, p3 = polygon[:4]
        edge1 = math.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)
        edge2 = math.sqrt((p3[0] - p0[0])**2 + (p3[1] - p0[1])**2)
        
        font_height = min(edge1, edge2)
        return font_height if font_height > 0 else 12.0
