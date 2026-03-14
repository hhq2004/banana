"""
增强框检测模块

作者：zhangjunkai
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import tempfile

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .data_types import ElementInfo, BoundingBox, ProcessingResult
from .base import ProcessingContext


def clahe_enhance(image: np.ndarray, clip_limit: float = 6.0, tile_size: int = 4) -> np.ndarray:
    """
    CLAHE自适应直方图均衡化增强（激进参数）
    
    Args:
        image: BGR格式的图像
        clip_limit: 对比度限制（越高对比度增强越强，默认6.0）
        tile_size: 网格大小（越小越局部化，默认4）
        
    Returns:
        增强后的BGR图像
    """
    # 转换到LAB颜色空间
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 对L通道应用CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_enhanced = clahe.apply(l)
    
    # 合并通道
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return result


def get_edge_enhanced_image(image: np.ndarray) -> np.ndarray:
    """
    边缘增强图像（让边框更清晰）
    
    Args:
        image: BGR格式的图像
        
    Returns:
        边缘增强后的BGR图像
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Canny边缘检测
    edges = cv2.Canny(gray, 50, 150)
    
    # 膨胀边缘让线条更粗
    kernel = np.ones((2, 2), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # 边缘叠加到原图（白色背景+黑色边缘）
    result = image.copy()
    result[edges_dilated > 0] = [0, 0, 0]  # 边缘处变黑
    
    # 增强对比度
    result = clahe_enhance(result, clip_limit=4.0, tile_size=8)
    
    return result


def sharpen_image(image: np.ndarray) -> np.ndarray:
    """
    USM锐化增强
    """
    gaussian = cv2.GaussianBlur(image, (0, 0), 3.0)
    result = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
    return result


def get_high_contrast_image(image: np.ndarray) -> np.ndarray:
    """
    高对比度图像（适度的对比度增强，保留颜色）
    """
    # 在LAB空间增强，保留颜色信息
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 适度CLAHE（不要太极端）
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    # 合并
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def get_morphology_image(image: np.ndarray) -> np.ndarray:
    """
    形态学处理（闭运算让边框更完整）
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 边缘检测
    edges = cv2.Canny(gray, 30, 100)
    
    # 闭运算连接断开的边缘
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 膨胀
    dilated = cv2.dilate(closed, kernel, iterations=1)
    
    # 叠加到白色背景
    result = np.ones_like(image) * 255
    result[dilated > 0] = [0, 0, 0]
    
    return result


def get_binary_edge_image(image: np.ndarray) -> np.ndarray:
    """
    二值化边缘图（纯黑白，边缘更干净）
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 自适应阈值
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    
    # 形态学处理去噪
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def get_laplacian_image(image: np.ndarray) -> np.ndarray:
    """
    拉普拉斯锐化（突出边缘细节）
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 拉普拉斯算子
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    
    # 增强
    laplacian = cv2.equalizeHist(laplacian)
    
    # 叠加到原图
    result = image.copy()
    lap_3ch = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
    result = cv2.addWeighted(result, 0.7, lap_3ch, 0.3, 0)
    
    return result


class EnhancedFrameDetector:
    """
    增强框检测器
    通过图像预处理提升SAM3对浅色/低对比度边框的检测能力
    """
    DEFAULT_CONFIG = {
        'enabled': True,
        'debug': False,                 # 是否启用debug可视化
        'debug_output_dir': None,       # debug输出目录（可视化结果）
        'temp_dir': '/shared_data/img2xml/temp',  # 临时文件目录（SAM3服务需要能访问）
        'enhance_levels': [1.0, 2.0, 5.0, 10.0],  # 颜色增强系数（与frame_det一致）
        'enable_downscale': True,       # 缩小检测（检测大框更容易）
        'downscale_factor': 0.5,        # 缩小比例（0.5 = 长宽减半）
        'enable_subregion': True,       # 是否启用子图检测
        'subregion_area_threshold': 0.12,  # 子图检测的面积阈值（占原图比例）
        'subregion_max_count': 4,       # 子图检测最大数量
        'iou_threshold': 0.5,           # 去重IoU阈值
        'score_threshold': 0.3,         # SAM3置信度阈值
        'min_area': 25,                 # 最小面积（降低以检测小矩形）
        'frame_prompts': [              # 框/矢量元素检测提示词
            # 基础矩形
            "rectangle",
            "rounded rectangle",
            "square",
            # 背景/容器
            "flowchart section",
            "workflow section",
            "title section",
            "border",
            "colored section",
            "dashed border",
            "dashed box",
            "dashed frame"
        ]
    }
    
    def __init__(self, config: dict = None):
        """
        初始化增强框检测器
        
        Args:
            config: 配置字典，会与DEFAULT_CONFIG合并
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self._temp_files = []  # 跟踪临时文件，用于清理
        self._debug_dir = None  # debug输出目录
        self._step_counter = 0  # 步骤计数器
    
    def _setup_debug(self, context: ProcessingContext):
        """设置debug输出目录"""
        if not self.config.get('debug', False):
            return
        
        debug_dir = self.config.get('debug_output_dir')
        if not debug_dir:
            print(f"  [DEBUG] 未设置debug_output_dir，跳过可视化")
            return
        
        os.makedirs(debug_dir, exist_ok=True)
        self._debug_dir = debug_dir
        self._step_counter = 0
        print(f"  [DEBUG] 可视化输出目录: {debug_dir}")
    
    def _save_debug_visualization(self, image_path: str, results: List[Dict], step_name: str):
        """
        保存某一步检测结果的可视化（只保存一张带框的图）
        """
        if not self._debug_dir:
            return
        
        self._step_counter += 1
        
        img = cv2.imread(image_path)
        if img is None:
            return
        
        img_vis = img.copy()
        
        for i, r in enumerate(results):
            bbox = r.get('bbox', [0, 0, 0, 0])
            score = r.get('score', 0)
            x1, y1, x2, y2 = map(int, bbox)
            color = ((i * 67) % 255, (i * 123 + 100) % 255, (i * 211 + 50) % 255)
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_vis, f"{score:.2f}", (x1, max(y1-5, 15)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        vis_path = os.path.join(self._debug_dir, 
            f"{self._step_counter:02d}_{step_name}_{len(results)}.png")
        cv2.imwrite(vis_path, img_vis)
        print(f"    [DEBUG] 已保存: {vis_path}")
    
    def detect(self, context: ProcessingContext, 
               existing_elements: List[ElementInfo]) -> List[ElementInfo]:
        """
        执行增强检测
        
        Args:
            context: 处理上下文，包含image_path
            existing_elements: 已有的元素列表（SAM3检测结果）
            
        Returns:
            新检测到的元素列表（已去重）
        """
        if not self.config.get('enabled', True):
            return []
        
        image_path = context.image_path
        if not os.path.exists(image_path):
            print(f"[EnhancedFrameDetector] 图片不存在: {image_path}")
            return []
        
        print(f"\n[EnhancedFrameDetector] 开始增强检测...")
        print(f"  图片尺寸: {Image.open(image_path).size}")
        
        # 设置debug
        self._setup_debug(context)
        
        # 收集所有检测结果
        all_new_results = []
        
        try:
            # 读取原图尺寸
            pil_img = Image.open(image_path)
            orig_w, orig_h = pil_img.size
            
            # 1. 多轮颜色增强检测
            enhance_levels = self.config.get('enhance_levels', [5.0, 10.0])
            for enhance in enhance_levels:
                level_name = f"增强{enhance}x"
                print(f"  [{level_name}] 检测中...")
                
                enhanced_path = self._create_enhanced_image(image_path, enhance)
                if enhanced_path:
                    results = self._call_sam3(enhanced_path)
                    self._save_debug_visualization(enhanced_path, results, f"enhance_{enhance}x")
                    all_new_results.extend(results)
                    print(f"    检测到 {len(results)} 个框")
            
            # 2. 缩小检测（长宽减半，检测大框更容易）
            if self.config.get('enable_downscale', True):
                downscale_factor = self.config.get('downscale_factor', 0.5)
                print(f"  [缩小{downscale_factor}x] 检测中...")
                
                for enhance in enhance_levels:
                    # 先增强再缩小
                    enhanced_path = self._create_enhanced_image(image_path, enhance)
                    if enhanced_path:
                        downscaled_path = self._create_downscaled_image(enhanced_path, downscale_factor)
                        if downscaled_path:
                            results = self._call_sam3(downscaled_path)
                            
                            # 先用原始坐标可视化（在缩小图上）
                            self._save_debug_visualization(downscaled_path, results, f"downscale_{enhance}x")
                            
                            # 再转换坐标回原图尺寸（转成整数）
                            for r in results:
                                bbox = r['bbox']
                                r['bbox'] = [
                                    int(bbox[0] / downscale_factor),
                                    int(bbox[1] / downscale_factor),
                                    int(bbox[2] / downscale_factor),
                                    int(bbox[3] / downscale_factor)
                                ]
                                r['source'] = f'downscale_{enhance}x'
                            
                            all_new_results.extend(results)
                            print(f"    [{enhance}x缩小] 检测到 {len(results)} 个框")
            
            # 3. 子图检测（使用所有已检测框，包括增强检测到的）
            if self.config.get('enable_subregion', True):
                print(f"  [子图检测] 检测中...")
                subregion_results = self._detect_subregions(image_path, existing_elements, all_new_results)
                all_new_results.extend(subregion_results)
                print(f"    检测到 {len(subregion_results)} 个框")
            
            # 保存去重前的所有结果
            self._save_debug_visualization(image_path, all_new_results, "all_before_dedup")
            
            # 4. 去重合并
            print(f"  [去重合并] 总候选: {len(all_new_results)}")
            new_elements = self._merge_and_deduplicate(
                all_new_results, 
                existing_elements,
                context
            )
            print(f"  [去重后] 新增: {len(new_elements)} 个框")
            
            return new_elements
            
        finally:
            # 清理临时文件
            self._cleanup_temp_files()
    
    def _create_downscaled_image(self, image_path: str, scale: float) -> Optional[str]:
        """
        创建缩小的图片（用于检测大框）
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            h, w = img.shape[:2]
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            img_small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            temp_path = self._get_temp_path(image_path, f'_downscale_{scale}')
            cv2.imwrite(temp_path, img_small)
            self._temp_files.append(temp_path)
            
            return temp_path
            
        except Exception as e:
            print(f"    [警告] 缩小图片失败: {e}")
            return None
    
    def _create_enhanced_image(self, image_path: str, enhance_factor: float) -> Optional[str]:
        """
        创建颜色增强的图片
        
        在HSV空间增强饱和度，让浅色变深色
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # RGB -> HSV
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            h, s, v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
            
            # 增加饱和度
            s_new = np.clip(s * enhance_factor, 0, 255)
            
            # gamma校正让中间和高亮度变暗（浅色变深）
            v_normalized = v / 255.0
            gamma = 1.0 + (enhance_factor - 1.0) * 0.5
            v_new = np.clip((v_normalized ** gamma) * 255, 0, 255)
            
            img_hsv[:, :, 1] = s_new
            img_hsv[:, :, 2] = v_new
            
            # HSV -> BGR
            img_hsv = np.clip(img_hsv, 0, 255).astype(np.uint8)
            img_enhanced = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
            
            # 保存临时文件
            temp_path = self._get_temp_path(image_path, f'_enhanced_{enhance_factor}')
            cv2.imwrite(temp_path, img_enhanced)
            self._temp_files.append(temp_path)
            
            return temp_path
            
        except Exception as e:
            print(f"    [警告] 颜色增强失败: {e}")
            return None
    
    def _create_inverted_image(self, image_path: str) -> Optional[str]:
        """
        创建颜色反转的图片
        
        255 - img，让白底浅边框变深色
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # RGB反转
            img_inverted = 255 - img
            
            # 保存临时文件
            temp_path = self._get_temp_path(image_path, '_inverted')
            cv2.imwrite(temp_path, img_inverted)
            self._temp_files.append(temp_path)
            
            return temp_path
            
        except Exception as e:
            print(f"    [警告] 颜色反转失败: {e}")
            return None
    
    def _create_edge_enhanced_image(self, image_path: str) -> Optional[str]:
        """
        创建边缘增强图像（让边框更清晰）
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            img_edge = get_edge_enhanced_image(img)
            
            temp_path = self._get_temp_path(image_path, '_edge')
            cv2.imwrite(temp_path, img_edge)
            self._temp_files.append(temp_path)
            
            return temp_path
            
        except Exception as e:
            print(f"    [警告] 边缘增强失败: {e}")
            return None
    
    def _create_sharpen_clahe_image(self, image_path: str) -> Optional[str]:
        """
        创建锐化+CLAHE增强的图片（边缘更清晰）
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # 先锐化
            img_sharp = sharpen_image(img)
            # 再CLAHE
            img_sharp_clahe = clahe_enhance(img_sharp)
            
            temp_path = self._get_temp_path(image_path, '_sharp_clahe')
            cv2.imwrite(temp_path, img_sharp_clahe)
            self._temp_files.append(temp_path)
            
            return temp_path
            
        except Exception as e:
            print(f"    [警告] 锐化+CLAHE失败: {e}")
            return None
    
    def _create_high_contrast_image(self, image_path: str) -> Optional[str]:
        """高对比度图像"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            img_hc = get_high_contrast_image(img)
            temp_path = self._get_temp_path(image_path, '_high_contrast')
            cv2.imwrite(temp_path, img_hc)
            self._temp_files.append(temp_path)
            return temp_path
        except Exception as e:
            print(f"    [警告] 高对比度失败: {e}")
            return None
    
    def _create_morphology_image(self, image_path: str) -> Optional[str]:
        """形态学处理图像"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            img_morph = get_morphology_image(img)
            temp_path = self._get_temp_path(image_path, '_morphology')
            cv2.imwrite(temp_path, img_morph)
            self._temp_files.append(temp_path)
            return temp_path
        except Exception as e:
            print(f"    [警告] 形态学处理失败: {e}")
            return None
    
    def _create_binary_edge_image(self, image_path: str) -> Optional[str]:
        """二值化边缘图像"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            img_binary = get_binary_edge_image(img)
            temp_path = self._get_temp_path(image_path, '_binary_edge')
            cv2.imwrite(temp_path, img_binary)
            self._temp_files.append(temp_path)
            return temp_path
        except Exception as e:
            print(f"    [警告] 二值化边缘失败: {e}")
            return None
    
    def _create_laplacian_image(self, image_path: str) -> Optional[str]:
        """拉普拉斯锐化图像"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            img_lap = get_laplacian_image(img)
            temp_path = self._get_temp_path(image_path, '_laplacian')
            cv2.imwrite(temp_path, img_lap)
            self._temp_files.append(temp_path)
            return temp_path
        except Exception as e:
            print(f"    [警告] 拉普拉斯锐化失败: {e}")
            return None
    
    def _detect_subregions(self, image_path: str, 
                           existing_elements: List[ElementInfo],
                           enhanced_results: List[Dict] = None) -> List[Dict]:
        """
        子图检测：对大框区域进行二次检测（包含CLAHE增强）
        
        找出被大框遮挡或在大框内部的小框
        
        Args:
            existing_elements: 原SAM3检测的元素
            enhanced_results: 增强检测到的新框（字典格式）
        """
        results = []
        enhanced_results = enhanced_results or []
        
        try:
            pil_image = Image.open(image_path)
            img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            img_width, img_height = pil_image.size
            total_area = img_width * img_height
            
            area_threshold = self.config.get('subregion_area_threshold', 0.12)
            
            # 收集所有大框（包括原SAM3结果和增强检测结果）
            large_boxes = []  # [(bbox, source)]
            
            # 从 existing_elements 找大框
            for elem in existing_elements:
                bbox = elem.bbox.to_list()
                box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if box_area / total_area > area_threshold:
                    large_boxes.append(bbox)
            
            # 从增强检测结果找大框
            for r in enhanced_results:
                bbox = r['bbox']
                box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if box_area / total_area > area_threshold:
                    # 检查是否与已有大框重复
                    is_dup = False
                    for existing_bbox in large_boxes:
                        if self._calculate_iou(bbox, existing_bbox) > 0.5:
                            is_dup = True
                            break
                    if not is_dup:
                        large_boxes.append(bbox)
            
            if not large_boxes:
                return []
            
            # 去重：如果大框包含小框，去掉小框
            large_boxes = self._filter_contained_boxes(large_boxes)
            
            # 按面积排序（大的优先）
            large_boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
            
            # 限制最大数量
            max_count = self.config.get('subregion_max_count', 4)
            if len(large_boxes) > max_count:
                large_boxes = large_boxes[:max_count]
                print(f"    发现大框超过{max_count}个，只取面积最大的{max_count}个")
            
            print(f"    处理 {len(large_boxes)} 个大框 (面积>{area_threshold*100:.0f}%) [仅CLAHE增强]")
            
            for i, bbox in enumerate(large_boxes):
                x1, y1, x2, y2 = bbox
                box_area = (x2-x1)*(y2-y1)
                
                # 只跑CLAHE增强（不跑原图，节省资源）
                sub_cv = img_cv[y1:y2, x1:x2].copy()
                sub_enhanced = clahe_enhance(sub_cv)
                temp_clahe_path = self._get_temp_path(image_path, f'_subregion_{i}_clahe')
                cv2.imwrite(temp_clahe_path, sub_enhanced)
                self._temp_files.append(temp_clahe_path)
                
                all_sub_results = self._call_sam3(temp_clahe_path)
                print(f"      子图{i+1} (面积{box_area/total_area*100:.1f}%) CLAHE检测到 {len(all_sub_results)} 个框")
                
                # 转换坐标回原图
                for r in all_sub_results:
                    sub_bbox = r['bbox']
                    r['bbox'] = [
                        sub_bbox[0] + x1,
                        sub_bbox[1] + y1,
                        sub_bbox[2] + x1,
                        sub_bbox[3] + y1
                    ]
                    r['source'] = 'subregion'
                    results.append(r)
            
            return results
            
        except Exception as e:
            print(f"    [警告] 子图检测失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _call_sam3(self, image_path: str) -> List[Dict]:
        """
        调用SAM3服务进行检测
        
        Returns:
            检测结果列表，每项包含 bbox, score, prompt, mask, polygon
        """
        try:
            from litserve_model.request import sam3_request
            
            prompts = self.config.get('frame_prompts', self.DEFAULT_CONFIG['frame_prompts'])
            score_threshold = self.config.get('score_threshold', 0.3)
            min_area = self.config.get('min_area', 25)
            
            results = sam3_request.call_sam3_service(
                image_path,
                prompts,
                score_threshold=score_threshold,
                min_area=min_area
            )
            
            return results
            
        except Exception as e:
            print(f"    [警告] SAM3调用失败: {e}")
            return []
    
    def _merge_and_deduplicate(self, new_results: List[Dict],
                               existing_elements: List[ElementInfo],
                               context: ProcessingContext) -> List[ElementInfo]:
        """
        合并去重，转换为ElementInfo
        
        Args:
            new_results: 新检测到的结果（字典格式）
            existing_elements: 已有元素
            context: 处理上下文
            
        Returns:
            去重后的新元素列表
        """
        iou_threshold = self.config.get('iou_threshold', 0.5)
        
        # 收集已有元素的bbox
        existing_bboxes = [elem.bbox.to_list() for elem in existing_elements]
        
        # 第一轮去重：与已有元素去重
        filtered_results = []
        for r in new_results:
            is_duplicate = False
            for exist_bbox in existing_bboxes:
                iou = self._calculate_iou(r['bbox'], exist_bbox)
                if iou > iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered_results.append(r)
        
        # 第二轮去重：新结果内部NMS去重
        filtered_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        final_results = []
        
        for r in filtered_results:
            is_duplicate = False
            for kept in final_results:
                iou = self._calculate_iou(r['bbox'], kept['bbox'])
                if iou > iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                final_results.append(r)
        
        # 转换为ElementInfo
        start_id = max([e.id for e in existing_elements], default=-1) + 1
        new_elements = []
        
        for i, r in enumerate(final_results):
            elem = self._convert_to_element(r, start_id + i)
            new_elements.append(elem)
        
        return new_elements
    
    def _convert_to_element(self, result: Dict, elem_id: int) -> ElementInfo:
        """
        将检测结果转换为ElementInfo
        """
        bbox = BoundingBox.from_list(result['bbox'])
        
        # 根据prompt确定元素类型
        prompt = result.get('prompt', 'rectangle')
        element_type = self._map_prompt_to_type(prompt)
        
        elem = ElementInfo(
            id=elem_id,
            element_type=element_type,
            bbox=bbox,
            score=result.get('score', 0.5),
            polygon=result.get('polygon', []),
            mask=result.get('mask'),
            source_prompt=prompt
        )
        
        # 标记来源
        source = result.get('source', 'enhanced')
        elem.processing_notes.append(f"source_group=enhanced_detection")
        elem.processing_notes.append(f"detection_source={source}")
        elem._source_group = 'background'  # 框类元素放到背景组
        elem._group_priority = 1  # 优先级最低，容易被其他元素覆盖
        
        return elem
    
    def _map_prompt_to_type(self, prompt: str) -> str:
        """
        将prompt映射到元素类型
        """
        prompt_lower = prompt.lower()
        
        if 'dashed' in prompt_lower:
            return 'section_panel'  # 虚线框
        elif 'rounded' in prompt_lower:
            return 'rounded rectangle'
        elif 'container' in prompt_lower:
            return 'container'
        else:
            return 'rectangle'  # 实线框
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """计算IoU"""
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
    
    def _filter_contained_boxes(self, boxes: List[List[int]]) -> List[List[int]]:
        """
        去除被其他大框包含的小框
        
        如果框A被框B包含超过80%，则去掉框A
        """
        if len(boxes) <= 1:
            return boxes
        
        to_remove = set()
        
        for i, box_i in enumerate(boxes):
            if i in to_remove:
                continue
            area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
            
            for j, box_j in enumerate(boxes):
                if i == j or j in to_remove:
                    continue
                area_j = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])
                
                # 计算交集
                x1 = max(box_i[0], box_j[0])
                y1 = max(box_i[1], box_j[1])
                x2 = min(box_i[2], box_j[2])
                y2 = min(box_i[3], box_j[3])
                
                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                    
                    # 如果小框被大框包含超过80%，去掉小框
                    if area_i < area_j:
                        containment = intersection / area_i if area_i > 0 else 0
                        if containment > 0.8:
                            to_remove.add(i)
                            break
                    else:
                        containment = intersection / area_j if area_j > 0 else 0
                        if containment > 0.8:
                            to_remove.add(j)
        
        return [b for i, b in enumerate(boxes) if i not in to_remove]
    
    def _get_temp_path(self, original_path: str, suffix: str) -> str:
        """生成临时文件路径（用于SAM3服务访问）- 返回绝对路径"""
        stem = Path(original_path).stem
        ext = Path(original_path).suffix or '.png'
        
        # 使用配置的 temp_dir（SAM3 服务需要能访问）
        temp_dir = self.config.get('temp_dir', '/shared_data/img2xml/temp')
        temp_dir = os.path.abspath(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        
        return os.path.join(temp_dir, f"{stem}{suffix}{ext}")
    
    def _cleanup_temp_files(self):
        """清理临时文件"""
        for f in self._temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except:
                pass
        self._temp_files = []


# ======================== 快捷函数 ========================
def enhance_frame_detection(context: ProcessingContext,
                            existing_elements: List[ElementInfo],
                            config: dict = None) -> List[ElementInfo]:
    """
    快捷函数 - 增强框检测
    
    Args:
        context: 处理上下文
        existing_elements: 已有元素
        config: 配置（可选）
        
    Returns:
        新检测到的元素列表
    """
    detector = EnhancedFrameDetector(config)
    return detector.detect(context, existing_elements)
