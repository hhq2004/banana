

"""
箭头处理模块

功能：
    - 处理箭头元素，生成 DrawIO 矢量箭头
    - 基于 SAM3 mask 的箭头头部检测（膨胀区域）
    - mask 骨架化 + Douglas-Peucker 简化 + 角度规整
    - 矢量化失败则转图片兜底

处理流程：
    1. 检测箭头头部：分析 mask 宽度分布，找"膨胀"区域（三角头部）
    2. 提取箭头颜色：从 mask 区域取中值颜色
    3. 路径提取：
       a. mask 骨架化得到初始路径
       b. 可选：从路径起点向外延伸（基于颜色追踪）
    4. 路径简化：
       a. Douglas-Peucker 简化（消除骨架抖动）
       b. 直线检测（接近直线则拉直为两点）
       c. 角度规整（规整到 0°/45°/90° 等标准角度）
    5. 曲线类型检测：sharp / rounded / curved
    6. 生成 DrawIO XML（Edge 类型）
    7. 矢量化失败则转图片处理（兜底）
"""

import io
import base64
from typing import List, Optional, Tuple, Dict, Any
import cv2
import numpy as np
from PIL import Image

try:
    from skimage.morphology import skeletonize
    SKELETONIZE_AVAILABLE = True
except ImportError:
    SKELETONIZE_AVAILABLE = False
    print("[ArrowProcessor] Warning: skimage not available, skeletonize disabled")

from .base import BaseProcessor, ProcessingContext
from .data_types import ElementInfo, BoundingBox, ProcessingResult, LayerLevel
from .utils import ArrowAttributeDetector, build_arrow_style
from litserve_model.request import rmbg_request

try:
    from .icon_picture_processor import RMBGModel
    _RMBG_AVAILABLE = True
except Exception:
    _RMBG_AVAILABLE = False
    RMBGModel = None


class ArrowProcessor(BaseProcessor):
    """
    箭头处理模块
    
    支持两种处理模式：
    1. 矢量模式：骨架化提取路径，生成DrawIO Edge
    2. 图片模式：抠图后转base64（兜底方案）
    """
    
    # 配置参数
    PADDING = 15
    
    def __init__(self, config=None):
        super().__init__(config)
        self._arrow_detector = ArrowAttributeDetector()
        self._rmbg_model = None
    
    def _get_rmbg(self):
        """懒加载 RMBG，无 mask 时用于箭头图片抠图。"""
        if self._rmbg_model is not None:
            return self._rmbg_model
        if not _RMBG_AVAILABLE or RMBGModel is None:
            return None
        try:
            return True
        except Exception:
            return None
    
    def process(self, context: ProcessingContext) -> ProcessingResult:
        """处理入口"""
        self._log("开始处理箭头元素")
        
        arrows = [e for e in context.elements 
                  if e.element_type.lower() in {'arrow', 'line', 'connector'}]
        
        if not arrows:
            self._log("没有找到箭头元素")
            return ProcessingResult(
                success=True,
                elements=context.elements,
                canvas_width=context.canvas_width,
                canvas_height=context.canvas_height,
                metadata={'arrows_processed': 0, 'total_arrows': 0}
            )
        
        pil_image = Image.open(context.image_path).convert("RGB")
        full_image_np = np.array(pil_image)
        img_h, img_w = full_image_np.shape[:2]
        
        processed_count = 0
        vector_count = 0
        image_count = 0
        
        for arrow in arrows:
            try:
                result = self._process_arrow(arrow, full_image_np, img_w, img_h)
                if result:
                    processed_count += 1
                if arrow.vector_points:
                    vector_count += 1
                elif arrow.base64:
                    image_count += 1
            except Exception as e:
                arrow.processing_notes.append(f"处理失败: {str(e)}")
                self._log(f"箭头{arrow.id}处理失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 【新增】去重：移除重复的箭头
        removed_count = self._remove_duplicate_arrows(arrows)
        if removed_count > 0:
            self._log(f"去重：移除了 {removed_count} 个重复箭头")
            vector_count = sum(1 for a in arrows if a.vector_points and a.xml_fragment)
        
        self._log(f"处理完成: {processed_count}/{len(arrows)}个箭头 "
                  f"(矢量:{vector_count}, 图片:{image_count})")
        
        return ProcessingResult(
            success=True,
            elements=context.elements,
            canvas_width=context.canvas_width,
            canvas_height=context.canvas_height,
            metadata={
                'arrows_processed': processed_count,
                'total_arrows': len(arrows),
                'vector_arrows': vector_count,
                'image_arrows': image_count
            }
        )
    
    def _process_arrow(self, arrow: ElementInfo, full_image_np: np.ndarray,
                       img_w: int, img_h: int) -> bool:
        """
        处理单个箭头
        
        流程：
        1. 从SAM3 mask检测箭头头部（三角形膨胀区域）
        2. 提取箭头颜色
        3. mask骨架化提取路径 + Douglas-Peucker简化 + 角度规整
        4. 失败则转图片处理（兜底）
        """
        x1, y1, x2, y2 = arrow.bbox.to_list()
        
        # ====== 阶段1：从mask检测箭头头部 ======
        arrow_heads = self._find_arrow_heads_from_mask(arrow, full_image_np)
        
        if not arrow_heads:
            self._log(f"箭头{arrow.id}: 未检测到头部，使用bbox中心")
            arrow_heads = [((x1 + x2) // 2, (y1 + y2) // 2)]
        else:
            self._log(f"箭头{arrow.id}: 检测到 {len(arrow_heads)} 个头部: {arrow_heads}")
        
        # 使用第一个头部
        tip = arrow_heads[0]
        
        # ====== 阶段2：提取箭头颜色 ======
        arrow_color = self._extract_arrow_color(arrow, full_image_np)
        if arrow_color is None:
            self._log(f"箭头{arrow.id}: 无法提取颜色")
            return self._fallback_to_image(arrow, full_image_np, img_w, img_h)
        
        self._log(f"箭头{arrow.id}: 颜色 RGB{tuple(arrow_color.astype(int))}")
        
        # ====== 粗细估计 + 预路径 fallback ======
        stroke_width, bulge_area, mask_area = self._estimate_stroke_width_and_bulge(arrow, full_image_np)
        arrow.arrow_thickness = stroke_width
        self._log(f"箭头{arrow.id}: 粗细估计 stroke_width={stroke_width:.2f}, bulge={bulge_area}, mask={mask_area}")
        
        if self._should_fallback_to_image_pre_path(arrow, stroke_width, bulge_area, mask_area, arrow_heads):
            return self._fallback_to_image(arrow, full_image_np, img_w, img_h)
        
        # ====== 阶段3：路径提取（骨架化 + Douglas-Peucker简化）======
        vector_points = None
        
        if arrow.mask is not None:
            vector_points = self._extract_path_from_mask(arrow, full_image_np, tip)
        
        if vector_points:
            self._log(f"箭头{arrow.id}: 路径提取成功")
        
        if vector_points and len(vector_points) >= 2:
            arrow.vector_points = vector_points
            arrow.arrow_start = tuple(vector_points[0])
            arrow.arrow_end = tuple(vector_points[-1])
            self._log(f"箭头{arrow.id}: 最终路径点数={len(vector_points)}")
            
            # 检测箭头属性
            p_x1 = max(0, x1 - 5)
            p_y1 = max(0, y1 - 5)
            p_x2 = min(img_w, x2 + 5)
            p_y2 = min(img_h, y2 + 5)
            small_crop = full_image_np[p_y1:p_y2, p_x1:p_x2]
            
            # 转换路径点为相对于裁剪区域的坐标
            local_path_points = [
                [pt[0] - p_x1, pt[1] - p_y1] for pt in vector_points
            ]
            
            # 传入路径点，用于曲线类型等属性检测
            arrow_attrs = self._arrow_detector.detect_all_attributes(
                small_crop, path_points=local_path_points, mask=None
            )
            # 矢量化输出为单一路径，箭头仅在终点（tip）显示，起点不画箭头，避免两端都出现三角形
            arrow_attrs['start_arrow'] = 'none'
            arrow_attrs['start_fill'] = False
            arrow_attrs['end_arrow'] = 'classic'
            arrow_attrs['end_fill'] = True
            
            # 检测曲线类型：sharp / rounded / curved
            curve_type = self._detect_curve_type(vector_points)
            arrow_attrs['curve_type'] = curve_type
            self._log(f"箭头{arrow.id}: 曲线类型={curve_type}")
            
            self._generate_vector_xml(arrow, arrow_attrs)
            arrow.processing_notes.append(f"矢量化成功: {len(vector_points)}个路径点")
            return True
        
        # 矢量化失败，转图片处理
        self._log(f"箭头{arrow.id}: 矢量化失败，转为图片处理")
        return self._fallback_to_image(arrow, full_image_np, img_w, img_h)
    
    # ==================== 去重逻辑 ====================
    
    def _remove_duplicate_arrows(self, arrows: List[ElementInfo]) -> int:
        """
        移除重复的箭头
        
        重复判断标准：
        1. 起点和终点都很接近（距离 < 15像素）
        2. 或者其中一个箭头只有头部（路径很短），另一个更完整
        
        保留更完整的箭头（路径更长的）
        
        Returns:
            移除的箭头数量
        """
        removed_count = 0
        
        # 收集所有有效的矢量箭头
        valid_arrows = [a for a in arrows if a.vector_points and len(a.vector_points) >= 2]
        
        # 标记要移除的箭头
        to_remove = set()
        
        for i, arrow1 in enumerate(valid_arrows):
            if arrow1.id in to_remove:
                continue
            
            for j, arrow2 in enumerate(valid_arrows):
                if i >= j or arrow2.id in to_remove:
                    continue
                
                # 检查是否重复
                if self._are_arrows_duplicate(arrow1, arrow2):
                    # 保留路径更长的
                    len1 = self._calculate_path_length(arrow1.vector_points)
                    len2 = self._calculate_path_length(arrow2.vector_points)
                    
                    if len1 >= len2:
                        to_remove.add(arrow2.id)
                        self._log(f"  去重：移除箭头{arrow2.id}（保留{arrow1.id}，长度{len1:.1f} >= {len2:.1f}）")
                    else:
                        to_remove.add(arrow1.id)
                        self._log(f"  去重：移除箭头{arrow1.id}（保留{arrow2.id}，长度{len2:.1f} > {len1:.1f}）")
                        break  # arrow1 已被移除，不再与其他比较
        
        # 清除被移除箭头的 XML
        for arrow in arrows:
            if arrow.id in to_remove:
                arrow.xml_fragment = None
                removed_count += 1
        
        return removed_count
    
    def _are_arrows_duplicate(self, arrow1: ElementInfo, arrow2: ElementInfo) -> bool:
        """
        判断两个箭头是否重复
        
        重复条件（满足任一）：
        1. 起点和终点都很近
        2. 一个箭头的起点/终点在另一个箭头的路径上
        """
        if not arrow1.vector_points or not arrow2.vector_points:
            return False
        
        DIST_THRESHOLD = 15  # 像素
        
        pts1 = arrow1.vector_points
        pts2 = arrow2.vector_points
        
        start1, end1 = pts1[0], pts1[-1]
        start2, end2 = pts2[0], pts2[-1]
        
        # 计算距离
        def dist(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        # 条件1：起点终点都很近
        start_close = dist(start1, start2) < DIST_THRESHOLD
        end_close = dist(end1, end2) < DIST_THRESHOLD
        
        # 正向或反向都算重复
        if (start_close and end_close) or (dist(start1, end2) < DIST_THRESHOLD and dist(end1, start2) < DIST_THRESHOLD):
            return True
        
        # 条件2：短箭头的端点在长箭头附近
        # （一个可能只是箭头头部，另一个是完整箭头）
        len1 = self._calculate_path_length(pts1)
        len2 = self._calculate_path_length(pts2)
        
        SHORT_THRESHOLD = 30  # 短箭头阈值
        
        if len1 < SHORT_THRESHOLD:
            # arrow1 很短，检查它的端点是否在 arrow2 附近
            for pt in [start1, end1]:
                for pt2 in pts2:
                    if dist(pt, pt2) < DIST_THRESHOLD:
                        return True
        
        if len2 < SHORT_THRESHOLD:
            # arrow2 很短，检查它的端点是否在 arrow1 附近
            for pt in [start2, end2]:
                for pt1 in pts1:
                    if dist(pt, pt1) < DIST_THRESHOLD:
                        return True
        
        return False
    
    # ==================== 箭头头部检测 ====================
    
    def _find_arrow_heads_from_mask(self, arrow: ElementInfo, 
                                     full_image: np.ndarray) -> List[Tuple[int, int]]:
        """
        从SAM3的mask中检测箭头头部
        
        原理：箭头头部（三角形）是mask中"膨胀"的部分，
        比线条部分更宽。通过分析mask的宽度分布来定位。
        
        Returns:
            头部位置列表 [(x, y), ...]，可能有多个
        """
        if arrow.mask is None:
            return self._find_heads_by_contour(arrow, full_image)
        
        x1, y1, x2, y2 = arrow.bbox.to_list()
        h_img, w_img = full_image.shape[:2]
        
        # 扩大一点范围
        pad = 10
        x1_p = max(0, x1 - pad)
        y1_p = max(0, y1 - pad)
        x2_p = min(w_img, x2 + pad)
        y2_p = min(h_img, y2 + pad)
        
        # 提取mask区域
        try:
            mask = arrow.mask
            if mask.shape[0] < y2_p or mask.shape[1] < x2_p:
                return self._find_heads_by_contour(arrow, full_image)
            
            mask_roi = mask[y1_p:y2_p, x1_p:x2_p]
            mask_binary = (mask_roi > 127).astype(np.uint8) * 255
        except:
            return self._find_heads_by_contour(arrow, full_image)
        
        if np.count_nonzero(mask_binary) < 10:
            return self._find_heads_by_contour(arrow, full_image)
        
        # 方法：分析mask每行/每列的宽度，找"膨胀"的位置
        arrow_heads = []
        
        # 计算每行的mask宽度
        row_widths = []
        for row_idx in range(mask_binary.shape[0]):
            row = mask_binary[row_idx, :]
            cols = np.where(row > 0)[0]
            if len(cols) > 0:
                width = cols[-1] - cols[0] + 1
                center = (cols[0] + cols[-1]) // 2
                row_widths.append((row_idx, width, center))
        
        # 计算每列的mask高度
        col_heights = []
        for col_idx in range(mask_binary.shape[1]):
            col = mask_binary[:, col_idx]
            rows = np.where(col > 0)[0]
            if len(rows) > 0:
                height = rows[-1] - rows[0] + 1
                center = (rows[0] + rows[-1]) // 2
                col_heights.append((col_idx, height, center))
        
        if not row_widths and not col_heights:
            return self._find_heads_by_contour(arrow, full_image)
        
        # 找膨胀点：宽度/高度明显大于平均值的位置
        if row_widths:
            widths = [w[1] for w in row_widths]
            avg_width = np.mean(widths)
            std_width = np.std(widths) if len(widths) > 1 else 0
            threshold = avg_width + max(std_width * 0.5, 2)
            
            # 找所有膨胀的行
            bulge_rows = [(r, w, c) for r, w, c in row_widths if w > threshold]
            
            if bulge_rows:
                # 聚类相邻的膨胀行
                clusters = self._cluster_bulge_points(bulge_rows, axis='row')
                for cluster in clusters:
                    # 取cluster中心
                    avg_row = int(np.mean([p[0] for p in cluster]))
                    avg_col = int(np.mean([p[2] for p in cluster]))
                    # 转换为全局坐标
                    global_x = x1_p + avg_col
                    global_y = y1_p + avg_row
                    arrow_heads.append((global_x, global_y))
        
        if col_heights and not arrow_heads:
            heights = [h[1] for h in col_heights]
            avg_height = np.mean(heights)
            std_height = np.std(heights) if len(heights) > 1 else 0
            threshold = avg_height + max(std_height * 0.5, 2)
            
            bulge_cols = [(c, h, r) for c, h, r in col_heights if h > threshold]
            
            if bulge_cols:
                clusters = self._cluster_bulge_points(bulge_cols, axis='col')
                for cluster in clusters:
                    avg_col = int(np.mean([p[0] for p in cluster]))
                    avg_row = int(np.mean([p[2] for p in cluster]))
                    global_x = x1_p + avg_col
                    global_y = y1_p + avg_row
                    arrow_heads.append((global_x, global_y))
        
        # 如果还是没找到，用轮廓方法
        if not arrow_heads:
            return self._find_heads_by_contour(arrow, full_image)
        
        return arrow_heads
    
    def _find_real_tip_by_path_direction(self, path: List[List[int]], 
                                          arrow: ElementInfo,
                                        #   rough_tip: Tuple[int, int]) -> Tuple[int, int]:
                                          rough_tip: Tuple[int, int]) -> Tuple[Tuple[int, int], Optional[Tuple[int, int]]]:
        """
        根据路径进入三角形的方向，找到真正的尖端位置和底边中点
        
        原理：
        1. 路径的一端在三角形（膨胀）区域内
        2. 用该端附近几个点的方向作为"进入方向"
        3. 在三角形轮廓顶点中，沿着进入方向最远的点就是尖端
        4. 其他顶点构成底边，计算底边中点（用于斜向时循迹校正）

        Args:
            path: 骨架化得到的路径（全局坐标，方向可能未定）
            arrow: 箭头元素信息
            rough_tip: 粗略的头部位置（膨胀中心）
            
        Returns:
            (真正的尖端位置, 底边中点) - 底边中点可能为None
        """
        if arrow.mask is None or len(path) < 2:
            return rough_tip, None
        
        try:
            x1, y1, x2, y2 = arrow.bbox.to_list()
            mask = arrow.mask
            
            # 1. 找到膨胀区域（三角形区域）
            pad = 10
            x1_p = max(0, x1 - pad)
            y1_p = max(0, y1 - pad)
            x2_p = min(mask.shape[1], x2 + pad)
            y2_p = min(mask.shape[0], y2 + pad)
            
            mask_roi = mask[y1_p:y2_p, x1_p:x2_p]
            mask_binary = (mask_roi > 127).astype(np.uint8) * 255
            
            # 分析膨胀区域：找宽度/高度明显增大的位置
            bulge_mask = self._detect_bulge_region(mask_binary)
            
            # 2. 确定路径哪一端在膨胀区域内
            # 检查 path[0] 和 path[-1] 哪个更靠近膨胀区域
            def point_in_bulge(px, py):
                """检查点是否在膨胀区域内"""
                local_x = px - x1_p
                local_y = py - y1_p
                if 0 <= local_y < bulge_mask.shape[0] and 0 <= local_x < bulge_mask.shape[1]:
                    return bulge_mask[local_y, local_x] > 0
                return False
            
            start_in_bulge = point_in_bulge(path[0][0], path[0][1])
            end_in_bulge = point_in_bulge(path[-1][0], path[-1][1])
            
            # 确定哪一端是头部端：一律用 rough_tip 距离判断，避免「尾端接大形状」时
            # 膨胀主要覆盖尾端导致误判 head_end='start'、real_tip 选到箭尾
            dist_start = (path[0][0] - rough_tip[0])**2 + (path[0][1] - rough_tip[1])**2
            dist_end = (path[-1][0] - rough_tip[0])**2 + (path[-1][1] - rough_tip[1])**2
            if dist_end < dist_start:
                head_end = 'end'
                head_idx = -1
            else:
                head_end = 'start'
                head_idx = 0
            
            # 3. 计算进入方向（从线条部分指向头部的方向）
            if head_end == 'end':
                # 路径从 path[0] 走向 path[-1]，取最后几个点的方向
                if len(path) >= 4:
                    dx = path[-1][0] - path[-4][0]
                    dy = path[-1][1] - path[-4][1]
                else:
                    dx = path[-1][0] - path[0][0]
                    dy = path[-1][1] - path[0][1]
                ref_point = path[-1]
            else:
                # 路径从 path[-1] 走向 path[0]，取最前几个点的方向
                if len(path) >= 4:
                    dx = path[0][0] - path[3][0]
                    dy = path[0][1] - path[3][1]
                else:
                    dx = path[0][0] - path[-1][0]
                    dy = path[0][1] - path[-1][1]
                ref_point = path[0]
            
            length = np.sqrt(dx*dx + dy*dy)
            if length < 1:
                return rough_tip, None
            
            dir_x = dx / length
            dir_y = dy / length
            
            # 4. 找三角形轮廓的顶点
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return rough_tip, None
            
            largest_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            if len(approx) < 3:
                return rough_tip, None
            
            vertices = [(x1_p + pt[0][0], y1_p + pt[0][1]) for pt in approx]
            
            # 4.5 只保留膨胀区域（三角形头部）内的顶点作为尖端候选，避免选到箭尾/箭杆角
            vertices_candidates = [v for v in vertices if point_in_bulge(v[0], v[1])]
            if not vertices_candidates:
                vertices_candidates = vertices
                self._log(f"  尖端候选: 膨胀内无顶点，兜底使用全部{len(vertices)}个轮廓顶点")
            else:
                self._log(f"  尖端候选: 仅膨胀内{len(vertices_candidates)}个顶点 (共{len(vertices)}个)")
            
            # 5. 在候选顶点中找尖端
            ref_x, ref_y = ref_point[0], ref_point[1]
            
            # 【改进】优先选择能产生标准方向（水平/垂直）的顶点
            # 因为流程图中大多数箭头是水平或垂直的
            standard_angles = [0, 90, 180, -90, -180, 270]  # 标准角度
            angle_tolerance = 15  # 允许 ±15° 的误差
            
            best_standard_vertex = None
            best_standard_angle_diff = float('inf')
            
            best_projection_vertex = None
            max_projection = -float('inf')
            
            for vx, vy in vertices_candidates:
                rel_x = vx - ref_x
                rel_y = vy - ref_y
                
                # 计算投影（用于备选）
                projection = rel_x * dir_x + rel_y * dir_y
                if projection > max_projection:
                    max_projection = projection
                    best_projection_vertex = (vx, vy)
                
                # 只考虑在进入方向前方的顶点（投影 > 0）
                if projection <= 0:
                    continue
                
                # 计算这个顶点作为尖端时的方向角度
                dist = np.sqrt(rel_x**2 + rel_y**2)
                if dist < 1:
                    continue
                
                angle_rad = np.arctan2(rel_y, rel_x)
                angle_deg = np.degrees(angle_rad)
                
                # 检查是否接近标准角度
                for std_angle in standard_angles:
                    diff = abs(angle_deg - std_angle)
                    # 处理角度环绕（如 179° 和 -179° 的差应该是 2°）
                    if diff > 180:
                        diff = 360 - diff
                    
                    if diff < angle_tolerance and diff < best_standard_angle_diff:
                        best_standard_angle_diff = diff
                        best_standard_vertex = (vx, vy)
            
            # 优先使用标准方向的顶点，否则用投影最远的
            if best_standard_vertex is not None:
                best_vertex = best_standard_vertex
                self._log(f"  尖端优化: 选择标准方向顶点{best_vertex} (角度偏差{best_standard_angle_diff:.1f}°)")
            elif best_projection_vertex is not None:
                best_vertex = best_projection_vertex
                self._log(f"  尖端优化: 无标准方向，用投影最远顶点{best_vertex}")
            else:
                return rough_tip, None

            # 6. 计算底边中点（非尖端顶点的中点，从同一候选集取）
            base_midpoint = None
            if len(vertices_candidates) >= 3:
                base_vertices = [v for v in vertices_candidates if v != best_vertex]
                if len(base_vertices) >= 2:
                    base_vertices_sorted = sorted(base_vertices,
                        key=lambda v: (v[0] - best_vertex[0])**2 + (v[1] - best_vertex[1])**2)
                    v1, v2 = base_vertices_sorted[0], base_vertices_sorted[1]
                    base_midpoint = (int((v1[0] + v2[0]) / 2), int((v1[1] + v2[1]) / 2))
                    self._log(f"  底边中点: {base_midpoint}")
            
            self._log(f"  尖端优化: 粗略{rough_tip} -> 真实{best_vertex} (头部在{head_end}端)")
            return best_vertex, base_midpoint
            
        except Exception as e:
            self._log(f"  尖端优化失败: {e}")
            return rough_tip, None
    
    def _detect_bulge_region(self, mask_binary: np.ndarray) -> np.ndarray:
        """
        检测mask中的膨胀区域（三角形头部所在的区域）
        
        原理：三角形头部比线条部分更宽/更高
        
        Returns:
            膨胀区域的二值mask
        """
        h, w = mask_binary.shape
        bulge_mask = np.zeros_like(mask_binary)
        
        # 计算每行的宽度
        row_widths = []
        for row_idx in range(h):
            row = mask_binary[row_idx, :]
            cols = np.where(row > 0)[0]
            if len(cols) > 0:
                width = cols[-1] - cols[0] + 1
                row_widths.append((row_idx, width))
        
        # 计算每列的高度
        col_heights = []
        for col_idx in range(w):
            col = mask_binary[:, col_idx]
            rows = np.where(col > 0)[0]
            if len(rows) > 0:
                height = rows[-1] - rows[0] + 1
                col_heights.append((col_idx, height))
        
        # 找膨胀的行（宽度 > 平均值 + 0.5*标准差）
        if row_widths:
            widths = [w for _, w in row_widths]
            avg_w = np.mean(widths)
            std_w = np.std(widths) if len(widths) > 1 else 0
            threshold = avg_w + max(std_w * 0.5, 2)
            
            for row_idx, width in row_widths:
                if width > threshold:
                    bulge_mask[row_idx, :] = mask_binary[row_idx, :]
        
        # 找膨胀的列（高度 > 平均值 + 0.5*标准差）
        if col_heights:
            heights = [h for _, h in col_heights]
            avg_h = np.mean(heights)
            std_h = np.std(heights) if len(heights) > 1 else 0
            threshold = avg_h + max(std_h * 0.5, 2)
            
            for col_idx, height in col_heights:
                if height > threshold:
                    bulge_mask[:, col_idx] = np.maximum(bulge_mask[:, col_idx], mask_binary[:, col_idx])
        
        return bulge_mask
    
    def _cluster_bulge_points(self, points: List[Tuple], axis: str, 
                               gap_threshold: int = 5) -> List[List[Tuple]]:
        """将相邻的膨胀点聚类"""
        if not points:
            return []
        
        # 按主轴排序
        sorted_points = sorted(points, key=lambda p: p[0])
        
        clusters = []
        current_cluster = [sorted_points[0]]
        
        for i in range(1, len(sorted_points)):
            prev_pos = sorted_points[i-1][0]
            curr_pos = sorted_points[i][0]
            
            if curr_pos - prev_pos <= gap_threshold:
                current_cluster.append(sorted_points[i])
            else:
                if len(current_cluster) >= 2:  # 至少2个点才算有效
                    clusters.append(current_cluster)
                current_cluster = [sorted_points[i]]
        
        if len(current_cluster) >= 2:
            clusters.append(current_cluster)
        
        # 如果没有有效cluster，返回最大的单点
        if not clusters and points:
            max_point = max(points, key=lambda p: p[1])
            clusters = [[max_point]]
        
        return clusters
    
    def _find_heads_by_contour(self, arrow: ElementInfo, 
                                full_image: np.ndarray) -> List[Tuple[int, int]]:
        """
        通过轮廓分析找箭头头部（备用方法）
        
        分析bbox区域内的轮廓，找最尖锐的角点
        """
        x1, y1, x2, y2 = arrow.bbox.to_list()
        h_img, w_img = full_image.shape[:2]
        
        pad = 5
        x1_p = max(0, x1 - pad)
        y1_p = max(0, y1 - pad)
        x2_p = min(w_img, x2 + pad)
        y2_p = min(h_img, y2 + pad)
        
        roi = full_image[y1_p:y2_p, x1_p:x2_p]
        if roi.size == 0:
            return []
        
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        
        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # 找最大轮廓
        max_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(max_contour) < 10:
            return []
        
        # 多边形近似
        peri = cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, 0.02 * peri, True)
        
        # 找最尖锐的角点（角度最小）
        points = approx.reshape(-1, 2)
        
        if len(points) < 3:
            # 用质心
            M = cv2.moments(max_contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return [(x1_p + cx, y1_p + cy)]
            return []
        
        # 计算每个顶点的角度
        min_angle = float('inf')
        sharpest_point = None
        
        for i in range(len(points)):
            p_prev = points[(i - 1) % len(points)]
            p_curr = points[i]
            p_next = points[(i + 1) % len(points)]
            
            v1 = p_prev - p_curr
            v2 = p_next - p_curr
            
            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)
            
            if len1 > 0 and len2 > 0:
                cos_angle = np.dot(v1, v2) / (len1 * len2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                
                if angle < min_angle:
                    min_angle = angle
                    sharpest_point = p_curr
        
        if sharpest_point is not None:
            return [(x1_p + sharpest_point[0], y1_p + sharpest_point[1])]
        
        return []
    
    # ==================== 颜色提取 ====================
    
    def _extract_arrow_color(self, arrow: ElementInfo, 
                              full_image: np.ndarray) -> Optional[np.ndarray]:
        """从mask或bbox区域提取箭头颜色（亮度分位数区间法）
        
        改进：排除最亮的50%（背景）和最暗的10%（噪点/边框），
        在剩下的40%中取中位数，得到更准确的箭头颜色
        """
        x1, y1, x2, y2 = arrow.bbox.to_list()
        h_img, w_img = full_image.shape[:2]
        
        pad = 5
        x1_p = max(0, x1 - pad)
        y1_p = max(0, y1 - pad)
        x2_p = min(w_img, x2 + pad)
        y2_p = min(h_img, y2 + pad)
        
        roi = full_image[y1_p:y2_p, x1_p:x2_p]
        
        # 如果有mask，用亮度分位数区间法提取颜色
        if arrow.mask is not None:
            try:
                mask = arrow.mask
                if mask.shape[0] >= y2_p and mask.shape[1] >= x2_p:
                    mask_roi = mask[y1_p:y2_p, x1_p:x2_p]
                    mask_binary = mask_roi > 127
                    if np.count_nonzero(mask_binary) > 0:
                        masked_pixels = roi[mask_binary]
                        
                        # 计算亮度并排序
                        luminance = np.dot(masked_pixels.astype(float), [0.299, 0.587, 0.114])
                        sorted_indices = np.argsort(luminance)
                        
                        n = len(sorted_indices)
                        # 排除最暗的2%（极端噪点）和最亮的20%（明显背景）
                        # 在亮度2%-80%之间取中位数（很宽松的区间）
                        start_idx = n // 10      # 排除最暗的5%
                        end_idx = n * 3 // 5     # 排除最亮的20%
                        
                        if end_idx > start_idx:
                            selected_indices = sorted_indices[start_idx:end_idx]
                            selected_pixels = masked_pixels[selected_indices]
                            return np.median(selected_pixels, axis=0).astype(np.uint8)
                        else:
                            # 像素太少，直接取最暗的那个
                            return masked_pixels[sorted_indices[0]].astype(np.uint8)
            except:
                pass
        
        # 没有mask，用最暗的颜色（假设箭头是深色）
        if roi.size > 0:
            pixels = roi.reshape(-1, 3)
            luminance = np.dot(pixels, [0.299, 0.587, 0.114])
            dark_idx = np.argmin(luminance)
            return pixels[dark_idx].astype(np.uint8)
        
        return None
    
    # ==================== 粗细与膨胀估计 ====================
    
    def _estimate_stroke_width_and_bulge(self, arrow: ElementInfo,
                                         full_image: np.ndarray
                                         ) -> Tuple[float, int, int]:
        """
        估计箭头粗细（箭杆）、膨胀区域面积、mask 总面积。
        用于循迹缩放与「非标准箭头 → 图片」规则。
        
        粗细与规则一律基于 mask；无 mask 或 mask 不可用时仅返回默认 stroke_width，
        由预路径规则视为无可用 mask 并 fallback。
        
        Returns:
            (stroke_width, bulge_area, mask_area)
        """
        x1, y1, x2, y2 = arrow.bbox.to_list()
        h_img, w_img = full_image.shape[:2]
        pad = 10
        x1_p = max(0, x1 - pad)
        y1_p = max(0, y1 - pad)
        x2_p = min(w_img, x2 + pad)
        y2_p = min(h_img, y2 + pad)
        stroke_width = 5.0
        bulge_area = 0
        mask_area = 0
        
        if arrow.mask is None:
            return (stroke_width, bulge_area, mask_area)
        
        try:
            mask = arrow.mask
            if mask.shape[0] < y2_p or mask.shape[1] < x2_p:
                return (stroke_width, bulge_area, mask_area)
            mask_roi = mask[y1_p:y2_p, x1_p:x2_p]
            mask_binary = (mask_roi > 127).astype(np.uint8)
        except Exception:
            return (stroke_width, bulge_area, mask_area)
        
        mask_area = int(np.count_nonzero(mask_binary))
        if mask_area < 10:
            return (stroke_width, bulge_area, mask_area)
        
        bulge_mask = self._detect_bulge_region(mask_binary)
        bulge_area = int(np.count_nonzero(bulge_mask))
        
        row_widths = []
        for row_idx in range(mask_binary.shape[0]):
            row = mask_binary[row_idx, :]
            cols = np.where(row > 0)[0]
            if len(cols) > 0:
                row_widths.append(cols[-1] - cols[0] + 1)
        
        col_heights = []
        for col_idx in range(mask_binary.shape[1]):
            col = mask_binary[:, col_idx]
            rows = np.where(col > 0)[0]
            if len(rows) > 0:
                col_heights.append(rows[-1] - rows[0] + 1)
        
        non_bulge = []
        if row_widths:
            avg = np.mean(row_widths)
            std = np.std(row_widths) if len(row_widths) > 1 else 0
            th = avg + max(std * 0.5, 2)
            non_bulge.extend([w for w in row_widths if w <= th])
        if col_heights:
            avg = np.mean(col_heights)
            std = np.std(col_heights) if len(col_heights) > 1 else 0
            th = avg + max(std * 0.5, 2)
            non_bulge.extend([h for h in col_heights if h <= th])
        
        if non_bulge:
            stroke_width = float(np.median(non_bulge))
            stroke_width = max(1.0, min(stroke_width, 50.0))
        
        return (stroke_width, bulge_area, mask_area)
    
    def _should_fallback_to_image_pre_path(self, arrow: ElementInfo,
                                           stroke_width: float,
                                           bulge_area: int, mask_area: int,
                                           arrow_heads: List[Tuple[int, int]]) -> bool:
        """
        预路径判断尽量暴力简化：太粗 → 抠图，其余 → 尝试矢量。
        仅保留：无 mask → 图片；过粗（绝对值 stroke_width > 35）→ 图片。
        """
        if mask_area <= 0:
            self._log(f"箭头{arrow.id}: 无可用 mask → 图片")
            return True
        if stroke_width > 15:
            self._log(f"箭头{arrow.id}: 过粗 stroke_width={stroke_width:.1f} > 15 → 图片")
            return True
        return False
    
    # ==================== 从Mask提取路径（骨架化 + 向外延伸）====================
    
    def _extract_path_from_mask(self, arrow: ElementInfo, 
                                 full_image: np.ndarray,
                                 tip: Tuple[int, int]) -> Optional[List[List[int]]]:
        """
        从SAM3的mask提取路径 - 骨架化 + 向外延伸
        
        流程：
        1. 对mask骨架化，得到mask内的路径
        2. 从路径的起点（非tip端）向外延伸，追踪同色像素
        3. 合并得到完整路径
        """
        if arrow.mask is None:
            return None
        
        try:
            x1, y1, x2, y2 = arrow.bbox.to_list()
            h_img, w_img = full_image.shape[:2]
            
            # 提取mask区域（带padding）
            pad = 10
            x1_p = max(0, x1 - pad)
            y1_p = max(0, y1 - pad)
            x2_p = min(w_img, x2 + pad)
            y2_p = min(h_img, y2 + pad)
            
            mask = arrow.mask
            if mask.shape[0] < y2_p or mask.shape[1] < x2_p:
                return None
            
            mask_roi = mask[y1_p:y2_p, x1_p:x2_p]
            mask_binary = (mask_roi > 127).astype(np.uint8)
            
            mask_pixel_count = np.count_nonzero(mask_binary)
            self._log(f"  mask_roi范围: [{y1_p}:{y2_p}, {x1_p}:{x2_p}], 形状={mask_roi.shape}, mask像素={mask_pixel_count}")
            
            if mask_pixel_count < 10:
                self._log(f"  mask像素太少({mask_pixel_count}<10)，跳过")
                return None
            
            # 多连通块 → 图片
            # num_labels, _ = cv2.connectedComponents(mask_binary)
            # if num_labels - 1 > 1:
            #     self._log(f"  多连通块 num_labels={num_labels} → 图片")
            #     return None
            
            # 骨架化
            if not SKELETONIZE_AVAILABLE:
                return self._fallback_extract_from_mask(arrow, full_image, tip)
            
            skeleton = skeletonize(mask_binary > 0)
            skel_points = np.argwhere(skeleton)
            
            self._log(f"  骨架化后点数: {len(skel_points)}")
            
            if len(skel_points) < 2:
                self._log(f"  骨架点太少({len(skel_points)}<2)，跳过")
                return None
            
            # 【新方法】直接使用骨架上距离最远的两点，并提取连接它们的路径
            # 这样可以避免箭头头部三角形产生的分叉干扰
            
            # 找骨架上距离最远的两个点
            extreme_endpoints = self._find_true_extreme_points(skel_points)
            self._log(f"  骨架极值端点: {extreme_endpoints}")
            
            if len(extreme_endpoints) < 2:
                return None
            
            # 转换为全局坐标
            global_endpoints = [
                [x1_p + ep[1], y1_p + ep[0]] for ep in extreme_endpoints
            ]
            
            # 尝试提取骨架上从一端到另一端的有序路径
            ordered_path = self._extract_ordered_skeleton_path(skeleton, extreme_endpoints[0], extreme_endpoints[1])
            self._log(f"  有序路径点数: {len(ordered_path) if ordered_path else 0}")
            
            if ordered_path and len(ordered_path) >= 2:
                global_path = [[x1_p + p[1], y1_p + p[0]] for p in ordered_path]
            else:
                global_path = global_endpoints
                self._log(f"  路径提取失败，使用端点")
            
            # 用路径+mask细化粗细：mask_area / 路径长度
            path_len = self._calculate_path_length(global_path)
            if path_len > 0 and mask_pixel_count > 0:
                refined = float(mask_pixel_count) / path_len
                refined = max(1.0, min(refined, 50.0))
                arrow.arrow_thickness = refined
                self._log(f"  细化粗细: stroke_width={refined:.2f} (mask_area/path_len)")
            
            sw = max(1.0, arrow.arrow_thickness or 1.0)
            
            # ====== 关键改进：利用路径方向找到真正的尖端位置和底边中点 ======
            real_tip, base_midpoint = self._find_real_tip_by_path_direction(global_path, arrow, tip)
            
            # 用更准确的尖端位置来确定路径方向
            global_path = self._orient_to_tip_simple(global_path, real_tip)
            
            # ====== 关键：从起点向外延伸（传入 stroke_width）======
            arrow_color = self._extract_arrow_color(arrow, full_image)
            if arrow_color is not None:
                original_len = len(global_path)
                max_ext = 800
                extended_path = self._extend_path_beyond_mask(
                    global_path, full_image, arrow_color, mask,
                    max_extension=max_ext,
                    base_midpoint=base_midpoint,
                    stroke_width=sw
                )
                if extended_path and len(extended_path) > original_len:
                    extension_count = len(extended_path) - original_len
                    self._log(f"箭头{arrow.id}: 路径延伸了 {extension_count} 个点")
                    global_path = extended_path
            
            # 骨架过短（延伸后仍很短）→ 图片
            final_len = self._calculate_path_length(global_path)
            if final_len < max(15.0, 2.0 * sw):
                self._log(f"  骨架过短 path_len={final_len:.1f} < max(15,2*sw) → 图片")
                return None
            
            # ====== 路径简化流程（epsilon 与粗细成比例）======
            eps = max(3.0, 0.7 * sw)
            if len(global_path) > 3:
                simplified = self._douglas_peucker_simplify(global_path, epsilon=eps)
                self._log(f"  Douglas-Peucker简化: {len(global_path)} -> {len(simplified)} 个点 (epsilon={eps:.1f})")
                global_path = simplified
            
            if len(global_path) > 2:
                straightened = self._straighten_if_line(global_path, threshold=5.0)
                if len(straightened) < len(global_path):
                    self._log(f"  直线拉直: {len(global_path)} -> {len(straightened)} 个点")
                    global_path = straightened
            
            if len(global_path) >= 2:
                global_path = self._snap_segment_angles(global_path, angle_tolerance=8.0)
            
            # 曲线过于复杂（多尖折 + 最小角小）→ 图片
            if self._is_curve_too_complex(global_path):
                self._log(f"  曲线过于复杂 → 图片")
                return None
            
            # 去除近重复点，减轻拐弯/直线上的细小来回
            global_path = self._remove_near_duplicate_points(global_path, min_dist=1.0)
            return global_path if len(global_path) >= 2 else None
            
        except Exception as e:
            self._log(f"从mask提取路径失败: {e}")
            return None
    
    def _extend_path_beyond_mask(self, path: List[List[int]], 
                                  full_image: np.ndarray,
                                  arrow_color: np.ndarray,
                                  mask: np.ndarray,
                                  max_extension: int = 800,
                                  base_midpoint: Optional[Tuple[int, int]] = None,
                                  stroke_width: float = 1.0) -> List[List[int]]:
        """
        从mask内路径的箭尾端向外延伸，补充SAM3未框选的部分
        
        约定：path[0]=尖端，path[-1]=箭尾。延伸从 path[-1]（箭尾）向外；
        当初始方向接近45度（斜向）时，改用三角形底边中点作为循迹起点。循迹步长/扇形半径等按 stroke_width 缩放。
        
        Args:
            path: mask内的路径，path[0]是尖端，path[-1]是箭尾
            full_image: 完整图像
            arrow_color: 箭头颜色
            mask: SAM3的mask
            max_extension: 最大延伸距离
            base_midpoint: 三角形底边中点，斜向时用作循迹起点
            stroke_width: 箭头粗细，用于缩放扇形/拐弯探索等
        Returns:
            延伸后的完整路径
        """
        if len(path) < 2:
            return path
        
        t = max(1.0, stroke_width)
        h_img, w_img = full_image.shape[:2]
        
        # 备用颜色阈值
        luminance = np.dot(arrow_color, [0.299, 0.587, 0.114])
        if luminance < 50:
            threshold = 70
        elif luminance < 100:
            threshold = 60
        else:
            threshold = 50
        
        # 默认：箭尾 path[-1] 作延伸起点，方向从尖端指向箭尾（向外）
        start_default = tuple(path[-1])
        dx_default = path[-1][0] - path[0][0]
        dy_default = path[-1][1] - path[0][1]
        length_default = np.sqrt(dx_default*dx_default + dy_default*dy_default)
        if length_default > 0:
            direction_default = (dx_default / length_default, dy_default / length_default)
        else:
            direction_default = None
        
        if direction_default is None:
            return path
        
        # 检查是否接近45度（斜向）
        angle_deg = np.degrees(np.arctan2(dy_default, dx_default))
        diagonal_angles = [45, 135, -45, -135]
        is_diagonal = any(abs(angle_deg - a) < 15 for a in diagonal_angles)
        
        # 斜向且底边中点可用时，用几何校正（从底边中点指向箭尾，避免延伸往尖端走回头路）
        if is_diagonal and base_midpoint is not None:
            start = base_midpoint
            dx = path[-1][0] - base_midpoint[0]
            dy = path[-1][1] - base_midpoint[1]
            length = np.sqrt(dx*dx + dy*dy)
            if length > 0:
                direction = (dx / length, dy / length)
                self._log(f"  检测到斜向方向({angle_deg:.1f}°)，使用底边中点校正，起点={start}")
            else:
                start = start_default
                direction = direction_default
        else:
            start = start_default
            direction = direction_default
            self._log(f"  循迹起点: 箭尾端 {start}，方向角度 {angle_deg:.1f}°")
       
        # 计算 mask 内路径的长度
        path_length = 0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            path_length += np.sqrt(dx*dx + dy*dy)
        
        # 判断是否需要延伸（短路径阈值按粗细缩放）
        short_thresh = max(30, 3 * t)
        if path_length < short_thresh:
            should_extend = True
            self._log(f"  路径长度={path_length:.1f}px（<{short_thresh:.0f}），直接尝试延伸")
        else:
            should_extend = self._should_extend_from_boundary(
                start, direction, full_image, arrow_color, threshold, mask, w_img, h_img, stroke_width=t
            )
            if should_extend:
                self._log(f"  路径长度={path_length:.1f}px，正前方有同色，需要延伸")
            else:
                self._log(f"  路径长度={path_length:.1f}px，正前方无箭头像素，不延伸")
        
        if not should_extend:
            return path
        
        # 斜向且用底边中点作起点时，延伸会形成「尾→绿→红」回头，直接保留骨架不延伸
        if is_diagonal and base_midpoint is not None and start == base_midpoint:
            self._log(f"  斜向使用底边中点作起点，为避免回头不延伸，保留骨架路径")
            return path
        
        # ====== 【改进】扇形检测 + 只走上下左右 ======
        
        # 把方向量化为四个主方向
        main_dir = self._get_main_direction(direction)
        self._log(f"  初始主方向: {main_dir}")
        
        # 从起点向外追踪
        extended_points = []
        current = start
        visited = set()
        visited.add(current)
        max_dist_from_start = 0
        idx_at_max = 0
        detected_loop = False
        
        loop_thresh = max(30, 2 * t)
        fan_radius = max(1, round(0.3 * t))
        for step in range(max_extension):
            # 防止绕圈（阈值按粗细缩放）
            dist_from_start = np.sqrt((current[0] - start[0])**2 + (current[1] - start[1])**2)
            if dist_from_start > max_dist_from_start:
                max_dist_from_start = dist_from_start
                idx_at_max = len(extended_points) - 1
            elif max_dist_from_start > loop_thresh and dist_from_start < max_dist_from_start * 0.5:
                self._log(f"  检测到绕圈，停止")
                detected_loop = True
                break
            
            x, y = current
            
            # 检查前方扇形区域是否有同色（半径按粗细缩放）
            fan_points = self._get_forward_fan(x, y, main_dir, radius=fan_radius)
            fan_has_same_color = self._check_fan_same_color(
                fan_points, full_image, arrow_color, threshold, w_img, h_img
            )
            
            if fan_has_same_color:
                # 前方有同色，沿主方向走一步
                next_x = x + main_dir[0]
                next_y = y + main_dir[1]
                
                if not (0 <= next_x < w_img and 0 <= next_y < h_img):
                    self._log(f"  到达图像边界，停止")
                    break
                if (next_x, next_y) in visited:
                    self._log(f"  到达已访问点，停止")
                    break
                next_point = (next_x, next_y)
            else:
                new_dir = self._detect_turn(x, y, main_dir, full_image, arrow_color, 
                                            threshold, w_img, h_img, visited, stroke_width=t)
                
                if new_dir is not None:
                    # 可以拐弯
                    main_dir = new_dir
                    next_x = x + main_dir[0]
                    next_y = y + main_dir[1]
                    
                    if not (0 <= next_x < w_img and 0 <= next_y < h_img):
                        break
                    if (next_x, next_y) in visited:
                        break
                    
                    next_point = (next_x, next_y)
                    self._log(f"  拐弯到方向 {main_dir}")
                else:
                    # 无路可走，停止
                    self._log(f"  前方和侧方都没有同色，停止")
                    break
            
            extended_points.append(list(next_point))
            visited.add(next_point)
            current = next_point
        
        # 合并路径：path 为 [尖端,...,箭尾]，延伸接在箭尾后，即 path + extended_points
        if extended_points:
            if detected_loop and idx_at_max >= 0:
                extended_points = extended_points[: idx_at_max + 1]
                self._log(f"  绕圈截断，保留延伸 {len(extended_points)} 个点（至最远）")
            else:
                self._log(f"  延伸了 {len(extended_points)} 个点")
            return path + extended_points
        
        return path
    
    def _get_main_direction(self, direction: Tuple[float, float]) -> Tuple[int, int]:
        """
        把浮点方向量化为四个主方向之一（上/下/左/右）
        
        Returns:
            (dx, dy) 其中 dx, dy ∈ {-1, 0, 1}，且只有一个非零
        """
        dx, dy = direction
        if abs(dx) > abs(dy):
            # 水平方向为主
            return (1, 0) if dx > 0 else (-1, 0)
        else:
            # 垂直方向为主
            return (0, 1) if dy > 0 else (0, -1)
    
    def _get_forward_fan(self, x: int, y: int, main_dir: Tuple[int, int],
                         radius: int = 1) -> List[Tuple[int, int]]:
        """
        获取前方扇形区域的点：左前、正前、右前。
        radius：半宽（偏移像素），按粗细缩放，至少 1。
        """
        r = max(1, int(radius))
        mdx, mdy = main_dir
        
        if mdx != 0:  # 水平方向（左或右）
            return [
                (x + mdx, y - r),  # 前上
                (x + mdx, y),      # 正前方
                (x + mdx, y + r),  # 前下
            ]
        else:  # 垂直方向（上或下）
            return [
                (x - r, y + mdy),  # 前左
                (x, y + mdy),      # 正前方
                (x + r, y + mdy),  # 前右
            ]
    
    def _check_fan_same_color(self, fan_points: List[Tuple[int, int]], 
                               full_image: np.ndarray,
                               arrow_color: np.ndarray,
                               threshold: float,
                               w_img: int, h_img: int) -> bool:
        """
        检查扇形区域是否有同色点
        """
        relaxed_threshold = threshold * 2.0
        for px, py in fan_points:
            if 0 <= px < w_img and 0 <= py < h_img:
                pixel = full_image[py, px]
                dist = np.linalg.norm(pixel.astype(float) - arrow_color.astype(float))
                if dist < relaxed_threshold:
                    return True
        return False
    
    def _detect_turn(self, x: int, y: int, main_dir: Tuple[int, int],
                     full_image: np.ndarray, arrow_color: np.ndarray,
                     threshold: float, w_img: int, h_img: int,
                     visited: set, stroke_width: float = 1.0) -> Optional[Tuple[int, int]]:
        """
        检测是否需要拐弯（多层探索 + 岔路检测）。探索步长、层数按粗细缩放。
        """
        t = max(1.0, stroke_width)
        mdx, mdy = main_dir
        relaxed_threshold = threshold * 2.0
        explore_depth = 5
        step = max(1, round(0.3 * t))
        min_consecutive = max(3, min(5, 2 + int(0.1 * t)))
        
        if mdx != 0:
            side_dirs = [(0, -1), (0, 1)]
        else:
            side_dirs = [(-1, 0), (1, 0)]
        
        valid_directions = []
        
        for side_dir in side_dirs:
            sdx, sdy = side_dir
            perp_x, perp_y = -sdy, sdx
            consecutive_layers = 0
            
            for depth in range(1, explore_depth + 1):
                layer_has_same_color = False
                depth_step = depth * step
                for offset in [-1, 0, 1]:
                    off = offset * step
                    probe_x = int(round(x + sdx * depth_step + perp_x * off))
                    probe_y = int(round(y + sdy * depth_step + perp_y * off))
                    if not (0 <= probe_x < w_img and 0 <= probe_y < h_img):
                        continue
                    if depth == 1 and offset == 0 and (probe_x, probe_y) in visited:
                        continue
                    probe_pixel = full_image[probe_y, probe_x]
                    probe_dist = np.linalg.norm(probe_pixel.astype(float) - arrow_color.astype(float))
                    if probe_dist < relaxed_threshold:
                        layer_has_same_color = True
                        break
                if layer_has_same_color:
                    consecutive_layers += 1
                else:
                    break
            
            if consecutive_layers >= min_consecutive:
                valid_directions.append((side_dir, consecutive_layers))
        
        # 【关键改进】岔路检测
        if len(valid_directions) >= 2:
            # 两边都有通路 = 岔路（可能遇到边框/其他线条），停止
            self._log(f"  检测到岔路（两边都有通路），停止")
            return None
        elif len(valid_directions) == 1:
            # 只有一边有通路 = 真正的拐弯
            return valid_directions[0][0]
        else:
            # 两边都没有通路 = 无路可走
            return None
    
    def _should_extend_from_boundary(self, start_point: Tuple[int, int],
                                      direction: Tuple[float, float],
                                      full_image: np.ndarray,
                                      arrow_color: np.ndarray,
                                      threshold: float,
                                      mask: np.ndarray,
                                      w_img: int, h_img: int,
                                      stroke_width: float = 1.0) -> bool:
        """
        判断是否应该从 mask 边界向外延伸（颜色距离）。垂直扩散按粗细缩放。
        """
        t = max(1.0, stroke_width)
        x, y = start_point
        dx, dy = direction
        perp_x, perp_y = -dy, dx
        extend_threshold = threshold * 2.0
        
        for dist in range(1, 21):
            max_offset = max(1, int(dist * 0.5) + round(0.2 * t))
            for offset in range(-max_offset, max_offset + 1):
                ext_x = int(round(x + dx * dist + perp_x * offset))
                ext_y = int(round(y + dy * dist + perp_y * offset))
                if not (0 <= ext_x < w_img and 0 <= ext_y < h_img):
                    continue
                if mask is not None and 0 <= ext_y < mask.shape[0] and 0 <= ext_x < mask.shape[1]:
                    if mask[ext_y, ext_x] > 127:
                        continue
                pixel_color = full_image[ext_y, ext_x]
                color_dist = np.linalg.norm(pixel_color.astype(float) - arrow_color.astype(float))
                if color_dist < extend_threshold:
                    return True
        return False
    
    def _find_color_neighbors(self, image: np.ndarray, x: int, y: int,
                               arrow_color: np.ndarray, threshold: float,
                               visited: set, mask: np.ndarray,
                               w_img: int, h_img: int) -> List[Tuple[int, int]]:
        """找8邻域中的同色像素"""
        neighbors = []
        for ddx in [-1, 0, 1]:
            for ddy in [-1, 0, 1]:
                if ddx == 0 and ddy == 0:
                    continue
                nx, ny = x + ddx, y + ddy
                
                if not (0 <= nx < w_img and 0 <= ny < h_img):
                    continue
                if (nx, ny) in visited:
                    continue
                
                # 【改进】不再跳过mask内的点，让颜色循迹可以自由追踪
                # 原来的设计是为了避免重复追踪，但这会导致：
                # 如果SAM3的mask覆盖了箭头的一部分线条，循迹会被阻挡
                # 现在改为完全用颜色追踪，不受mask限制
                
                pixel_color = image[ny, nx]
                color_dist = np.linalg.norm(
                    pixel_color.astype(float) - arrow_color.astype(float)
                )
                # 使用放宽的阈值（2.0倍），允许接近颜色的点也通过
                if color_dist < threshold * 2.0:
                    neighbors.append((nx, ny))
        return neighbors
    
    def _choose_best_neighbor(self, neighbors: List[Tuple[int, int]], 
                               x: int, y: int,
                               direction: Tuple[float, float]) -> Tuple[int, int]:
        """选择最接近当前方向的邻居"""
        best = neighbors[0]
        best_cos = -2
        for nx, ny in neighbors:
            ndx, ndy = nx - x, ny - y
            nlen = np.sqrt(ndx*ndx + ndy*ndy)
            if nlen > 0:
                cos = (direction[0] * ndx + direction[1] * ndy) / nlen
                if cos > best_cos:
                    best_cos = cos
                    best = (nx, ny)
        return best
    
    def _find_skeleton_endpoints(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """找骨架的端点（度为1的点，即只有一个邻居的点）"""
        h, w = skeleton.shape
        endpoints = []
        
        skel_points = np.argwhere(skeleton)
        for r, c in skel_points:
            # 数8邻域中的骨架点数
            count = 0
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and skeleton[nr, nc]:
                        count += 1
            
            if count == 1:  # 端点
                endpoints.append((r, c))
        
        return endpoints
    
    def _merge_close_endpoints(self, endpoints: List[Tuple[int, int]], 
                                min_dist: int = 20) -> List[Tuple[int, int]]:
        """
        合并距离太近的端点
        
        箭头头部三角形骨架化后会产生多个很近的端点，
        需要合并成一个代表点
        """
        if len(endpoints) <= 2:
            return endpoints
        
        merged = []
        used = set()
        
        for i, ep1 in enumerate(endpoints):
            if i in used:
                continue
            
            # 找所有和 ep1 距离小于 min_dist 的端点
            cluster = [ep1]
            for j, ep2 in enumerate(endpoints):
                if j != i and j not in used:
                    dist = np.sqrt((ep1[0] - ep2[0])**2 + (ep1[1] - ep2[1])**2)
                    if dist < min_dist:
                        cluster.append(ep2)
                        used.add(j)
            
            # 用簇的中心点代替
            if cluster:
                center_r = int(np.mean([p[0] for p in cluster]))
                center_c = int(np.mean([p[1] for p in cluster]))
                merged.append((center_r, center_c))
            
            used.add(i)
        
        return merged
    
    def _find_extreme_points(self, skel_points: np.ndarray) -> List[Tuple[int, int]]:
        """找骨架的极值点（最远的两个点）- 旧方法"""
        if len(skel_points) < 2:
            return []
        
        # 简化：取y坐标最小和最大的点，或x坐标最小和最大的点
        min_y_idx = np.argmin(skel_points[:, 0])
        max_y_idx = np.argmax(skel_points[:, 0])
        min_x_idx = np.argmin(skel_points[:, 1])
        max_x_idx = np.argmax(skel_points[:, 1])
        
        # 计算y方向和x方向的跨度
        y_span = skel_points[max_y_idx, 0] - skel_points[min_y_idx, 0]
        x_span = skel_points[max_x_idx, 1] - skel_points[min_x_idx, 1]
        
        if y_span >= x_span:
            p1 = tuple(skel_points[min_y_idx])
            p2 = tuple(skel_points[max_y_idx])
        else:
            p1 = tuple(skel_points[min_x_idx])
            p2 = tuple(skel_points[max_x_idx])
        
        return [p1, p2]
    
    def _find_true_extreme_points(self, skel_points: np.ndarray) -> List[Tuple[int, int]]:
        """
        找骨架上真正距离最远的两个点
        
        使用两次 BFS：
        1. 从任意点出发，找最远的点 A
        2. 从 A 出发，找最远的点 B
        A 和 B 就是骨架的两个真正端点
        """
        if len(skel_points) < 2:
            return []
        
        # 构建点集用于快速查找
        point_set = set(map(tuple, skel_points))
        
        # 从第一个点开始 BFS 找最远点
        start = tuple(skel_points[0])
        farthest_from_start = self._bfs_farthest(start, point_set)
        
        if farthest_from_start is None:
            return [start, tuple(skel_points[-1])]
        
        # 从最远点再次 BFS 找另一个端点
        farthest_from_a = self._bfs_farthest(farthest_from_start, point_set)
        
        if farthest_from_a is None:
            return [farthest_from_start, start]
        
        return [farthest_from_start, farthest_from_a]
    
    def _bfs_farthest(self, start: Tuple[int, int], point_set: set) -> Optional[Tuple[int, int]]:
        """BFS 找距离起点最远的点"""
        from collections import deque
        
        visited = {start}
        queue = deque([start])
        farthest = start
        
        while queue:
            current = queue.popleft()
            farthest = current
            
            r, c = current
            # 8邻域
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    neighbor = (r + dr, c + dc)
                    if neighbor in point_set and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
        
        return farthest if farthest != start else None
    
    def _extract_ordered_skeleton_path(self, skeleton: np.ndarray,
                                        start: Tuple[int, int],
                                        end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        提取从 start 到 end 的有序骨架路径
        
        使用 BFS 追踪，同时记录路径
        """
        from collections import deque
        
        h, w = skeleton.shape
        
        # 验证起点和终点在骨架上
        if not (0 <= start[0] < h and 0 <= start[1] < w and skeleton[start[0], start[1]]):
            return None
        if not (0 <= end[0] < h and 0 <= end[1] < w and skeleton[end[0], end[1]]):
            return None
        
        # BFS with path tracking
        visited = {start}
        queue = deque([(start, [start])])
        
        while queue:
            current, path = queue.popleft()
            
            if current == end:
                return path
            
            r, c = current
            # 8邻域
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    neighbor = (nr, nc)
                    if (0 <= nr < h and 0 <= nc < w and 
                        skeleton[nr, nc] and neighbor not in visited):
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
        
        # 如果没找到路径，返回所有访问过的点（按访问顺序）
        # 这种情况可能是骨架不连通
        if len(visited) > 2:
            return list(visited)
        return None
    
    def _trace_skeleton_path(self, skeleton: np.ndarray,
                              endpoints: List[Tuple[int, int]],
                              tip_local: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """沿骨架追踪路径，从离tip最近的端点开始"""
        if len(endpoints) < 2:
            return None
        
        # 选择离tip最近的端点作为起点
        start = min(endpoints, key=lambda ep: 
                    (ep[0] - tip_local[0])**2 + (ep[1] - tip_local[1])**2)
        
        h, w = skeleton.shape
        visited = set()
        path = [start]
        visited.add(start)
        current = start
        
        # BFS追踪
        while True:
            r, c = current
            next_point = None
            
            # 找下一个点
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < h and 0 <= nc < w and 
                        skeleton[nr, nc] and (nr, nc) not in visited):
                        next_point = (nr, nc)
                        break
                if next_point:
                    break
            
            if next_point is None:
                break
            
            path.append(next_point)
            visited.add(next_point)
            current = next_point
        
        return path if len(path) >= 2 else None
    
    def _fallback_extract_from_mask(self, arrow: ElementInfo,
                                     full_image: np.ndarray,
                                     tip: Tuple[int, int]) -> Optional[List[List[int]]]:
        """如果skeletonize不可用，用轮廓方法"""
        if arrow.mask is None:
            return None
        
        x1, y1, x2, y2 = arrow.bbox.to_list()
        
        # 简单方法：用bbox的两端作为路径
        cx = (x1 + x2) // 2
        
        # 约定：path[0]=尖端。判断 tip 在 bbox 哪侧，另一端为箭尾
        if abs(tip[1] - y1) < abs(tip[1] - y2):
            # tip靠近顶部 → 尖端在上，箭尾在下
            return [[tip[0], tip[1]], [cx, y2]]
        else:
            # tip靠近底部 → 尖端在下，箭尾在上
            return [[tip[0], tip[1]], [cx, y1]]
    
    def _orient_to_tip_simple(self, path: List[List[int]], 
                               tip: Tuple[int, int]) -> List[List[int]]:
        """确保路径起点靠近箭头头部（尖端=path[0]，与循迹方向一致）"""
        if len(path) < 2:
            return path
        
        start_dist = (path[0][0] - tip[0])**2 + (path[0][1] - tip[1])**2
        end_dist = (path[-1][0] - tip[0])**2 + (path[-1][1] - tip[1])**2
        
        # 若尖端更靠近 path[-1]，则反转使 path[0]=尖端
        if end_dist < start_dist:
            return list(reversed(path))
        return path
    
    def _calculate_path_length(self, path: List[List[int]]) -> float:
        """计算路径的总长度（像素）"""
        if not path or len(path) < 2:
            return 0.0
        
        total = 0.0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            total += np.sqrt(dx*dx + dy*dy)
        
        return total
    
    def _remove_near_duplicate_points(self, path: List[List[int]], min_dist: float = 1.0) -> List[List[int]]:
        """
        去除距离过近的连续点，减轻拐弯处与直线上的细小来回。始终保留起点与终点。
        """
        if not path or len(path) < 3:
            return path
        out = [list(path[0])]
        for i in range(1, len(path) - 1):
            dx = path[i][0] - out[-1][0]
            dy = path[i][1] - out[-1][1]
            d = np.sqrt(dx * dx + dy * dy)
            if d >= min_dist:
                out.append(list(path[i]))
        end = list(path[-1])
        if (out[-1][0], out[-1][1]) != (end[0], end[1]):
            out.append(end)
        return out if len(out) >= 2 else path
    
    # ==================== 路径简化方法 ====================
    
    def _douglas_peucker_simplify(self, path: List[List[int]], epsilon: float = 2.0) -> List[List[int]]:
        """
        Douglas-Peucker 算法简化路径
        
        消除骨架化产生的抖动，保留路径的主要形状。
        
        Args:
            path: 原始路径点
            epsilon: 允许的最大偏差（像素），越大简化程度越高
        
        Returns:
            简化后的路径点
        """
        if len(path) < 3:
            return path
        
        # 转换为 numpy 数组，cv2.approxPolyDP 需要特定格式
        path_array = np.array(path, dtype=np.float32).reshape(-1, 1, 2)
        
        # Douglas-Peucker 简化
        simplified = cv2.approxPolyDP(path_array, epsilon, closed=False)
        
        # 转回列表格式
        result = [[int(p[0][0]), int(p[0][1])] for p in simplified]
        
        return result if len(result) >= 2 else path
    
    def _straighten_if_line(self, path: List[List[int]], threshold: float = 5.0) -> List[List[int]]:
        """
        如果路径接近直线，强制拉直为两点
        
        使用最小二乘法拟合直线，计算所有点到直线的最大距离。
        如果最大距离小于阈值，认为是直线。
        
        Args:
            path: 路径点
            threshold: 最大允许偏差（像素）
        
        Returns:
            简化后的路径（可能只有2个点）
        """
        if len(path) < 3:
            return path
        
        try:
            points = np.array(path, dtype=np.float32)
            
            # 用最小二乘拟合直线
            line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
            vx = float(line[0][0])
            vy = float(line[1][0])
            x0 = float(line[2][0])
            y0 = float(line[3][0])
            
            # 计算所有点到拟合线的距离
            max_dist = 0.0
            for p in points:
                # 点到直线距离公式
                d = abs(vy * (p[0] - x0) - vx * (p[1] - y0))
                max_dist = max(max_dist, d)
            
            # 如果最大偏差 < 阈值，认为是直线
            if max_dist < threshold:
                return [path[0], path[-1]]
            
            return path
        except Exception:
            return path
    
    def _snap_segment_angles(self, path: List[List[int]], angle_tolerance: float = 8.0) -> List[List[int]]:
        """
        将接近标准角度的线段规整到标准角度
        
        标准角度：0°, 45°, 90°, 135°, 180°（及其负值）
        
        【改进】对最后一段线段使用更大的容差（15°），
        因为最后一段决定箭头头部方向，歪歪扭扭很明显。
        
        Args:
            path: 路径点
            angle_tolerance: 角度容差（度）
        
        Returns:
            角度规整后的路径
        """
        if len(path) < 2:
            return path
        
        import math
        
        # 标准角度
        standard_angles = [0, 45, 90, 135, 180, -45, -90, -135, -180]
        
        result = [path[0]]
        
        for i in range(1, len(path)):
            prev = result[-1]
            curr = path[i]
            
            dx = curr[0] - prev[0]
            dy = curr[1] - prev[1]
            
            if abs(dx) < 0.001 and abs(dy) < 0.001:
                continue  # 跳过重复点
            
            # 计算当前角度
            angle = math.degrees(math.atan2(dy, dx))
            
            # 找最接近的标准角度
            closest_angle = min(standard_angles, key=lambda a: abs(angle - a))
            
            # 【改进】最后一段线段使用更大的容差（15°）
            # 因为最后一段决定箭头头部方向
            is_last_segment = (i == len(path) - 1)
            tolerance = 15.0 if is_last_segment else angle_tolerance
            
            # 如果偏差小于容差，规整到标准角度
            if abs(angle - closest_angle) <= tolerance:
                length = math.sqrt(dx**2 + dy**2)
                new_dx = length * math.cos(math.radians(closest_angle))
                new_dy = length * math.sin(math.radians(closest_angle))
                new_point = [int(prev[0] + new_dx), int(prev[1] + new_dy)]
                result.append(new_point)
            else:
                result.append(curr)
        
        return result
    
    def _is_curve_too_complex(self, path: List[List[int]]) -> bool:
        """
        曲线过于复杂（多尖折 + 最小角小）→ 矢量化易翻车，建议走图片。
        条件：转角数 > 8 且 min_angle < 60°。
        """
        if not path or len(path) < 3:
            return False
        angles = []
        for i in range(1, len(path) - 1):
            p0, p1, p2 = path[i - 1], path[i], path[i + 1]
            v1 = (p0[0] - p1[0], p0[1] - p1[1])
            v2 = (p2[0] - p1[0], p2[1] - p1[1])
            len1 = np.sqrt(v1[0]**2 + v1[1]**2)
            len2 = np.sqrt(v2[0]**2 + v2[1]**2)
            if len1 < 0.001 or len2 < 0.001:
                continue
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            cos_a = max(-1, min(1, dot / (len1 * len2)))
            angles.append(np.degrees(np.arccos(cos_a)))
        if not angles:
            return False
        min_angle = min(angles)
        num_sharp = sum(1 for a in angles if a < 120)
        return num_sharp > 8 and min_angle < 60
    
    def _detect_curve_type(self, path: List[List[int]]) -> str:
        """
        检测曲线类型：sharp / rounded / curved
        
        判断逻辑：
        1. curved（曲线）：路径点多，角度变化平滑，无明显尖角
        2. sharp（尖角）：有明显的尖锐转角（角度 < 120°）
        3. rounded（圆角）：有转角但较平滑，或路径点少的简单折线
        
        DrawIO 样式对应：
        - sharp: rounded=0
        - rounded: rounded=1
        - curved: curved=1
        
        Returns:
            'sharp', 'rounded', 或 'curved'
        """
        if not path or len(path) < 2:
            return 'sharp'
        
        # 只有2个点（直线）
        if len(path) == 2:
            return 'sharp'
        
        # 计算所有转角的角度
        angles = []
        for i in range(1, len(path) - 1):
            p0 = path[i - 1]
            p1 = path[i]
            p2 = path[i + 1]
            
            # 计算两个向量
            v1 = (p0[0] - p1[0], p0[1] - p1[1])
            v2 = (p2[0] - p1[0], p2[1] - p1[1])
            
            # 计算向量长度
            len1 = np.sqrt(v1[0]**2 + v1[1]**2)
            len2 = np.sqrt(v2[0]**2 + v2[1]**2)
            
            if len1 < 0.001 or len2 < 0.001:
                continue
            
            # 计算夹角（点积）
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            cos_angle = dot / (len1 * len2)
            cos_angle = max(-1, min(1, cos_angle))  # 限制范围
            
            # 转为角度（0-180度，180度是直线，0度是180度折返）
            angle_deg = np.degrees(np.arccos(cos_angle))
            angles.append(angle_deg)
        
        if not angles:
            return 'sharp'
        
        # 分析角度分布
        min_angle = min(angles)
        avg_angle = sum(angles) / len(angles)
        num_sharp_turns = sum(1 for a in angles if a < 120)  # 尖锐转角（< 120°）
        num_points = len(path)
        
        # 判断逻辑
        # 1. 如果有很多路径点（> 10）且平均角度大（> 150°），说明是平滑曲线
        if num_points > 10 and avg_angle > 150:
            return 'curved'
        
        # 2. 如果有尖锐转角（角度 < 90°）
        if min_angle < 90:
            return 'sharp'
        
        # 3. 如果路径点多（> 6）且没有特别尖的角（> 100°），是曲线
        if num_points > 6 and min_angle > 100:
            return 'curved'
        
        # 4. 如果有尖锐转角但不是特别尖（90-120°），用圆角
        if num_sharp_turns > 0 and min_angle >= 90:
            return 'rounded'
        
        # 5. 简单的折线（3-6个点），根据角度判断
        if num_points <= 6:
            if min_angle < 120:
                return 'rounded'  # 有转角的折线用圆角
            else:
                return 'sharp'
        
        # 默认：曲线
        return 'curved'
    
    # ==================== 图片兜底 ====================
    
    def _fallback_to_image(self, arrow: ElementInfo, full_image_np: np.ndarray,
                           img_w: int, img_h: int) -> bool:
        """矢量化失败，转为图片处理
        
        改进：
        1. 动态 padding：根据周围是否有其他元素调整
        2. 优先使用 mask 抠图（更精准）
        3. 后处理：连通域分析，只保留最大区域
        """
        x1, y1, x2, y2 = arrow.bbox.to_list()
        
        # 【改进1】动态 padding：根据周围内容调整
        pad = self._calculate_dynamic_padding(arrow, full_image_np, x1, y1, x2, y2, img_w, img_h)
        self._log(f"  动态 padding: {pad}px")
        
        x1_p = max(0, x1 - pad)
        y1_p = max(0, y1 - pad)
        x2_p = min(img_w, x2 + pad)
        y2_p = min(img_h, y2 + pad)
        
        cropped = full_image_np[y1_p:y2_p, x1_p:x2_p]
        ch, cw = y2_p - y1_p, x2_p - x1_p
        
        # 获取 mask：支持全图尺寸切片 或 bbox 尺寸（与 crop 一致）
        mask_crop = None
        if arrow.mask is not None:
            try:
                mh, mw = arrow.mask.shape[:2]
                if (mh, mw) == (ch, cw):
                    mask_crop = arrow.mask
                elif mh >= y2_p and mw >= x2_p:
                    mask_crop = arrow.mask[y1_p:y2_p, x1_p:x2_p]
            except Exception:
                pass
        
        # 【改进2】优先使用 mask 抠图（更精准），RMBG 作为 fallback
        processed = self._process_arrow_image(cropped, mask_crop)
        
        # 【改进3】后处理：连通域分析，只保留最大区域
        processed = self._keep_largest_component(processed)
        
        # 【改进4】边缘优化：移除外圈的浅色背景残留
        processed = self._refine_alpha_edge(processed, cropped)
        
        # 转base64
        arrow.base64 = self._image_to_base64(processed)
        arrow.bbox = BoundingBox(x1_p, y1_p, x2_p, y2_p)
        
        # 简单起终点
        arrow.arrow_start = (x1_p, (y1_p + y2_p) // 2)
        arrow.arrow_end = (x2_p, (y1_p + y2_p) // 2)
        
        # 生成XML
        self._generate_image_xml(arrow)
        arrow.processing_notes.append("转为图片处理")
        
        return True
    
    def _calculate_dynamic_padding(self, arrow: ElementInfo, full_image_np: np.ndarray,
                                    x1: int, y1: int, x2: int, y2: int,
                                    img_w: int, img_h: int) -> int:
        """
        动态计算 padding 值
        
        策略：
        1. 基础 padding = 8（比原来的 15 更保守）
        2. 检测四边是否有其他内容（非背景像素）
        3. 如果某边有内容，减少该方向的扩展
        4. 如果有 mask，根据 mask 边缘到 bbox 边缘的距离调整
        
        Returns:
            动态计算的 padding 值（4-15 像素）
        """
        base_padding = 8  # 基础 padding，比原来的 15 更保守
        min_padding = 4   # 最小 padding，确保抗锯齿
        max_padding = 15  # 最大 padding
        
        # 如果有 mask，检查 mask 边缘是否已经接近 bbox 边缘
        if arrow.mask is not None:
            try:
                mh, mw = arrow.mask.shape[:2]
                if mh >= y2 and mw >= x2:
                    mask_roi = arrow.mask[y1:y2, x1:x2]
                    mask_binary = mask_roi > 127
                    
                    if np.any(mask_binary):
                        # 检查 mask 是否接触 bbox 边缘
                        touches_top = np.any(mask_binary[0, :])
                        touches_bottom = np.any(mask_binary[-1, :])
                        touches_left = np.any(mask_binary[:, 0])
                        touches_right = np.any(mask_binary[:, -1])
                        
                        # 如果 mask 接触边缘，说明箭头可能延伸到 bbox 外，需要更大 padding
                        edge_touch_count = sum([touches_top, touches_bottom, touches_left, touches_right])
                        if edge_touch_count >= 2:
                            # mask 多边接触 bbox，可能需要更大 padding
                            base_padding = 12
                        elif edge_touch_count == 0:
                            # mask 完全在 bbox 内，可以用较小 padding
                            base_padding = 6
            except Exception:
                pass
        
        # 检测 padding 区域是否有其他内容（非白色背景）
        content_detected = False
        check_margin = max_padding
        
        # 检查四边的 padding 区域
        regions_to_check = []
        
        # 上边
        if y1 > check_margin:
            top_region = full_image_np[max(0, y1-check_margin):y1, x1:x2]
            regions_to_check.append(top_region)
        # 下边
        if y2 + check_margin < img_h:
            bottom_region = full_image_np[y2:min(img_h, y2+check_margin), x1:x2]
            regions_to_check.append(bottom_region)
        # 左边
        if x1 > check_margin:
            left_region = full_image_np[y1:y2, max(0, x1-check_margin):x1]
            regions_to_check.append(left_region)
        # 右边
        if x2 + check_margin < img_w:
            right_region = full_image_np[y1:y2, x2:min(img_w, x2+check_margin)]
            regions_to_check.append(right_region)
        
        for region in regions_to_check:
            if region.size > 0:
                # 检测是否有非背景内容（灰度 < 240 的像素）
                if region.ndim == 3:
                    gray = np.mean(region, axis=2)
                else:
                    gray = region
                non_bg_ratio = np.mean(gray < 240)
                if non_bg_ratio > 0.1:  # 超过 10% 的像素是非背景
                    content_detected = True
                    break
        
        # 如果周围有内容，减少 padding 避免引入杂质
        if content_detected:
            final_padding = min_padding
            self._log(f"  检测到周围有内容，使用最小 padding")
        else:
            final_padding = base_padding
        
        return max(min_padding, min(final_padding, max_padding))
    
    def _process_arrow_image(self, cropped_np: np.ndarray, 
                              mask_crop: Optional[np.ndarray]) -> Image.Image:
        """
        箭头图片处理：优先使用 mask 抠图（更精准），RMBG 作为 fallback
        
        改进：
        1. 优先使用 SAM3 mask 抠图（避免 RMBG 误判周围内容为前景）
        2. 只有在 mask 不可用时才使用 RMBG
        3. 减少膨胀次数，避免引入过多背景
        """
        crop_pil = Image.fromarray(cropped_np)
        
        # 【改进】优先使用 mask 抠图（更精准）
        if mask_crop is not None and np.count_nonzero(mask_crop) > 0:
            self._log(f"  使用 mask 抠图（优先）")
            mask_binary = (mask_crop > 127).astype(np.uint8) * 255
            
            # 【优化】减少膨胀次数：从 2 改为 1，避免引入过多背景
            # 只做轻微膨胀保留抗锯齿边缘
            kernel = np.ones((3, 3), np.uint8)
            mask_dilated = cv2.dilate(mask_binary, kernel, iterations=1)
            
            alpha = mask_dilated
            if cropped_np.ndim == 2:
                rgba = np.stack([cropped_np, cropped_np, cropped_np, alpha], axis=-1)
            else:
                rgba = np.concatenate([cropped_np, alpha[:, :, np.newaxis]], axis=-1)
            return Image.fromarray(rgba.astype(np.uint8), mode="RGBA")
        
        # mask 不可用时，尝试 RMBG（fallback）
        rmbg = self._get_rmbg()
        if rmbg is not None:
            try:
                self._log(f"  mask 不可用，使用 RMBG 抠图（fallback）")
                return rmbg_request.call_rmbg_service(crop_pil)
            except Exception as e:
                self._log(f"  RMBG 抠图失败: {e}")
        
        # 都不可用，返回原图（带透明通道）
        self._log(f"  mask 和 RMBG 都不可用，返回原图")
        return crop_pil.convert("RGBA")
    
    def _keep_largest_component(self, image: Image.Image) -> Image.Image:
        """
        后处理：只保留最大的连通区域，移除杂质
        
        原理：
        1. 提取 alpha 通道
        2. 对 alpha 做连通域分析
        3. 只保留面积最大的连通区域
        4. 其他区域的 alpha 设为 0（透明）
        
        Args:
            image: RGBA 图像
            
        Returns:
            清理后的 RGBA 图像
        """
        if image.mode != "RGBA":
            return image
        
        try:
            img_np = np.array(image)
            alpha = img_np[:, :, 3]
            
            # 二值化 alpha 通道
            alpha_binary = (alpha > 127).astype(np.uint8)
            
            # 连通域分析
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                alpha_binary, connectivity=8
            )
            
            if num_labels <= 2:
                # 只有背景和一个前景，无需处理
                return image
            
            # 找最大的连通区域（排除背景，label=0）
            # stats 的第 5 列是面积
            areas = stats[1:, cv2.CC_STAT_AREA]  # 排除背景
            if len(areas) == 0:
                return image
            
            largest_label = np.argmax(areas) + 1  # +1 因为排除了背景
            largest_area = areas[largest_label - 1]
            total_foreground = np.sum(areas)
            
            # 如果最大区域占前景的比例很小，可能是多个同等重要的部分，不做清理
            if largest_area < total_foreground * 0.5:
                self._log(f"  连通域分析：最大区域占比 < 50%，保留所有区域")
                return image
            
            # 创建新的 alpha 通道，只保留最大区域
            new_alpha = np.zeros_like(alpha)
            new_alpha[labels == largest_label] = alpha[labels == largest_label]
            
            # 计算移除的杂质数量
            removed_components = num_labels - 2  # 减去背景和保留的主体
            if removed_components > 0:
                removed_pixels = total_foreground - largest_area
                self._log(f"  连通域清理：移除 {removed_components} 个杂质区域，共 {removed_pixels} 像素")
            
            # 合并回 RGBA
            img_np[:, :, 3] = new_alpha
            return Image.fromarray(img_np, mode="RGBA")
            
        except Exception as e:
            self._log(f"  连通域分析失败: {e}，返回原图")
            return image
    
    def _refine_alpha_edge(self, image: Image.Image, original_crop: np.ndarray) -> Image.Image:
        """
        边缘优化：移除外圈的浅色背景残留
        
        原理：
        1. 检测 alpha 通道的边缘区域（膨胀 - 原始）
        2. 在边缘区域，如果像素颜色接近白色/浅色背景，设为透明
        3. 可选：对整个 alpha 做轻微侵蚀，收缩边界
        
        Args:
            image: RGBA 图像
            original_crop: 原始裁剪的 RGB 图像（用于颜色判断）
            
        Returns:
            边缘优化后的 RGBA 图像
        """
        if image.mode != "RGBA":
            return image
        
        try:
            img_np = np.array(image)
            alpha = img_np[:, :, 3].copy()
            rgb = img_np[:, :, :3]
            
            # ===== 步骤1：移除边缘的浅色像素 =====
            # 二值化 alpha
            alpha_binary = (alpha > 127).astype(np.uint8)
            
            # 检测边缘区域：膨胀后的区域 - 侵蚀后的区域
            kernel = np.ones((3, 3), np.uint8)
            alpha_dilated = cv2.dilate(alpha_binary, kernel, iterations=2)
            alpha_eroded = cv2.erode(alpha_binary, kernel, iterations=1)
            edge_region = (alpha_dilated - alpha_eroded) > 0
            
            # 计算 RGB 亮度
            luminance = np.mean(rgb, axis=2)
            
            # 在边缘区域，如果亮度高（接近白色背景），设为透明
            # 阈值：灰度 > 230 认为是浅色背景
            light_threshold = 230
            light_edge_mask = edge_region & (luminance > light_threshold)
            
            # 统计移除的像素数
            removed_pixels = np.sum(light_edge_mask)
            
            if removed_pixels > 0:
                alpha[light_edge_mask] = 0
                self._log(f"  边缘优化：移除 {removed_pixels} 个浅色边缘像素")
            
            # ===== 步骤2：轻微收缩整体边界（可选）=====
            # 如果边缘还是有残留，可以做轻微侵蚀
            # 这里使用更保守的策略：只在边缘像素较亮时收缩
            
            # 再次检测当前边缘
            alpha_binary_new = (alpha > 127).astype(np.uint8)
            edge_current = cv2.dilate(alpha_binary_new, kernel, iterations=1) - alpha_binary_new
            edge_current = edge_current > 0
            
            # 边缘像素的平均亮度
            if np.any(edge_current):
                edge_luminance = luminance[edge_current]
                avg_edge_luminance = np.mean(edge_luminance)
                
                # 如果边缘平均亮度较高（> 200），说明可能还有背景残留，做轻微侵蚀
                if avg_edge_luminance > 200:
                    alpha_eroded_final = cv2.erode(alpha, kernel, iterations=1)
                    # 只在高亮度边缘区域应用侵蚀
                    high_lum_edge = edge_current & (luminance > 200)
                    alpha[high_lum_edge] = alpha_eroded_final[high_lum_edge]
                    eroded_count = np.sum(high_lum_edge & (alpha_eroded_final < alpha))
                    if eroded_count > 0:
                        self._log(f"  边缘收缩：侵蚀 {eroded_count} 个高亮度边缘像素")
            
            # 合并回 RGBA
            img_np[:, :, 3] = alpha
            return Image.fromarray(img_np, mode="RGBA")
            
        except Exception as e:
            self._log(f"  边缘优化失败: {e}，返回原图")
            return image
    
    # ==================== XML生成 ====================
    
    def _generate_vector_xml(self, arrow: ElementInfo, 
                              arrow_attrs: Dict[str, Any] = None):
        """生成矢量箭头的XML。内部约定 path[0]=尖端、path[-1]=箭尾；drawio 箭头在 target，故按 尾→尖 输出几何。"""
        if not arrow.vector_points or len(arrow.vector_points) < 2:
            return
        
        # 内部：vector_points[0]=尖端，[-1]=箭尾。drawio 在 target 画箭头，故 source=尾、target=尖
        points = arrow.vector_points
        points_tail_to_tip = list(reversed(points))
        start = points_tail_to_tip[0]   # 尾
        end = points_tail_to_tip[-1]    # 尖
        
        # 构建样式
        if arrow_attrs:
            # 使用 **kwargs 方式传参
            style = build_arrow_style(**arrow_attrs)
        else:
            style = "html=1;edgeStyle=orthogonalEdgeStyle;endArrow=classic;rounded=0;strokeWidth=2;strokeColor=#000000;orthogonalLoop=1;jettySize=auto;"
        
        cell_id = arrow.id + 2
        
        # 构建waypoints（尾→尖顺序的中间点）
        waypoints_xml = ""
        if len(points_tail_to_tip) > 2:
            waypoints_xml = '<Array as="points">\n'
            for pt in points_tail_to_tip[1:-1]:
                waypoints_xml += f'              <mxPoint x="{pt[0]}" y="{pt[1]}"/>\n'
            waypoints_xml += '            </Array>'
        
        arrow.xml_fragment = f'''<mxCell id="{cell_id}" parent="1" edge="1" style="{style}">
          <mxGeometry relative="1" as="geometry">
            <mxPoint x="{start[0]}" y="{start[1]}" as="sourcePoint"/>
            <mxPoint x="{end[0]}" y="{end[1]}" as="targetPoint"/>
            {waypoints_xml}
          </mxGeometry>
        </mxCell>'''
        
        arrow.layer_level = LayerLevel.ARROW.value
    
    def _generate_image_xml(self, arrow: ElementInfo, 
                            arrow_attrs: Dict[str, Any] = None):
        """生成图片箭头的XML"""
        if not arrow.base64:
            return
        
        x1, y1, x2, y2 = arrow.bbox.to_list()
        width = x2 - x1
        height = y2 - y1
        
        cell_id = arrow.id + 2
        
        style = f"shape=image;imageAspect=0;aspect=fixed;verticalLabelPosition=bottom;verticalAlign=top;image=data:image/png,{arrow.base64}"
        
        arrow.xml_fragment = f'''<mxCell id="{cell_id}" parent="1" vertex="1" value="" style="{style}">
          <mxGeometry x="{x1}" y="{y1}" width="{width}" height="{height}" as="geometry"/>
        </mxCell>'''
        
        arrow.layer_level = LayerLevel.ARROW.value
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """将PIL图像转为base64字符串"""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
