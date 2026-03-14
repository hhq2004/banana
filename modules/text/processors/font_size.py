"""
字号处理器模块

功能：
    1. 从 OCR 多边形高度计算字号（px → pt）
    2. 空间聚类统一字号（解决同一行字号微小差异问题）

负责人：[填写负责人姓名]

接口说明：
    输入：text_blocks 列表，每个块包含 geometry（含 height）
    输出：text_blocks 列表，每个块增加 font_size 字段

使用示例：
    from processors.font_size import FontSizeProcessor
    
    processor = FontSizeProcessor()
    blocks = processor.process(text_blocks)
"""

import copy
import statistics
from typing import List, Dict, Any


class FontSizeProcessor:
    """
    字号处理器
    
    处理流程：
    1. calculate_font_size: 根据几何高度计算初始字号
    2. unify_by_clustering: 空间聚类统一字号
    """
    
    def __init__(self, formula_ratio: float = 0.6, text_offset: float = 1.0):
        """
        初始化字号处理器
        
        Args:
            formula_ratio: 公式字号 = height * formula_ratio
            text_offset: 普通文字字号 = height - text_offset
        """
        self.formula_ratio = formula_ratio
        self.text_offset = text_offset
    
    def process(
        self, 
        text_blocks: List[Dict[str, Any]],
        unify: bool = True,
        vertical_threshold_ratio: float = 0.5,
        font_diff_threshold: float = 5.0
    ) -> List[Dict[str, Any]]:
        """
        处理字号（主入口）
        
        Args:
            text_blocks: 文字块列表
            unify: 是否执行聚类统一
            vertical_threshold_ratio: 垂直距离阈值比例
            font_diff_threshold: 字号差异阈值
            
        Returns:
            处理后的文字块列表
        """
        # 步骤 1: 计算初始字号
        blocks = self.calculate_font_sizes(text_blocks)
        
        # 步骤 2: 聚类统一
        if unify and len(blocks) > 1:
            blocks = self.unify_by_clustering(
                blocks, 
                vertical_threshold_ratio, 
                font_diff_threshold
            )
        
        return blocks
    
    def calculate_font_sizes(self, text_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        计算每个文字块的字号
        
        公式：
        - 普通文字：font_size = height - 1
        - 公式：font_size = height * 0.6
        """
        result = []
        for block in text_blocks:
            block = copy.copy(block)
            geometry = block.get("geometry", {})
            height = geometry.get("height", 12)
            width = geometry.get("width", 12)
            is_latex = block.get("is_latex", False)
            
            is_vertical = block.get("is_vertical", False)
            rotation = abs(geometry.get("rotation", 0))
            
            # 检查是否是旋转90度的竖排文字
            is_rotated_90 = 75 < rotation < 105 or 255 < rotation < 285
            
            if not is_vertical and not is_latex and height > width * 2.0:
                # 简单起见，如果极度高瘦，大概率是竖排
                is_vertical = True
            
            if is_vertical:
                if is_rotated_90:
                    # 旋转90度的竖排文字：geometry 的 width 是文字行长度，height 才是字符高度
                    # 字号应该用 height 计算
                    font_size = height - 0.5
                else:
                    # 非旋转的竖排：字号 = 宽度 - 偏移
                    font_size = width - 0.5


            elif is_latex:
                font_size = height * self.formula_ratio
            else:
                font_size = height - self.text_offset
            
            block["font_size"] = max(font_size, 6)  # 最小字号 6pt
            result.append(block)
        
        return result
    
    def unify_by_clustering(
        self,
        text_blocks: List[Dict[str, Any]],
        vertical_threshold_ratio: float = 0.5,
        font_diff_threshold: float = 5.0
    ) -> List[Dict[str, Any]]:
        """
        通过空间聚类统一字号
        
        算法：
        1. 并查集聚类：将空间相邻且字号相近的文字块分组
        2. 组内统一：使用中位数作为统一字号
        
        Args:
            text_blocks: 文字块列表
            vertical_threshold_ratio: 垂直距离阈值（相对于行高）
            font_diff_threshold: 字号差异阈值（pt）
        """
        if not text_blocks:
            return text_blocks
        
        n = len(text_blocks)
        
        # 并查集
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # 聚类
        for i in range(n):
            for j in range(i + 1, n):
                if self._should_group(
                    text_blocks[i], text_blocks[j],
                    vertical_threshold_ratio, font_diff_threshold
                ):
                    union(i, j)
        
        # 分组
        groups = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        
        # 统一字号
        result = copy.deepcopy(text_blocks)
        adjusted_count = 0
        
        # 先标记所有块的字号分组ID（供后续字体统一使用）
        for group_id, group_indices in enumerate(groups.values()):
            for idx in group_indices:
                result[idx]["font_size_group"] = group_id
        
        # 然后统一字号
        for group_indices in groups.values():
            if len(group_indices) < 2:
                continue
            
            # 收集组内字号，使用最小值统一
            font_sizes = [result[i].get("font_size", 12) for i in group_indices]
            min_size = min(font_sizes)
            
            # 统一
            for idx in group_indices:
                old_size = result[idx].get("font_size", 12)
                if abs(old_size - min_size) > 0.1:
                    adjusted_count += 1
                result[idx]["font_size"] = round(min_size, 1)
        
        # 打印统计
        multi_groups = [g for g in groups.values() if len(g) > 1]
        if multi_groups:
            print(f"     发现 {len(multi_groups)} 个需要统一的组（包含 {sum(len(g) for g in multi_groups)} 个文字块）")
            if adjusted_count > 0:
                print(f"     已调整 {adjusted_count} 个文字块的字号")
        else:
            print(f"     未发现需要统一的组")
        
        return result
    
    def _should_group(
        self, 
        block_a: Dict, 
        block_b: Dict,
        vertical_threshold_ratio: float,
        font_diff_threshold: float
    ) -> bool:
        """判断两个文字块是否应该分到同一组"""
        geo_a = block_a.get("geometry", {})
        geo_b = block_b.get("geometry", {})
        
        x1, y1 = geo_a.get("x", 0), geo_a.get("y", 0)
        w1, h1 = geo_a.get("width", 0), geo_a.get("height", 0)
        x2, y2 = geo_b.get("x", 0), geo_b.get("y", 0)
        w2, h2 = geo_b.get("width", 0), geo_b.get("height", 0)
        
        font_a = block_a.get("font_size", 12)
        font_b = block_b.get("font_size", 12)
        
        # 条件 1: 垂直距离接近
        bottom_a, bottom_b = y1 + h1, y2 + h2
        gap_a_above_b = y2 - bottom_a
        gap_b_above_a = y1 - bottom_b
        
        if gap_a_above_b < 0 and gap_b_above_a < 0:
            vertical_distance = 0
        else:
            vertical_distance = min(abs(gap_a_above_b), abs(gap_b_above_a))
        
        min_height = min(h1, h2) if min(h1, h2) > 0 else 1
        vertical_close = vertical_distance < min_height * vertical_threshold_ratio
        
        # 条件 2: 水平有重叠
        right_a, left_b = x1 + w1, x2
        right_b, left_a = x2 + w2, x1
        horizontal_overlap = not (right_a < left_b or right_b < left_a)
        
        # 条件 3: 字号接近
        abs_diff = abs(font_a - font_b)
        avg_font = (font_a + font_b) / 2 if (font_a + font_b) > 0 else 1
        rel_diff = abs_diff / avg_font
        font_close = abs_diff < font_diff_threshold or rel_diff < 0.30
        
        return vertical_close and horizontal_overlap and font_close


if __name__ == "__main__":
    # 测试代码
    processor = FontSizeProcessor()
    
    test_blocks = [
        {"geometry": {"x": 100, "y": 100, "width": 200, "height": 25}},
        {"geometry": {"x": 100, "y": 130, "width": 180, "height": 24}},
        {"geometry": {"x": 100, "y": 160, "width": 190, "height": 26}},
    ]
    
    result = processor.process(test_blocks)
    for i, block in enumerate(result):
        print(f"Block {i+1}: font_size = {block['font_size']}pt")
