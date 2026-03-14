"""
字体处理器模块

功能：
    1. 从 Azure OCR 结果提取字体名称
    2. 字体名称标准化（如 ArialMT → Arial）
    3. 字体推测（当 OCR 未返回字体时，根据文本内容推测）
    4. 空间聚类统一字体

负责人：[填写负责人姓名]

接口说明：
    输入：text_blocks 列表，每个块包含 text 和可选的 font_family
    输出：text_blocks 列表，每个块填充 font_family 字段

使用示例：
    from processors.font_family import FontFamilyProcessor
    
    processor = FontFamilyProcessor()
    blocks = processor.process(text_blocks, global_font="Arial")
"""

import re
import copy
from typing import List, Dict, Any, Optional

try:
    from .font_detector import FontDetector
    FONT_DETECTOR_AVAILABLE = True
except ImportError:
    FONT_DETECTOR_AVAILABLE = False
    print("警告: FontDetector 不可用，将跳过图像字体检测")


class FontFamilyProcessor:
    """
    字体处理器
    
    处理流程：
    1. extract_from_ocr: 从 OCR 结果提取字体
    2. standardize: 标准化字体名称
    3. infer_from_text: 根据文本内容推测字体
    4. unify_by_clustering: 空间聚类统一字体
    """
    
    # 代码风格关键字
    CODE_KEYWORDS = ["id_", "code_", "0x", "struct", "func_", "var_", "ptr_", 
                     "def ", "class ", "import ", "__", "::", "{}"]
    
    # 字体标准化映射表
    FONT_MAPPING = {
        # 中文无衬线体
        "microsoft yahei": "Microsoft YaHei",
        "微软雅黑": "Microsoft YaHei",
        "simhei": "SimHei",
        "黑体": "SimHei",
        "dengxian": "DengXian",
        "等线": "DengXian",
        
        # 英文无衬线体
        "arial": "Arial Narrow",
        "calibri": "Calibri",
        "verdana": "Verdana",
        "helvetica": "Helvetica",
        "roboto": "Roboto",
        "comic sans": "Comic Sans MS",
        "comic sans ms": "Comic Sans MS",
        
        # 衬线体
        "simsun": "SimSun",
        "宋体": "SimSun",
        "times new roman": "Times New Roman",
        "times": "Times New Roman",
        "georgia": "Georgia",
        "yu mincho": "SimSun",
        "ms mincho": "SimSun",
        
        # 等宽体
        "courier new": "Courier New",
        "courier": "Courier New",
        "consolas": "Courier New",
        "monaco": "Courier New",
        "menlo": "Courier New",
    }
    
    # 字体类别关键词（用于未知字体的归类）
    SERIF_KEYWORDS = ["baskerville", "garamond", "palatino", "didot", "bodoni"]
    SANS_KEYWORDS = ["segoe", "tahoma", "trebuchet", "lucida"]
    MONO_KEYWORDS = ["mono", "consolas", "menlo", "monaco", "courier"]
    
    def __init__(self, default_font: str = "Arial Narrow", enable_image_detection: bool = True):
        """
        初始化字体处理器
        
        Args:
            default_font: 默认字体
            enable_image_detection: 是否启用图像检测（区分 Arial 和 Comic Sans）
        """
        self.default_font = default_font
        self.font_cache = {}
        self.enable_image_detection = enable_image_detection and FONT_DETECTOR_AVAILABLE
        
        # 初始化字体检测器
        if self.enable_image_detection:
            self.font_detector = FontDetector(debug=True)  # 开启调试
            self.detection_stats = {
                "checked": 0,
                "detected_comic_sans": 0,
                "skipped_no_key_letters": 0,
                "skipped_wrong_font": 0,
                "skipped_no_image": 0
            }
            print("  ✓ 字体图像检测器已启用（可区分 Arial 和 Comic Sans）")
        else:
            self.font_detector = None
            self.detection_stats = None
    
    def process(
        self,
        text_blocks: List[Dict[str, Any]],
        global_font: str = None,
        unify: bool = True
    ) -> List[Dict[str, Any]]:
        """
        处理字体（主入口）
        
        Args:
            text_blocks: 文字块列表
            global_font: 全局主字体（从最大文字块识别）
            unify: 是否执行聚类统一
            
        Returns:
            处理后的文字块列表
        """
        global_font = global_font or self.default_font
        result = []
        
        for block in text_blocks:
            block = copy.copy(block)
            
            # 已有字体则标准化
            if block.get("font_family"):
                block["font_family"] = self.standardize(block["font_family"])
            else:
                # 推测字体
                block["font_family"] = self.infer_from_text(
                    block.get("text", ""),
                    is_bold=block.get("is_bold", False),
                    is_latex=block.get("is_latex", False),
                    default_font=global_font
                )
            
            # 图像检测：如果字体是 Arial 相关，尝试用图像检测区分 Comic Sans
            if self.enable_image_detection and self._should_check_with_image(block):
                detected_font = self._detect_font_from_image(block)
                if detected_font:
                    block["font_family"] = detected_font
            
            result.append(block)
        
        # 聚类统一
        if unify and len(result) > 1:
            result = self.unify_by_clustering(result)

        # 保证字号分组内字体一致
        result = self._unify_within_font_size_groups(result, default_font=global_font)
        
        # 打印检测统计
        if self.detection_stats and (self.detection_stats['checked'] > 0 or self.detection_stats['detected_comic_sans'] > 0):
            print(f"     🎯 图像检测: 检查 {self.detection_stats['checked']} 个块, "
                  f"发现 {self.detection_stats['detected_comic_sans']} 个 Comic Sans")
        
        return result
    
    def standardize(self, font_name: str) -> str:
        """
        标准化字体名称
        
        策略：
        1. 精确匹配映射表
        2. 模糊匹配（如 ArialMT → Arial）
        3. 归类未知字体
        """
        if not font_name:
            return self.default_font
        
        # 清理
        original = font_name.strip()
        main_font = original.split(',')[0].strip()
        clean_name = main_font.lower()
        
        # 精确匹配
        if clean_name in self.FONT_MAPPING:
            return self.FONT_MAPPING[clean_name]
        
        # 模糊匹配
        for key, value in self.FONT_MAPPING.items():
            if key in clean_name:
                return value
        
        # 归类未知字体
        if any(kw in clean_name for kw in self.SERIF_KEYWORDS):
            return "Times New Roman"
        if any(kw in clean_name for kw in self.SANS_KEYWORDS):
            return "Arial Narrow"
        if any(kw in clean_name for kw in self.MONO_KEYWORDS):
            return "Courier New"
        
        # 保留原始字体
        return main_font

    def _unify_within_font_size_groups(
        self,
        text_blocks: List[Dict[str, Any]],
        default_font: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        在字号分组内统一字体

        规则：
        - 仅对非公式块统一字体，避免影响公式字体
        - 使用组内出现次数最多的字体
        """
        default_font = default_font or self.default_font
        result = copy.deepcopy(text_blocks)

        groups: Dict[int, List[int]] = {}
        for idx, block in enumerate(result):
            group_id = block.get("font_size_group")
            if group_id is None:
                continue
            groups.setdefault(group_id, []).append(idx)

        unified_groups = 0
        unified_blocks = 0
        
        for group_indices in groups.values():
            if len(group_indices) < 2:
                continue

            font_counts: Dict[str, int] = {}
            for idx in group_indices:
                block = result[idx]
                if block.get("is_latex", False):
                    continue
                font = block.get("font_family") or default_font
                font_counts[font] = font_counts.get(font, 0) + 1

            if not font_counts:
                continue

            dominant_font = max(font_counts.items(), key=lambda item: (item[1], item[0]))[0]

            # 统计需要改变的块数
            changed = 0
            for idx in group_indices:
                if result[idx].get("is_latex", False):
                    continue
                if result[idx].get("font_family") != dominant_font:
                    changed += 1
                result[idx]["font_family"] = dominant_font
            
            if changed > 0:
                unified_groups += 1
                unified_blocks += changed

        if unified_groups > 0:
            print(f"     字号分组内统一字体: {unified_groups} 组，调整了 {unified_blocks} 个块")
        
        return result
    
    def infer_from_text(
        self,
        text: str,
        is_bold: bool = False,
        is_latex: bool = False,
        default_font: str = None
    ) -> str:
        """
        根据文本内容推测字体
        
        推测规则（按优先级）：
        1. LaTeX 公式 → Times New Roman
        2. 中文字符 → SimSun
        3. 代码特征 → Courier New
        4. 其他 → default_font
        """
        default_font = default_font or self.default_font
        
        # 缓存
        cache_key = f"{text}_{is_bold}_{is_latex}"
        if cache_key in self.font_cache:
            return self.font_cache[cache_key]
        
        font = default_font
        
        # 公式
        if is_latex:
            font = "Times New Roman"
        # 中文
        elif re.search(r'[\u4e00-\u9fff]', text):
            font = "SimSun"
        # 代码
        elif self._is_code_text(text):
            font = "Courier New"
        # 学术文本
        elif self._is_academic_text(text):
            font = "Times New Roman"
        
        self.font_cache[cache_key] = font
        return font
    
    def _is_code_text(self, text: str) -> bool:
        """检测是否为代码风格文本"""
        text_lower = text.lower()
        
        # 关键字匹配
        if any(kw in text_lower for kw in self.CODE_KEYWORDS):
            return True
        
        # 变量名风格（含下划线且为单词）
        if '_' in text and len(text.split()) == 1:
            return True
        
        return False
    
    def _is_academic_text(self, text: str) -> bool:
        """检测是否为学术文本"""
        academic_keywords = ['figure', 'table', 'equation', 'result', 
                            'method', 'data', 'analysis']
        
        # 含学术关键词且为完整句子
        if any(kw in text.lower() for kw in academic_keywords) and len(text) > 10:
            return True
        
        # 完整句子（含空格、标点、适当长度）
        if ' ' in text and len(text) > 15 and any(p in text for p in ['.', ',', ';']):
            return True
        
        return False
    
    def _should_check_with_image(self, block: Dict[str, Any]) -> bool:
        """
        判断是否需要使用图像检测字体
        
        条件：
        1. 不是斜体（斜体会严重影响字母形状特征）
        2. 字体是 Arial 相关（可能是 Comic Sans 误识别）
        3. 文本包含关键字母（a, e, y, t）
        4. 有可用的图像数据
        
        Args:
            block: 文字块
            
        Returns:
            是否需要图像检测
        """
        if not self.font_detector:
            return False
        
        # 条件1: 不是斜体（斜体字母形状变形严重，无法准确检测）
        if block.get("is_italic", False):
            if self.detection_stats:
                self.detection_stats["skipped_wrong_font"] += 1
            return False
        
        # 条件2: 字体是 Arial 或未知的无衬线体（包括 Courier，因为 OCR 可能误将 Comic Sans 识别为 Courier）
        font = block.get("font_family", "")
        if not any(keyword in font for keyword in ["Arial", "Narrow", "Sans", "Helvetica", "Verdana", "Courier"]):
            if self.detection_stats:
                self.detection_stats["skipped_wrong_font"] += 1
            return False
        
        # 条件3: 文本包含关键字母
        text = block.get("text", "").lower()
        if not any(char in text for char in ['a', 'e', 'y', 't']):
            if self.detection_stats:
                self.detection_stats["skipped_no_key_letters"] += 1
            return False
        
        # 条件4: 有图像数据
        crop_image = block.get("crop_image")
        if crop_image is None:
            if self.detection_stats:
                self.detection_stats["skipped_no_image"] += 1
            return False
        
        # 通过所有条件
        if self.detection_stats:
            self.detection_stats["checked"] += 1
        return True
    
    def _detect_font_from_image(self, block: Dict[str, Any]) -> Optional[str]:
        """
        使用图像检测字体
        
        Args:
            block: 文字块（包含图像数据）
            
        Returns:
            检测到的字体，None 表示检测失败或不确定
        """
        if not self.font_detector:
            return None
        
        # 获取图像
        image = block.get("crop_image")
        if image is None:
            return None
        
        text = block.get("text", "")
        
        try:
            # 调用检测器
            detected_font = self.font_detector.detect_font_from_text(text, image)
            
            # 只在有明确特征时才返回
            if detected_font == "Comic Sans MS":
                if self.detection_stats:
                    self.detection_stats["detected_comic_sans"] += 1
                return detected_font
            
            # 如果检测结果是 Arial，保持原字体不变
            return None
            
        except Exception as e:
            # 静默失败，不影响主流程
            return None
    
    def unify_by_clustering(
        self,
        text_blocks: List[Dict[str, Any]],
        vertical_threshold: float = 0.5,
        horizontal_threshold: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        通过空间聚类统一字体
        
        算法：
        1. 并查集聚类：将空间相邻的文字块分组
        2. 组内统一：使用多数投票选择字体
        """
        if not text_blocks:
            return text_blocks
        
        n = len(text_blocks)
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # 提取边界框
        boxes = []
        for block in text_blocks:
            polygon = block.get('polygon', [])
            if polygon and len(polygon) >= 4:
                xs = [p[0] for p in polygon]
                ys = [p[1] for p in polygon]
                boxes.append({
                    'x_min': min(xs), 'y_min': min(ys),
                    'x_max': max(xs), 'y_max': max(ys),
                    'width': max(xs) - min(xs),
                    'height': max(ys) - min(ys),
                    'font_family': block.get('font_family', '')
                })
            else:
                geo = block.get('geometry', {})
                boxes.append({
                    'x_min': geo.get('x', 0),
                    'y_min': geo.get('y', 0),
                    'x_max': geo.get('x', 0) + geo.get('width', 100),
                    'y_max': geo.get('y', 0) + geo.get('height', 20),
                    'width': geo.get('width', 100),
                    'height': geo.get('height', 20),
                    'font_family': block.get('font_family', '')
                })
        
        # 聚类
        for i in range(n):
            for j in range(i + 1, n):
                if self._should_merge(boxes[i], boxes[j], vertical_threshold, horizontal_threshold):
                    union(i, j)
        
        # 分组
        groups = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        
        # 多数投票统一
        result = copy.deepcopy(text_blocks)
        
        for group_indices in groups.values():
            if len(group_indices) < 2:
                continue
            
            # 统计字体出现次数
            font_counts = {}
            for idx in group_indices:
                font = result[idx].get('font_family', '')
                if font:
                    font_counts[font] = font_counts.get(font, 0) + 1
            
            if font_counts:
                # 选择出现最多的字体
                winner_font = max(font_counts.items(), key=lambda x: x[1])[0]
                for idx in group_indices:
                    result[idx]['font_family'] = winner_font
        
        return result
    
    def _should_merge(
        self, 
        box1: Dict, 
        box2: Dict,
        vertical_threshold: float,
        horizontal_threshold: float
    ) -> bool:
        """判断两个文字块是否应该合并"""
        # 字体必须相同或其中一个为空
        font1 = box1.get('font_family', '')
        font2 = box2.get('font_family', '')
        if font1 and font2 and font1 != font2:
            return False
        
        avg_height = (box1['height'] + box2['height']) / 2
        avg_width = (box1['width'] + box2['width']) / 2
        
        # 计算间距
        vertical_gap = min(
            abs(box1['y_min'] - box2['y_max']),
            abs(box2['y_min'] - box1['y_max'])
        )
        horizontal_gap = min(
            abs(box1['x_min'] - box2['x_max']),
            abs(box2['x_min'] - box1['x_max'])
        )
        
        # 垂直/水平重叠
        y_overlap = not (box1['y_max'] < box2['y_min'] or box2['y_max'] < box1['y_min'])
        x_overlap = not (box1['x_max'] < box2['x_min'] or box2['x_max'] < box1['x_min'])
        
        # 同一行或同一列
        same_row = (y_overlap or vertical_gap < avg_height * vertical_threshold) and \
                   horizontal_gap < avg_width * horizontal_threshold
        same_col = (x_overlap or horizontal_gap < avg_width * 0.3) and \
                   vertical_gap < avg_height * 1.5
        
        return same_row or same_col


if __name__ == "__main__":
    processor = FontFamilyProcessor()
    
    # 测试标准化
    print("=== 字体标准化测试 ===")
    test_fonts = ["ArialMT", "Times New Roman", "微软雅黑", "Consolas"]
    for font in test_fonts:
        print(f"  {font} → {processor.standardize(font)}")
    
    # 测试推测
    print("\n=== 字体推测测试 ===")
    test_texts = ["Hello World", "你好世界", "def main():", "Figure 1. Results"]
    for text in test_texts:
        print(f"  '{text}' → {processor.infer_from_text(text)}")
