"""
文字还原器 - 主接口模块

功能：
    将流程图图片中的文字和公式识别并转换为 draw.io XML 格式。

Pipeline 接口：
    from modules.text import TextRestorer
    
    restorer = TextRestorer()
    xml_string = restorer.process("input.png")  # 返回 XML 字符串
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image

# OCR 模块（相对导入）
from .ocr.azure import AzureOCR
from .coord_processor import CoordProcessor
from .xml_generator import MxGraphXMLGenerator

# 四个处理器（相对导入）
from .processors.font_size import FontSizeProcessor
from .processors.font_family import FontFamilyProcessor
from .processors.style import StyleProcessor
from .processors.formula import FormulaProcessor


# 默认配置
DEFAULT_AZURE_ENDPOINT = "http://localhost:5000"


class TextRestorer:
    """
    文字还原器
    
    协调 OCR、各处理器和输出模块，完成文字还原。
    """
    
    def __init__(self, endpoint: str = None, formula_engine: str = 'pix2text', formula_device: str = 'cuda', debug_crops: bool = False):
        """
        初始化文字还原器
        
        Args:
            endpoint: Azure 容器地址（默认使用 localhost:5000）
            formula_engine: 公式识别引擎 ('pix2text', 'none')
                - 'pix2text': 使用 Pix2Text（默认）
                - 'none': 不使用公式识别
            formula_device: 公式识别设备 ('cpu' 或 'cuda')
            debug_crops: 是否保存调试裁剪图（用于诊断公式识别问题）
        """
        self.endpoint = endpoint or DEFAULT_AZURE_ENDPOINT
        self.formula_engine = formula_engine
        self.formula_device = formula_device
        self.debug_crops = debug_crops
        
        # OCR 客户端（延迟初始化）
        self._azure_ocr = None
        self._pix2text_ocr = None
        
        # 处理器
        self.font_size_processor = FontSizeProcessor()
        self.font_family_processor = FontFamilyProcessor()
        self.style_processor = StyleProcessor()
        self.formula_processor = FormulaProcessor()
        
        # 耗时统计
        self.timing = {
            "azure_ocr": 0.0,
            "pix2text_ocr": 0.0,
            "processing": 0.0,
            "total": 0.0
        }
    
    @property
    def azure_ocr(self) -> AzureOCR:
        """延迟初始化 Azure OCR"""
        if self._azure_ocr is None:
            self._azure_ocr = AzureOCR(endpoint=self.endpoint)
        return self._azure_ocr
    
    @property
    def pix2text_ocr(self):
        """延迟初始化 Pix2Text OCR"""
        if self._pix2text_ocr is None:
            from .ocr.pix2text import Pix2TextOCR
            try:
                self._pix2text_ocr = Pix2TextOCR(device=self.formula_device)
            except Exception as e:
                if self.formula_device != 'cpu':
                    print(f"   ⚠️  Pix2Text 使用 {self.formula_device} 初始化失败，回退到 CPU: {e}")
                    self._pix2text_ocr = Pix2TextOCR(device='cpu')
                else:
                    raise
        return self._pix2text_ocr
    
    def process(self, image_path: str) -> str:
        """
        处理图像，返回 XML 字符串（Pipeline 主接口）
        
        Args:
            image_path: 输入图像路径
            
        Returns:
            draw.io 格式的 XML 字符串
        """
        image_path = Path(image_path)
        
        # 获取图像尺寸
        with Image.open(image_path) as img:
            image_width, image_height = img.size
        
        # 处理图像
        text_blocks = self.process_image(str(image_path))
        
        # 生成 XML
        generator = MxGraphXMLGenerator(
            diagram_name=image_path.stem,
            page_width=image_width,
            page_height=image_height
        )
        
        text_cells = []
        for block in text_blocks:
            geo = block["geometry"]
            cell = generator.create_text_cell(
                text=block["text"],
                x=geo["x"],
                y=geo["y"],
                width=max(geo["width"], 20),
                height=max(geo["height"], 10),
                font_size=block.get("font_size", 12),
                is_latex=block.get("is_latex", False),
                rotation=geo.get("rotation", 0),
                is_vertical=block.get("is_vertical"),
                font_weight=block.get("font_weight"),
                font_style=block.get("font_style"),
                font_color=block.get("font_color"),
                font_family=block.get("font_family")
            )
            text_cells.append(cell)
        
        return generator.generate_xml(text_cells)
    
    def process_image(self, image_path: str) -> List[Dict[str, Any]]:
        """
        处理图像，返回文字块列表
        
        Args:
            image_path: 输入图像路径
            
        Returns:
            处理后的文字块列表
        """
        total_start = time.time()
        image_path = Path(image_path)
        
        # 获取图像尺寸
        with Image.open(image_path) as img:
            image_width, image_height = img.size
        
        # Step 1: OCR 识别
        azure_result, formula_result = self._run_ocr(str(image_path))
        
        # Step 2: 公式处理（合并 Azure 和 Pix2Text）
        processing_start = time.time()
        
        if formula_result:
            print("\n🔗 公式处理...")
            merged_blocks = self.formula_processor.merge_ocr_results(azure_result, formula_result)
            text_blocks = self.formula_processor.to_dict_list(merged_blocks)
        else:
            text_blocks = self._azure_to_dict_list(azure_result)
        
        print(f"   {len(text_blocks)} 个文字块")
        
        # Step 3: 坐标转换
        print("\n📐 坐标转换...")
        coord_processor = CoordProcessor(
            source_width=image_width,
            source_height=image_height
        )
        
        for block in text_blocks:
            polygon = block.get("polygon", [])
            if polygon:
                geometry = coord_processor.polygon_to_geometry(polygon)
                block["geometry"] = geometry
            else:
                block["geometry"] = {"x": 0, "y": 0, "width": 100, "height": 20, "rotation": 0}
        self._detect_vertical_text_blocks(text_blocks)
        # Step 4: 字号处理
        print("\n🔧 字号处理...")
        text_blocks = self.font_size_processor.process(text_blocks)
        
        # Step 4.5: 图像裁剪（为字体检测准备）
        print("\n✂️  图像裁剪（字体检测用）...")
        self._crop_text_regions(text_blocks, str(image_path))
        
        # Step 5: 样式处理（必须在字体检测前，用于斜体过滤）
        print("\n🎨 样式处理...")
        azure_styles = getattr(azure_result, "styles", [])
        text_blocks = self.style_processor.process(text_blocks, azure_styles=azure_styles)
        
        # Step 6: 字体处理（在样式处理后，确保is_italic已设置）
        print("\n🎨 字体处理...")
        global_font = self._detect_global_font(azure_result)
        text_blocks = self.font_family_processor.process(text_blocks, global_font=global_font)
        
        self.timing["processing"] = time.time() - processing_start
        self.timing["total"] = time.time() - total_start
        
        return text_blocks
    
    def restore(
        self,
        image_path: str,
        output_path: str = None,
        save_metadata: bool = True,
        save_debug_image: bool = True
    ) -> str:
        """
        完整还原流程：处理图像并生成 draw.io 文件
        
        Args:
            image_path: 输入图像路径
            output_path: 输出文件路径
            save_metadata: 是否保存元数据
            save_debug_image: 是否生成调试图
            
        Returns:
            输出文件路径
        """
        image_path = Path(image_path)
        
        # 设置输出路径
        if output_path is None:
            output_path = image_path.with_suffix(".drawio")
        else:
            output_path = Path(output_path)
        
        # 获取图像尺寸
        with Image.open(image_path) as img:
            image_width, image_height = img.size
        
        print(f"📄 输入: {image_path}")
        print(f"📝 输出: {output_path}")
        print(f"📐 尺寸: {image_width} x {image_height}")
        
        # 处理图像
        text_blocks = self.process_image(str(image_path))
        
        # 生成 XML
        print("\n📄 生成 XML...")
        xml_start = time.time()
        
        generator = MxGraphXMLGenerator(
            diagram_name=image_path.stem,
            page_width=image_width,
            page_height=image_height
        )
        
        text_cells = []
        for block in text_blocks:
            geo = block["geometry"]
            cell = generator.create_text_cell(
                text=block["text"],
                x=geo["x"],
                y=geo["y"],
                width=max(geo["width"], 20),
                height=max(geo["height"], 10),
                font_size=block.get("font_size", 12),
                is_latex=block.get("is_latex", False),
                rotation=geo.get("rotation", 0),
                is_vertical=block.get("is_vertical"),
                font_weight=block.get("font_weight"),
                font_style=block.get("font_style"),
                font_color=block.get("font_color"),
                font_family=block.get("font_family")
            )
            text_cells.append(cell)
        
        generator.save_to_file(text_cells, str(output_path))
        
        xml_time = time.time() - xml_start
        self.timing["total"] += xml_time
        
        # 保存元数据
        if save_metadata:
            self._save_metadata(str(image_path), str(output_path), text_blocks, image_width, image_height)
        
        # 生成调试图
        if save_debug_image:
            debug_path = output_path.parent / "debug.png"
            self._generate_debug_image(str(image_path), str(debug_path))
        
        # 打印统计
        self._print_stats(text_blocks)
        
        return str(output_path)
    
    def _detect_vertical_text_blocks(
        self,
        text_blocks: List[Dict[str, Any]],
        aspect_ratio_threshold: float = 2.2,
        min_group_size: int = 2,
        x_align_ratio: float = 0.6,
        max_x_gap: float = 10.0,
        max_vertical_gap_ratio: float = 1.2
    ) -> None:
        """
        检测竖排文字
        """
        if not text_blocks:
            return

        print("   检测竖排文字...")
        count = 0 
        
        for block in text_blocks:
            # 跳过公式及明显的水平长文本
            if block.get("is_latex"):
                continue
                
            geo = block.get("geometry", {})
            w = max(float(geo.get("width", 0)), 1.0)
            h = max(float(geo.get("height", 0)), 1.0)
            rotation = float(geo.get("rotation", 0.0))
            
            aspect_ratio = h / w
            vertical_by_rotation = abs(abs(rotation) - 90) < 15
            
            # 简单判定：高宽比 >= 2或者 旋转角度接近垂直
            if aspect_ratio >= 2.0 or vertical_by_rotation:
                block["is_vertical"] = True
                count += 1
                
        print(f"   已标记 {count} 个竖排块")
    
    def _crop_text_regions(self, text_blocks: List[Dict[str, Any]], image_path: str) -> None:
        """
        为每个文字块裁剪图像区域（用于字体图像检测）
        
        Args:
            text_blocks: 文字块列表
            image_path: 原始图像路径
        """
        try:
            # 加载完整图像
            full_image = cv2.imread(image_path)
            if full_image is None:
                print(f"   ⚠️  无法加载图像，跳过图像裁剪")
                return
            
            cropped_count = 0
            
            for block in text_blocks:
                # 跳过公式块（不需要字体检测）
                if block.get("is_latex", False):
                    continue
                
                polygon = block.get("polygon", [])
                if not polygon or len(polygon) < 3:
                    continue
                
                try:
                    # 获取边界框
                    xs = [p[0] for p in polygon]
                    ys = [p[1] for p in polygon]
                    x_min = max(0, int(min(xs)))
                    x_max = min(full_image.shape[1], int(max(xs)))
                    y_min = max(0, int(min(ys)))
                    y_max = min(full_image.shape[0], int(max(ys)))
                    
                    # 裁剪图像（添加小边距以获取完整字符）
                    margin = 2
                    x_min = max(0, x_min - margin)
                    x_max = min(full_image.shape[1], x_max + margin)
                    y_min = max(0, y_min - margin)
                    y_max = min(full_image.shape[0], y_max + margin)
                    
                    # 确保有效区域
                    if x_max > x_min and y_max > y_min:
                        crop = full_image[y_min:y_max, x_min:x_max]
                        
                        # 添加到块中
                        block["crop_image"] = crop
                        cropped_count += 1
                        
                except Exception as e:
                    # 单个块裁剪失败不影响全局
                    if self.debug_crops:
                        print(f"   ⚠️  块裁剪失败: {e}")
                    continue
            
            print(f"   已裁剪 {cropped_count} 个文字区域（用于字体检测）")
            
        except Exception as e:
            print(f"   ⚠️  图像裁剪失败: {e}")
    
    def _run_ocr(self, image_path: str):
        """运行 OCR 识别（Azure + Pix2Text）"""
        # Azure OCR - 文字识别
        print("\n📖 Azure OCR...")
        azure_start = time.time()
        azure_result = self.azure_ocr.analyze_image(image_path)
        self.timing["azure_ocr"] = time.time() - azure_start
        
        # 统计字体大小变化的块
        size_var_count = sum(1 for block in azure_result.text_blocks if getattr(block, 'has_size_variation', False))
        print(f"   {len(azure_result.text_blocks)} 个文字块 ({self.timing['azure_ocr']:.2f}s)")
        if size_var_count > 0:
            print(f"   🔍 检测到 {size_var_count} 个块包含字体大小变化（可能有下标/上标）")
        
        # 公式识别
        formula_result = None
        
        if self.formula_engine == 'pix2text':
            # 切换为 Refinement 模式：基于 Azure 结果进行局部重识别
            print("\n🔬公式优化 (Refinement Mode)...")
            refine_start = time.time()
            fixed_count = 0
            
            # 1. 预处理：识别候选组（尝试合并临近的短块以解决公式断裂问题）
            processed_indices = set()
            new_blocks_map = {}
            indices_to_remove = set()
            
            blocks = azure_result.text_blocks
            i = 0
            while i < len(blocks):
                if i in processed_indices:
                    i += 1
                    continue
                
                # 当前块
                curr_block = blocks[i]
                curr_poly = curr_block.polygon
                
                # 检查是否值得 Refine (初步过滤)
                if not self._should_refine_block(curr_block):
                    i += 1
                    continue
                
                # 尝试向后寻找可以合并的块
                group_indices = [i]
                group_polygon = curr_poly
                
                j = i + 1
                while j < len(blocks):
                    next_block = blocks[j]
                    
                    # 距离检查
                    if self._is_spatially_close(group_polygon, next_block.polygon):
                        if self._should_refine_block(next_block):
                            # 检查是否都是混合文本：如果都是，则不合并（避免合并独立的行）
                            curr_text_combined = " ".join([blocks[k].text for k in group_indices])
                            next_text = next_block.text
                            
                            curr_is_mixed = self._split_text_and_formula(curr_text_combined)['has_both']
                            next_is_mixed = self._split_text_and_formula(next_text)['has_both']
                            
                            if curr_is_mixed and next_is_mixed:
                                # 两个都是混合文本，不合并
                                break
                            
                            group_indices.append(j)
                            group_polygon = self._merge_polygons(group_polygon, next_block.polygon)
                            j += 1
                        else:
                            break
                    else:
                        break
                
                # 确定最终的识别区域
                target_polygon = group_polygon
                original_text_combined = " ".join([blocks[k].text for k in group_indices])
                
                # ⭐ 检测是否为"普通文字 + 公式"的混合模式
                text_parts = self._split_text_and_formula(original_text_combined)
                
                if text_parts and text_parts['has_both']:
                    # 混合模式：只对公式部分进行 Refine
                    formula_part = text_parts['formula']
                    text_part = text_parts['text']
                    
                    # 估算公式部分的区域（简化：使用整个区域，后续可优化）
                    # TODO: 根据文字比例精确裁剪 polygon
                    formula_latex = self.pix2text_ocr.recognize_region(
                        image_path, 
                        target_polygon,
                        save_debug_crop=self.debug_crops
                    )
                    
                    if formula_latex and self.formula_processor.is_valid_formula(formula_latex):
                        # ⭐ 智能提取：从 Pix2Text 结果中移除被错误转换的普通文字部分
                        cleaned_formula_latex = self._extract_formula_from_mixed_latex(
                            formula_latex, 
                            text_part, 
                            formula_part
                        )
                        
                        # 检查提取后的公式是否有意义
                        if cleaned_formula_latex and self._is_refinement_meaningful(formula_part, cleaned_formula_latex):
                            cleaned_latex = self.formula_processor.clean_latex(cleaned_formula_latex)
                            
                            # 组合：普通文字 + 公式
                            combined_text = f"{text_part} ${cleaned_latex}$"
                            
                            import copy
                            new_block = copy.deepcopy(curr_block)
                            new_block.text = combined_text
                            new_block.is_latex = False  # 混合内容，不全是 LaTeX
                            new_block.polygon = target_polygon
                            
                            if len(group_indices) > 1:
                                best_color = self._get_dominant_color_by_length(blocks, group_indices)
                                if best_color:
                                    new_block.font_color = best_color
                                print(f"   Refine [Merge {group_indices}, Mixed]: '{original_text_combined}' -> '{combined_text}'")
                                indices_to_remove.update(group_indices)
                                new_blocks_map[i] = new_block
                            else:
                                dominant_color = self._get_dominant_color_from_styles(curr_block, azure_result.styles)
                                if dominant_color:
                                    curr_block.font_color = dominant_color
                                
                                print(f"   Refine [{i}, Mixed]: '{curr_block.text}' -> '{combined_text}'")
                                curr_block.text = combined_text
                                curr_block.is_latex = False
                                fixed_count += 1
                            
                            processed_indices.update(group_indices)
                            i = j
                            continue
                    else:
                        # ⭐ Pix2Text识别失败时的fallback：使用简单规则转换
                        # 简单规则：(Xi) → (X_{i}), (wi) → (w_{i})
                        import re
                        formula_fallback = re.sub(r'\(([A-Za-z])i\)', r'(\1_{i})', formula_part)
                        
                        if formula_fallback != formula_part:
                            # 成功转换
                            combined_text = f"{text_part} ${formula_fallback}$"
                            
                            import copy
                            new_block = copy.deepcopy(curr_block)
                            new_block.text = combined_text
                            new_block.is_latex = False
                            new_block.polygon = target_polygon
                            
                            if len(group_indices) > 1:
                                best_color = self._get_dominant_color_by_length(blocks, group_indices)
                                if best_color:
                                    new_block.font_color = best_color
                                print(f"   Refine [Merge {group_indices}, Mixed, Fallback]: '{original_text_combined}' -> '{combined_text}'")
                                indices_to_remove.update(group_indices)
                                new_blocks_map[i] = new_block
                            else:
                                dominant_color = self._get_dominant_color_from_styles(curr_block, azure_result.styles)
                                if dominant_color:
                                    curr_block.font_color = dominant_color
                                
                                print(f"   Refine [{i}, Mixed, Fallback]: '{curr_block.text}' -> '{combined_text}'")
                                curr_block.text = combined_text
                                curr_block.is_latex = False
                                fixed_count += 1
                            
                            processed_indices.update(group_indices)
                            i = j
                            continue
                
                # 📝 原有的全公式处理逻辑
                # 调用 Pix2Text（带调试选项）
                latex_text = self.pix2text_ocr.recognize_region(
                    image_path, 
                    target_polygon,
                    save_debug_crop=self.debug_crops
                )
                
                if latex_text and self.formula_processor.is_valid_formula(latex_text):
                    original_text_combined = " ".join([blocks[k].text for k in group_indices])
                    
                    if self._is_refinement_meaningful(original_text_combined, latex_text):
                        cleaned_latex = self.formula_processor.clean_latex(latex_text)
                        
                        import copy
                        new_block = copy.deepcopy(curr_block)
                        new_block.text = f"${cleaned_latex}$"
                        new_block.is_latex = True
                        new_block.polygon = target_polygon
                        new_block.font_family = "Latin Modern Math"
                        
                        if len(group_indices) > 1:
                            # 合并多个块时，继承字符数最多的块的颜色
                            best_color = self._get_dominant_color_by_length(blocks, group_indices)
                            if best_color:
                                new_block.font_color = best_color
                            print(f"   Refine [Merge {group_indices}]: '{original_text_combined}' -> '${cleaned_latex}$'")
                            indices_to_remove.update(group_indices)
                            new_blocks_map[i] = new_block
                        else:
                            # 单个块：使用 styles 中按字符数统计的主导颜色
                            dominant_color = self._get_dominant_color_from_styles(curr_block, azure_result.styles)
                            if dominant_color:
                                curr_block.font_color = dominant_color
                            
                            print(f"   Refine [{i}]: '{curr_block.text}' -> '${cleaned_latex}$'")
                            curr_block.text = f"${cleaned_latex}$"
                            curr_block.is_latex = True
                            curr_block.font_family = "Latin Modern Math"
                            fixed_count += 1
                        
                        processed_indices.update(group_indices)
                        i = j
                        continue
                
                i += 1
            
            # 处理合并后的块列表更新
            if indices_to_remove:
                final_blocks = []
                for idx, block in enumerate(blocks):
                    if idx in new_blocks_map:
                        final_blocks.append(new_blocks_map[idx])
                        fixed_count += 1
                    elif idx not in indices_to_remove:
                        final_blocks.append(block)
                azure_result.text_blocks = final_blocks

            self.timing["pix2text_ocr"] = time.time() - refine_start
            print(f"   优化了 {fixed_count} 个公式块 ({self.timing['pix2text_ocr']:.2f}s)")
            
            formula_result = None
            
        else:
            print("\n⏭️  跳过公式识别")
        
        return azure_result, formula_result

    def _should_refine_block(self, block) -> bool:
        """
        判断是否需要尝试 Refine
        
        核心思路：只对"看起来像公式但 OCR 可能识别错误"的文本进行 refine
        
        Args:
            block: TextBlock 对象（包含 text, has_size_variation 等属性）
        """
        # 兼容处理：如果传入的是字符串（旧调用方式），转换为 block like 对象
        if isinstance(block, str):
            class SimpleBlock:
                def __init__(self, text):
                    self.text = text
                    self.has_size_variation = False
            block = SimpleBlock(block)
        
        text = block.text
        if not text: 
            return False
        
        import re
        
        # ⭐ 排除SQL代码：检测SQL关键字和模式
        # 检测明确的SQL模式（多个关键字组合，或关键字+表名.字段名）
        # 1. 检测包含多个SQL关键字（如"WHERE xxx AND yyy", "AND xxx AND yyy"）
        sql_multi_keywords = r'\b(WHERE|AND|OR)\b.*\b(WHERE|AND|OR|IN)\b'
        if re.search(sql_multi_keywords, text, re.IGNORECASE):
            return False
        
        # 2. 检测SQL变量模式：表名.字段名 + 运算符（如P.age>35, T.t_name=xxx）
        # 这是SQL特有的模式，数学公式中不会出现
        if re.search(r'[A-Z]\.[a-z_][\w\-]*\s*[><=!]', text):
            return False
        
        # 3. 检测SQL条件表达式模式：xxx IN [...]
        if re.search(r'\w+\s+IN\s+\[', text, re.IGNORECASE):
            return False
        
        # 4. 检测简单的下划线变量名（如t_name, user_id等），且不含括号
        # 这种通常是SQL字段名/编程变量名，不是数学公式
        if re.match(r'^[a-z][a-z0-9]*(_[a-z0-9]+)+$', text, re.IGNORECASE) and len(text) <= 20:
            # 排除包含括号的（可能是函数调用f_i(x)等数学表达式）
            if '(' not in text and ')' not in text:
                return False
        
        # ⭐ 排除竖排文字（宽高比>2.2）：避免将竖排英文识别成公式
        if hasattr(block, 'polygon') and block.polygon:
            poly = block.polygon
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            if w > 0 and h / w > 2.2:
                return False
        
        # 排除：括号内的描述性文本 (如"(Multiple Nodes U1 to Un)")
        # 特征：括号包围，包含多个普通单词
        if re.match(r'^\([A-Z][a-z]+(?:\s+[A-Za-z]+){2,}.*\)$', text):
            return False
        
        # ⭐ 最高优先级：纯英文句子/短语（不需要 refine）
        # 需要在其他规则之前检查，避免被误判
        # 包含省略号（…、...）和常见标点，但不包括数学符号
        if re.match(r'^[a-zA-Z\s\-,.:;!\?\'"…]+$', text):
            if len(text) >= 5:
                # 排除明显的变量名（如C2f, X0等单词）
                words = text.split()
                if not (len(words) == 1 and re.match(r'^[A-Za-z]+\d+$', words[0])):
                    return False
        
        # ⭐ 优先检查：如果检测到字体大小变化，很可能包含下标/上标
        if getattr(block, 'has_size_variation', False):
            # 有字体大小变化，且文本看起来像公式（包含数学符号或变量）
            if re.search(r'[=+\-*/^_(){}[\]]|[a-zA-Z]\d|\d[a-zA-Z]', text):
                return True
        
        # 1. OCR 识别出问号通常表示不确定，需要 refine
        if '?' in text or '？' in text or '(?)' in text:
            return True
        
        # 2. 太长的文本不太可能是公式（但包含等号的除外）
        words = text.split()
        if len(words) > 10 and '=' not in text: 
            return False
        
        # 3. 排除常见的非公式格式
        # 尺寸格式：224×224, 1920x1080, (224×224) 等
        if re.match(r'^\(?[\d,]+\s*[×xX]\s*[\d,]+\)?$', text):
            return False
        
        # 4. 检测明确的数学符号（最高优先级）
        # 包括希腊字母、数学运算符、特殊符号
        math_symbols = r'[α-ωΑ-Ω\^≈≠≤≥∑∏∫√·×÷±∈∉⊂⊃∪∩€~]'
        if re.search(math_symbols, text):
            return True
        
        # 4. 包含等号的文本很可能是公式（优先检测）
        if '=' in text:
            # 排除明显的赋值语句（如 t=0, t=1 这种简单的）
            if not re.match(r'^[a-z]=[0-9]$', text):
                return True
        
        # 5. 检测明确的代码/标识符模式（不需要 refine）
        identifier_patterns = [
            r'^[A-Z][a-z]+([A-Z][a-z]+)+$',          # CamelCase: ConvFormer, ResNet
            r'^[A-Z][a-z]+[A-Z][a-z]*\d*$',          # CamelCase+数字: ResNet50
            r'^[A-Z][a-z]*\d+[a-zA-Z][a-zA-Z\-]*$',  # 带数字和后续字母: C2f, Conv2d
            r'^[a-zA-Z]+[_\-][a-zA-Z0-9_\-]+$',      # snake/kebab: conv_layer, C2f-ConvFormer
            r'^[A-Z]{2,}\d*$',                       # 纯大写缩写: CNN, GPU, YOLO, V2
        ]
        for pattern in identifier_patterns:
            if re.match(pattern, text):
                return False
        
        # 6. 检测短变量名（可能是数学变量，需要 refine）
        # 单字母+单/双数字：a1, Q3, x12, X0, X1 → 可能是下标
        if re.match(r'^[a-zA-Z][0-9]{1,2}$', text):
            return True
        
        # 6.5. 检测可能的数学下标文字（如 Rbatch, Xmax, Nmin, Kavg 等）
        # 单字母后跟常见数学后缀词（batch, max, min, avg, sum, total 等）
        subscript_suffixes = r'(batch|max|min|avg|sum|total|mean|std|var|norm|dim)'
        if re.match(rf'^[A-Z][a-z]?{subscript_suffixes}$', text, re.IGNORECASE):
            return True
        
        # 7. 中文文本（不需要 refine）
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        if chinese_chars > len(text) * 0.3:
            return False
        
        # 9. 检测数学表达式模式
        math_expr_patterns = [
            r'(?<![a-zA-Z])_(?![a-zA-Z])',           # 独立下标
            r'\\',                                   # LaTeX 转义
            r'\d+/\d+',                              # 分数: 1/2
            r'[a-zA-Z]\([a-zA-Z0-9,\s]+\)',          # 函数调用: f(x), V(X1, t, c)
            r'\|\|',                                 # 范数符号 ||
            r'\(\d+[+\-][a-zA-Z]\)',                 # 括号表达式: (1-t)
            r'[a-zA-Z]\d*\s*[\+\-\*]\s*[a-zA-Z]',   # 变量运算: X0 + t.X1
            r'^[\+\-][A-Za-z]',                      # 以运算符开头: +V, -X
            r'[A-Za-z]\.[A-Za-z]',                   # 点乘: V.At, x.y
        ]
        for pattern in math_expr_patterns:
            if re.search(pattern, text):
                return True
        
        # 10. 短的"字母+数字"组合
        if re.match(r'^[A-Za-z]+\d+$', text) and len(text) <= 4:
            return True
        
        # 11. 包含括号和点号的短文本（可能是公式）
        if '(' in text and ')' in text and len(text) <= 30:
            return True
        
        # 12. 默认不 refine（保守策略）
        return False

    def _split_text_and_formula(self, text: str) -> dict:
        """
        分析并分割混合文本（普通文字 + 公式）
        
        Args:
            text: 输入文本
            
        Returns:
            {
                'has_both': bool,      # 是否包含普通文字和公式
                'text': str,           # 普通文字部分
                'formula': str,        # 公式部分
                'split_pos': int       # 分割位置
            }
        """
        import re
        
        # 常见的混合模式：
        # 1. "描述文字 变量名 = 公式" → Full Private Key SKi = (Wi, Si)
        # 2. "标题：公式" → Loss: MSE(x, y)
        # 3. "说明 (公式)" → Attention (Q, K, V)
        # 4. "(描述性文本 变量下标)" → (Multiple Nodes U1 to Un)
        
        # 模式4：括号内的描述性文本 + 下标变量（如"(Multiple Nodes U1 to Un)"）
        # 特征：括号包围，包含多个普通单词和下标变量
        paren_pattern = r'^\(([A-Z][a-z]+(?:\s+[A-Za-z]+)+)\s+([A-Z]\d+(?:\s+to\s+[A-Z][a-z])?)\)$'
        paren_match = re.match(paren_pattern, text)
        
        if paren_match:
            # "(Multiple Nodes U1 to Un)" → "(Multiple Nodes" + "U_{1} to U_{n})"
            desc_part = paren_match.group(1).strip()  # "Multiple Nodes"
            var_part = paren_match.group(2).strip()   # "U1 to Un"
            
            # 这种情况下，整个括号都是描述性的，只有变量部分需要下标
            # 但为了保持格式一致，我们标记为混合文本
            # 注意：这种情况比较特殊，可能应该完全不进行Refine
            # 暂时返回False，在_should_refine_block中过滤
            return {
                'has_both': False,
                'text': text,
                'formula': '',
                'split_pos': 0
            }
        
        # 检测模式1：连续2+个普通英文单词（首字母可大写）+ 公式部分
        # 如："Full Private Key SKi"（3词）、"Select Secret (wi)"（2词）
        pattern = r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,})\s+(.*?)$'
        match = re.match(pattern, text)
        
        if match:
            text_part = match.group(1).strip()  # 普通文字
            formula_part = match.group(2).strip()  # 公式部分
            
            # 验证后半部分确实像公式
            # 必须包含：下标模式、等号、括号等
            formula_indicators = [
                r'[A-Z][a-z]?\d',  # SKi, Xi
                r'[A-Z][a-z]?(batch|max|min)',  # Rbatch
                r'=\s*\(',  # = (
                r'[A-Z]{2}\d',  # SK1
                r'\([A-Za-z][a-z]?i\)',  # (wi), (Yi), (Vi) - 括号内的下标变量（i结尾）
            ]
            
            is_formula = any(re.search(ind, formula_part) for ind in formula_indicators)
            
            if is_formula and len(text_part.split()) >= 2:
                return {
                    'has_both': True,
                    'text': text_part,
                    'formula': formula_part,
                    'split_pos': len(text_part)
                }
        
        # 没有检测到混合模式
        return {
            'has_both': False,
            'text': '',
            'formula': text,
            'split_pos': 0
        }

    def _extract_formula_from_mixed_latex(self, latex: str, text_part: str, formula_part: str) -> str:
        """
        从混合的 LaTeX 结果中提取纯公式部分
        
        策略：尝试移除前面被错误转换的普通文字部分
        
        Args:
            latex: Pix2Text 完整识别结果
            text_part: 原始的普通文字部分（如"Full Private Key"）
            formula_part: 原始的公式部分（如"SKi = (Wi, Si)"）
            
        Returns:
            提取后的纯公式 LaTeX
        """
        import re
        
        # 策略1：查找并移除前缀的 \mathXX{拆分的普通单词}
        # 例如：\mathsf{F u l l ~ P r i v a t e ~ K e y ~} SK_{i}=...
        #      → SK_{i}=...
        
        # 匹配模式：连续的 \mathXX{拆分单词}
        prefix_pattern = r'^(?:\\math(?:sf|bf|rm|it)\{[a-zA-Z\s~]+\}\s*)+'
        cleaned = re.sub(prefix_pattern, '', latex).strip()
        
        if cleaned and len(cleaned) < len(latex):
            # 成功移除了前缀
            return cleaned
        
        # 策略2：尝试从等号开始提取（因为公式通常包含 '='）
        if '=' in formula_part:
            # 查找第一个下标模式或等号之前的内容
            match = re.search(r'([A-Z_a-z\\]+\s*[_\{].*)', latex)
            if match:
                return match.group(1)
        
        # 策略3：直接返回原结果（无法分离）
        return latex

    def _is_refinement_meaningful(self, original: str, new_latex: str) -> bool:
        """
        判断 Refine 结果是否有实质性改变
        
        核心思路：只有当 Pix2Text 返回的是"真正的 LaTeX 公式"时才接受
        """
        import re
        
        # 0. 检查原文是否包含中文
        chinese_in_original = len(re.findall(r'[\u4e00-\u9fff]', original))
        
        if chinese_in_original > 0:
            # 原文包含中文，Pix2Text 对中文识别效果差
            chinese_in_latex = len(re.findall(r'[\u4e00-\u9fff]', new_latex))
            
            # 如果原文有中文但 Pix2Text 结果没有中文，说明中文被乱识别了
            if chinese_in_original >= 2 and chinese_in_latex == 0:
                return False
            
            # 检查是否包含乱七八糟的符号（Pix2Text 误识别中文的特征）
            garbage_patterns = [
                r'\\mathbb\{[A-Z]\}\s*\\frac',
                r'\\Re\\frac',
                r'\\pm1\s*\\pm',
                r'\\bot\\bot',
                r'\\overrightarrow\{1\s*0\}',
                r'\\Xi\}\s*\{5\}',
            ]
            for pattern in garbage_patterns:
                if re.search(pattern, new_latex):
                    return False
        
        # 0.5 检测 Pix2Text 的低质量输出
        # 检测常见函数被拆开的模式：e x p, l o g, s i n 等
        bad_spaced_funcs = [r'e\s+x\s+p', r'l\s+o\s+g', r's\s+i\s+n', r'c\s+o\s+s', r'l\s+n\s*\(']
        for pattern in bad_spaced_funcs:
            if re.search(pattern, new_latex, re.IGNORECASE):
                return False
        
        # 检测无效的 LaTeX 命令
        invalid_commands = [r'\\break', r'\\linebreak', r'\\\s+\/', r'\/\s*\/']
        for pattern in invalid_commands:
            if re.search(pattern, new_latex):
                return False
        
        # 检测英文单词被拆开的模式：如 v i s u a l, t h r e s h o l d
        # 但排除下标/上标中的拆分（那是有意义的）
        # 连续5个以上的 "字母+空格" 模式说明单词被拆开了
        # 先移除下标和上标中的内容
        temp_latex = re.sub(r'[_\^]\{[^\}]+\}', '', new_latex)
        temp_latex = re.sub(r'[_\^][a-zA-Z0-9]', '', temp_latex)
        spaced_letters = re.findall(r'[a-zA-Z]\s+[a-zA-Z]\s+[a-zA-Z]\s+[a-zA-Z]\s+[a-zA-Z]', temp_latex)
        if len(spaced_letters) >= 2:
            # 有多处单词被拆开（不在下标/上标中），说明 Pix2Text 输出质量差
            return False
        
        # 0.6 检测 Pix2Text 完全乱识别的情况
        # 如果原文包含多个英文单词，但 Pix2Text 输出完全不包含这些单词，说明乱识别了
        original_words = re.findall(r'[a-zA-Z]{3,}', original)
        if len(original_words) >= 2:
            # 原文有多个英文单词，检查 Pix2Text 是否保留了至少一个
            latex_text = re.sub(r'\\[a-zA-Z]+', '', new_latex)  # 移除 LaTeX 命令
            latex_text = re.sub(r'[^a-zA-Z]', '', latex_text)   # 只保留字母
            
            # 检查原文的单词是否在 LaTeX 输出中出现
            # ⚠️ 对于短变量名（≤3字符），使用模糊匹配（允许1个字符差异）
            words_preserved = 0
            for w in original_words:
                if w.lower() in latex_text.lower():
                    words_preserved += 1
                elif len(w) <= 3:
                    # 短变量名：允许1个字符差异（如vyj→vij, laEt→lat）
                    # 检查是否有至少50%的字符匹配
                    matched_chars = sum(1 for c in w.lower() if c in latex_text.lower())
                    if matched_chars >= len(w) * 0.5:
                        words_preserved += 1
            
            if words_preserved == 0:
                # 原文的单词一个都没保留，说明 Pix2Text 完全乱识别了
                return False
        
        # 1. 检测 Pix2Text 把英文单词转成公式的错误模式
        # 特征：\mathsf{T r a n s f o r m} 或 \mathbf{p r o p o r t i o n} 这种每个字母间加空格/波浪号
        # 匹配模式：\mathXX{字母 (空格或~) 字母 ...}
        spaced_word_pattern = r'\\math(sf|bf|rm|it)\{[a-zA-Z]([\s~]+[a-zA-Z]){3,}'
        has_spaced_words = bool(re.search(spaced_word_pattern, new_latex))
        
        # 检查原文是否包含长英文单词（4个字母以上的非公式词汇）
        original_words = re.findall(r'[a-zA-Z]{4,}', original)
        # 排除常见的数学函数词
        math_words = {'exp', 'log', 'sin', 'cos', 'tan', 'ln', 'logit', 'softmax', 'attn', 'mlp'}
        non_math_words = [w for w in original_words if w.lower() not in math_words]
        
        if has_spaced_words and len(non_math_words) >= 1:
            # 原文有非数学的长英文单词，且 Pix2Text 把它们转成了公式格式
            # 检查这些单词是否是描述性文本
            descriptive_words = {'transform', 'expected', 'score', 'proportion', 'weights', 
                                'link', 'from', 'curvature', 'features', 'inputs', 'outputs',
                                'noise', 'maturity', 'signal', 'rows', 'clean', 'invalid',
                                'daily', 'volume', 'word', 'vector', 'linguistic', 'feature',
                                'failure', 'probability', 'confusion', 'matrix', 'optional',
                                'attention', 'representation', 'standardize', 'normalize',
                                'full', 'private', 'public', 'key', 'secret', 'node', 'sender',
                                'receiver', 'client', 'server', 'batch', 'verification'}
            has_descriptive = any(w.lower() in descriptive_words for w in non_math_words)
            
            if has_descriptive:
                # 包含描述性英文单词，不应该被转成公式
                return False
        
        # 2. 清理 LaTeX 命令，提取核心文本
        core_latex = re.sub(r'\\(mathbf|mathrm|mathsf|textit|text|boldsymbol|mathcal|mathscr)\{([^\}]+)\}', r'\2', new_latex)
        core_latex = re.sub(r'\s|~', '', core_latex)
        core_latex = core_latex.replace('$', '')
        
        core_original = re.sub(r'\s', '', original)
        
        # 3. 如果去掉空格后完全相同，说明 Pix2Text 只是加了空格，没有意义
        if core_latex == core_original:
            return False
        
        # 4. 检查 Pix2Text 结果是否包含真正的数学命令（不是字体命令）
        # 注意：使用非 raw string，这样 \\ 会被解析成单个 \
        real_math_commands = [
            '\\frac', '\\sqrt', '\\sum', '\\prod', '\\int',
            '\\alpha', '\\beta', '\\gamma', '\\delta', '\\theta',
            '\\sigma', '\\lambda', '\\mu', '\\pi', '\\epsilon', '\\varepsilon',
            '\\partial', '\\nabla', '\\infty',
            '\\times', '\\cdot', '\\div',
            '\\leq', '\\geq', '\\neq', '\\approx', '\\equiv',
            '\\left', '\\right', '\\langle', '\\rangle',
            '\\mathbb', '\\mathcal',
            '\\ln', '\\log', '\\exp', '\\sin', '\\cos',
            '\\to', '\\rightarrow', '\\leftarrow', '\\sim',
        ]
        has_real_math = any(cmd in new_latex for cmd in real_math_commands)
        
        # 检查下标/上标是否有意义（不是单纯的字体下标）
        has_meaningful_subscript = bool(re.search(r'_\{[0-9a-z]\}|_[0-9]|\^\{[0-9a-zT]\}|\^[0-9T]', new_latex))
        
        # 5. 检查原文是否像公式
        original_looks_like_formula = bool(re.search(r'[=<>≤≥∑∏∈]|[a-zA-Z]\(|ln\(|log\(', original))
        
        # 6. 如果原文不像公式，且 Pix2Text 只是加了字体包装，拒绝
        if not original_looks_like_formula:
            if not has_real_math and not has_meaningful_subscript:
                return False
        
        # 7. 如果原文像公式，检查 Pix2Text 是否真正改进了
        if original_looks_like_formula:
            # 必须有真正的数学命令或有意义的下标
            if not has_real_math and not has_meaningful_subscript:
                return False
        
        # 8. 检查是否只是简单的字符替换
        original_no_space = re.sub(r'\s', '', original)
        latex_no_space = re.sub(r'\s', '', new_latex.replace('$', ''))
        latex_no_space = re.sub(r'\\[a-zA-Z]+\{|\}', '', latex_no_space)
        
        if original_no_space.lower() == latex_no_space.lower():
            return False
        
        return True
    
    def _get_dominant_color_by_length(self, blocks, indices) -> str:
        """
        根据文字长度选择主导颜色
        
        返回字符数最多的块的颜色，因为字符最多的块最可能代表主体颜色。
        
        Args:
            blocks: 文字块列表
            indices: 要考虑的块索引列表
            
        Returns:
            主导颜色（十六进制），如果无法确定则返回 None
        """
        if not indices:
            return None
        
        best_color = None
        max_length = 0
        
        for idx in indices:
            if idx >= len(blocks):
                continue
            block = blocks[idx]
            text_length = len(block.text) if hasattr(block, 'text') else 0
            color = getattr(block, 'font_color', None)
            
            if text_length > max_length and color:
                max_length = text_length
                best_color = color
        
        return best_color
    
    def _get_dominant_color_from_styles(self, block, azure_styles) -> str:
        """
        根据 Azure styles 中的字符级颜色信息，选择字符数最多的颜色
        
        Azure OCR 的 styles 包含每段文字的精确颜色信息，比 block 级别的颜色更准确。
        
        Args:
            block: 文字块
            azure_styles: Azure OCR 返回的 styles 列表
            
        Returns:
            主导颜色（十六进制），如果无法确定则返回 None
        """
        if not azure_styles or not hasattr(block, 'spans') or not block.spans:
            return None
        
        # 获取 block 的 span 范围
        block_span = block.spans[0] if isinstance(block.spans[0], dict) else {}
        block_offset = block_span.get('offset', 0)
        block_length = block_span.get('length', 0)
        block_end = block_offset + block_length
        
        # 统计各颜色覆盖的字符数
        color_char_counts = {}
        
        for style in azure_styles:
            color = style.get('color')
            if not color:
                continue
            
            for span in style.get('spans', []):
                span_start = span.get('offset', 0)
                span_length = span.get('length', 0)
                span_end = span_start + span_length
                
                # 计算与 block 重叠的字符数
                overlap_start = max(block_offset, span_start)
                overlap_end = min(block_end, span_end)
                overlap_length = max(0, overlap_end - overlap_start)
                
                if overlap_length > 0:
                    # 将颜色归类：深色（黑/灰）vs 彩色
                    normalized_color = self._normalize_color_category(color)
                    color_char_counts[normalized_color] = color_char_counts.get(normalized_color, 0) + overlap_length
        
        if not color_char_counts:
            return None
        
        # 选择字符数最多的颜色类别
        dominant_color = max(color_char_counts.items(), key=lambda x: x[1])[0]
        return dominant_color
    
    def _normalize_color_category(self, hex_color: str) -> str:
        """
        将颜色归类为代表性颜色
        
        深色（黑/灰）-> #000000
        其他保持原样
        """
        if not hex_color:
            return "#000000"
        
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6:
            return f"#{hex_color}"
        
        try:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
        except ValueError:
            return f"#{hex_color}"
        
        # 判断是否是深色（黑/灰）：RGB 值都较低且差异不大
        avg = (r + g + b) / 3
        max_diff = max(abs(r - g), abs(g - b), abs(r - b))
        
        # 如果平均值 < 80 且通道差异 < 40，认为是黑/灰色
        if avg < 80 and max_diff < 40:
            return "#000000"
        
        return f"#{hex_color}"

    def _is_spatially_close(self, poly1, poly2) -> bool:
        """判断两个多边形是否在空间上接近"""
        def get_bbox(p):
            xs, ys = [pt[0] for pt in p], [pt[1] for pt in p]
            return min(xs), min(ys), max(xs), max(ys)
        
        x1_min, y1_min, x1_max, y1_max = get_bbox(poly1)
        x2_min, y2_min, x2_max, y2_max = get_bbox(poly2)
        
        h1, h2 = y1_max - y1_min, y2_max - y2_min
        ref_h = max(h1, h2)
        
        # 水平方向检查（同一行的情况）
        y_overlap = min(y1_max, y2_max) - max(y1_min, y2_min)
        # 修改：Y重叠必须大于-0.2倍高度，避免将垂直排列的块误判为水平排列
        is_y_aligned = y_overlap > -ref_h * 0.2 
        
        if is_y_aligned:
            x_dist = max(0, x2_min - x1_max) if x1_min < x2_min else max(0, x1_min - x2_max)
            if x_dist < ref_h * 1.2:
                h_ratio = min(h1, h2) / max(h1, h2)
                if h_ratio > 0.6:
                    return True

        x_overlap = min(x1_max, x2_max) - max(x1_min, x2_min)
        wmin = min(x1_max - x1_min, x2_max - x2_min)
        
        if x_overlap > wmin * 0.2: 
            y_dist = max(0, y2_min - y1_max) if y1_min < y2_min else max(0, y1_min - y2_max)
            # 降低阈值：0.5 → 0.2，避免合并流程图中独立的多行文本
            if y_dist < ref_h * 0.2:
                return True
                
        return False

    def _merge_polygons(self, poly1, poly2):
        """合并两个多边形"""
        xs = [p[0] for p in poly1] + [p[0] for p in poly2]
        ys = [p[1] for p in poly1] + [p[1] for p in poly2]
        min_x, min_y, max_x, max_y = min(xs), min(ys), max(xs), max(ys)
        return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
    
    def _azure_to_dict_list(self, azure_result) -> List[Dict[str, Any]]:
        """将 Azure 结果转换为字典列表"""
        return [
            {
                "text": block.text,
                "polygon": block.polygon,
                "confidence": getattr(block, 'confidence', 1.0),
                "font_size_px": block.font_size_px,
                "is_latex": getattr(block, 'is_latex', False),
                "font_family": getattr(block, 'font_family', getattr(block, 'font_name', None)),
                "font_weight": getattr(block, 'font_weight', None),
                "font_style": getattr(block, 'font_style', None),
                "font_color": getattr(block, 'font_color', None),
                "is_bold": getattr(block, 'is_bold', False),
                "is_italic": getattr(block, 'is_italic', False),
                "spans": getattr(block, 'spans', [])
            }
            for block in azure_result.text_blocks
        ]
    
    def _detect_global_font(self, azure_result) -> str:
        """检测全局主字体"""
        if not azure_result.text_blocks:
            return "Arial Narrow"
        
        def get_area(block):
            polygon = block.polygon
            if not polygon or len(polygon) < 4:
                return 0
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            return (max(xs) - min(xs)) * (max(ys) - min(ys))
        
        best_block = max(azure_result.text_blocks, key=get_area)
        font = getattr(best_block, 'font_name', None)
        
        if font:
            print(f"   ✨ 识别到主字体: {font}")
            return font
        
        return "Arial Narrow"
    
    def _save_metadata(self, image_path: str, output_path: str, text_blocks: List[Dict], 
                       image_width: int, image_height: int):
        """保存元数据"""
        import json
        from datetime import datetime
        
        metadata_path = Path(output_path).parent / "metadata.json"
        
        font_stats = {}
        for block in text_blocks:
            font = block.get("font_family", "unknown")
            font_stats[font] = font_stats.get(font, 0) + 1
        
        metadata = {
            "version": "3.0",
            "generated_at": datetime.now().isoformat(),
            "input": {"path": image_path, "width": image_width, "height": image_height},
            "output": {"drawio_path": output_path},
            "mode": f"azure+{self.formula_engine}",
            "timing": self.timing,
            "statistics": {
                "total_cells": len(text_blocks),
                "text_cells": sum(1 for b in text_blocks if not b.get("is_latex")),
                "formula_cells": sum(1 for b in text_blocks if b.get("is_latex")),
                "fonts": font_stats
            },
            "text_blocks": [
                {
                    "id": i + 1,
                    "text": block["text"][:100],
                    "position": block["geometry"],
                    "style": {
                        "font_size": block.get("font_size"),
                        "font_family": block.get("font_family"),
                        "font_weight": block.get("font_weight"),
                        "font_color": block.get("font_color"),
                        "is_formula": block.get("is_latex", False)
                    }
                }
                for i, block in enumerate(text_blocks)
            ]
        }
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"   元数据已保存: {metadata_path}")
    
    def _generate_debug_image(self, image_path: str, output_path: str):
        """生成调试图"""
        try:
            # 简单实现：复制原图作为调试图
            from PIL import Image
            img = Image.open(image_path)
            img.save(output_path)
        except Exception as e:
            print(f"   ⚠️ 调试图生成失败: {e}")
    
    def _print_stats(self, text_blocks: List[Dict]):
        """打印统计信息"""
        print(f"\n⏱️  耗时:")
        print(f"   Azure OCR: {self.timing['azure_ocr']:.2f}s")
        print(f"   Pix2Text:  {self.timing['pix2text_ocr']:.2f}s")
        print(f"   处理:      {self.timing['processing']:.2f}s")
        print(f"   总计:      {self.timing['total']:.2f}s")
        
        print(f"\n✅ 完成！共 {len(text_blocks)} 个文本单元格")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python restorer.py <image_path> [output_path]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    restorer = TextRestorer()
    restorer.restore(image_path, output_path)
