"""
字体特征检测器 — 区分 Arial 和 Comic Sans

核心差异（从字母图像对比可见）：

字母 a:
    Comic Sans: 单层圆形结构（类似 o 加右侧尾巴），左侧轮廓为平滑弧线（C形）
    Arial:      双层结构（顶盖 + 细颈 + 肚子），左侧轮廓有明显 S 形凹陷

字母 t:
    Comic Sans: 笔画宽度均匀，底部无明显右弯钩，整体有机/圆润
    Arial:      底部向右弯曲扩张，底端明显比竖干宽

检测流程：
    1. 将文本块图像按连通组件拆分为单个字母
    2. 根据文本内容定位小写 a 和 t 的位置
    3. 对每个 a/t 单独进行特征检测
    4. 多数投票：超过半数判定为 Comic Sans → 整块 Comic Sans

使用示例：
    from processors.font_detector import FontDetector
    detector = FontDetector()
    font = detector.detect_font_from_text("hello", text_image)
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import os


class FontDetector:
    """基于图像特征的字体检测器，主要区分 Arial 和 Comic Sans MS"""

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.cache = {}
        self._debug_counter = 0
        self._debug_dir = "/tmp/font_crop_debug"
        if debug:
            os.makedirs(self._debug_dir, exist_ok=True)

    # ================================================================
    #  公开接口
    # ================================================================

    def detect_font_from_text(
        self,
        text: str,
        text_image: Optional[np.ndarray] = None,
    ) -> str:
        """
        检测文本块字体。

        Args:
            text: 文本内容（用于定位字母位置）
            text_image: 文本区域的裁剪图像

        Returns:
            "Comic Sans MS" 或 "Arial Narrow"
        """
        if text_image is None:
            return "Arial Narrow"

        cache_key = f"{text}_{text_image.shape}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        result = self._detect(text, text_image)

        if self.debug:
            print(f"  字体检测: '{text}' -> {result}")

        self.cache[cache_key] = result
        return result

    # ================================================================
    #  核心检测流程
    # ================================================================

    def _detect(self, text: str, image: np.ndarray) -> str:
        # 只检测小写 a/t（大写 A/T 形状完全不同，不适用此算法）
        if 'a' not in text and 't' not in text:
            if self.debug:
                print(f"    ⚠ 不含小写 a/t，默认 Arial")
            return "Arial Narrow"

        # 1) 分割为单字符
        char_crops = self._segment_characters(image)
        if not char_crops:
            if self.debug:
                print(f"    ⚠ 分割失败，默认 Arial")
            return "Arial Narrow"

        # 2) 建立文本 ↔ 裁剪区域的对应（保留原始大小写）
        text_chars = [c for c in text if c.isalnum()]
        if not text_chars:
            return "Arial Narrow"

        ratio = len(char_crops) / len(text_chars)
        if self.debug:
            print(f"    分割: {len(char_crops)} 个组件 vs {len(text_chars)} 个字符 (ratio={ratio:.2f})"
                  f" [原图:{image.shape[1]}x{image.shape[0]}]")
        if ratio < 0.7 or ratio > 1.3:
            if self.debug:
                print(f"    ⚠ 分割数量不匹配 ({len(char_crops)} vs {len(text_chars)})，默认 Arial")
            return "Arial Narrow"

        # 3) 计算大写字母参考高度（用于逐字符判断是否实为大写）
        #    OCR 大小写可能不准（如图中全大写但 OCR 返回混合大小写）
        #    方法：收集 OCR 标记为大写的字符高度，作为参考
        #    对每个待分析的 'a'/'t'，如果其高度接近大写参考 → 实际是大写 → 跳过
        n = min(len(char_crops), len(text_chars))
        uppercase_heights = []
        for i in range(n):
            if text_chars[i].isupper():
                uppercase_heights.append(char_crops[i][1][3])
        uppercase_ref = float(np.mean(uppercase_heights)) if uppercase_heights else 0
        if self.debug and uppercase_ref > 0:
            print(f"    大写参考高度: {uppercase_ref:.1f} (来自 {len(uppercase_heights)} 个大写字符)")

        a_results = []
        t_results = []

        for i in range(n):
            char_img = char_crops[i][0]
            char = text_chars[i]

            if char == 'a':
                # 判断是否实为大写 A：高度接近大写参考 → 跳过
                char_h = char_crops[i][1][3]
                if uppercase_ref > 0 and char_h / uppercase_ref > 0.85:
                    if self.debug:
                        print(f"      a[{i}]: 高度{char_h:.0f}接近大写参考{uppercase_ref:.0f}"
                              f"(ratio={char_h/uppercase_ref:.2f})，疑似大写A，跳过")
                    continue

                v = self._detect_letter_a(char_img)
                if self.debug:
                    label = 'Comic' if v else ('Arial' if v is False else 'Skip')
                    self._debug_counter += 1
                    cv2.imwrite(
                        f"{self._debug_dir}/{self._debug_counter:04d}_a_{label}_{text[:10]}.png",
                        char_img
                    )
                if v is not None:
                    a_results.append(v)
                    if self.debug:
                        print(f"    a[{i}]: {'Comic' if v else 'Arial'}")

            elif char == 't':
                # 判断是否实为大写 T：高度接近大写参考 → 跳过
                char_h = char_crops[i][1][3]
                if uppercase_ref > 0 and char_h / uppercase_ref > 0.92:
                    if self.debug:
                        print(f"      t[{i}]: 高度{char_h:.0f}接近大写参考{uppercase_ref:.0f}"
                              f"(ratio={char_h/uppercase_ref:.2f})，疑似大写T，跳过")
                    continue

                v = self._detect_letter_t(char_img)
                if self.debug:
                    label = 'Comic' if v else ('Arial' if v is False else 'Skip')
                    self._debug_counter += 1
                    cv2.imwrite(
                        f"{self._debug_dir}/{self._debug_counter:04d}_t_{label}_{text[:10]}.png",
                        char_img
                    )
                if v is not None:
                    t_results.append(v)
                    if self.debug:
                        print(f"    t[{i}]: {'Comic' if v else 'Arial'}")

        # 4) 投票
        all_results = a_results + t_results
        if not all_results:
            if self.debug:
                print(f"    ⚠ 无有效检测结果，默认 Arial")
            return "Arial Narrow"

        comic_count = sum(1 for r in all_results if r)
        total = len(all_results)

        if self.debug:
            print(f"    📊 a={a_results} t={t_results} => Comic {comic_count}/{total}")

        if comic_count > total / 2:
            return "Comic Sans MS"
        return "Arial Narrow"

    # ================================================================
    #  字符分割（连通组件 + 自动放大 + i/j点合并）
    # ================================================================

    def _segment_characters(
        self, image: np.ndarray
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """将文本块图像拆分为单个字符，按 x 坐标排序。
        
        处理流程：
        1. 自动放大（确保足够分辨率）
        2. 二值化 + 轮廓提取
        3. 合并 i/j 的点（小组件与正下方组件合并）
        4. 按 x 排序，裁剪
        """
        # --- 1. 自动放大 ---
        ih, iw = image.shape[:2]
        target_h = 60
        scale = 1.0
        if ih < target_h:
            scale = target_h / ih
            scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            scaled = image
        
        sh, sw = scaled.shape[:2]
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY) if len(scaled.shape) == 3 else scaled.copy()
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        # --- 2. 提取所有有效边界框 ---
        raw_boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w < 2 or h < 2:
                continue
            if w > sw * 0.5:
                continue
            raw_boxes.append([x, y, w, h])

        if not raw_boxes:
            return []

        # --- 3. 合并 i/j 的点：小组件 + 正下方大组件 → 合并 ---
        # 计算中位高度
        heights = [b[3] for b in raw_boxes]
        median_h = float(np.median(heights))

        # 分离：小组件（高度 < 中位高度的 40%）和正常组件
        small = []
        normal = []
        for b in raw_boxes:
            if b[3] < median_h * 0.4:
                small.append(b)
            else:
                normal.append(b)

        # 对每个小组件，检查是否有正常组件在其正下方且水平重叠
        merged_flags = [False] * len(small)
        for si, sb in enumerate(small):
            sx, sy, sw_s, sh_s = sb
            sx_center = sx + sw_s / 2
            for ni, nb in enumerate(normal):
                nx, ny, nw_n, nh_n = nb
                # 水平重叠：小组件中心在大组件水平范围内
                if nx <= sx_center <= nx + nw_n:
                    # 小组件在大组件上方（y 更小）
                    if sy + sh_s <= ny + median_h * 0.3:
                        # 合并：扩展大组件的边界框
                        new_x = min(nx, sx)
                        new_y = min(ny, sy)
                        new_x2 = max(nx + nw_n, sx + sw_s)
                        new_y2 = max(ny + nh_n, sy + sh_s)
                        normal[ni] = [new_x, new_y, new_x2 - new_x, new_y2 - new_y]
                        merged_flags[si] = True
                        break

        # 未合并的小组件如果高度太小（碎片），丢弃
        boxes = normal[:]
        for si, sb in enumerate(small):
            if not merged_flags[si] and sb[3] >= median_h * 0.3:
                boxes.append(sb)

        # --- 4. 过滤异常 ---
        filtered = []
        for x, y, w, h in boxes:
            if h > sh * 0.8:
                continue
            ar = h / w if w > 0 else 0
            if ar < 0.3 or ar > 8.0:
                continue
            filtered.append((x, y, w, h))

        filtered.sort(key=lambda b: b[0])

        # --- 5. 裁剪 ---
        result = []
        for x, y, w, h in filtered:
            pad = 1
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(scaled.shape[1], x + w + pad), min(scaled.shape[0], y + h + pad)
            crop = scaled[y1:y2, x1:x2]
            if crop.size > 0:
                result.append((crop, (x, y, w, h)))
        return result

    # ================================================================
    #  字母 a 检测
    # ================================================================

    def _detect_letter_a(self, char_img: np.ndarray) -> Optional[bool]:
        """
        检测单个字母 a 的字体。

        核心原理 — 三段式左侧轮廓分析（S-shape vs C-shape）：
            Arial 'a' (双层):
                顶盖(靠左) → 细颈(往右凹) → 肚子(靠左)
                → 左侧轮廓呈 S 形：中段比两端更靠右
            Comic Sans 'a' (单层圆形):
                弧线顶部(靠右/窄) → 弧线中间(靠左/宽) → 弧线底部(靠右/窄)
                → 左侧轮廓呈 C 形：中段比两端更靠左

        检测方法：
            1. 逐行扫描最左白像素，得到 left_edge 曲线
            2. 平滑去噪
            3. 将曲线三等分，比较中段 vs 两端的平均值
            4. 中段 > 两端（S形）= Arial
            5. 中段 < 两端（C形）= Comic Sans
            6. 差异不明显 = 无法判定

        Returns:
            True: Comic Sans, False: Arial, None: 无法检测
        """
        try:
            binary = self._binarize(char_img)
            h, w = binary.shape
            if h < 8 or w < 5:
                return None

            # 字母 'a' 形状检查：大致方形到稍高
            ar = h / w
            if ar < 0.7 or ar > 2.5:
                if self.debug:
                    print(f"      a: 宽高比异常({w}x{h}, ar={ar:.1f})，跳过")
                return None
            raw_edges = []
            for row in range(h):
                whites = np.where(binary[row] > 128)[0]
                if len(whites) > 0:
                    raw_edges.append(whites[0])
                # 空行跳过，不填充假值（避免扭曲三段平均）

            if len(raw_edges) < 6:
                return None

            # 2. 去掉顶部/底部 10% 的边缘噪声
            margin = max(1, int(len(raw_edges) * 0.1))
            edges = np.array(raw_edges[margin : len(raw_edges) - margin], dtype=float)
            if len(edges) < 6:
                return None

            # 3. 平滑（窗口 = 高度的 1/5，最小3）
            k = max(3, len(edges) // 5)
            if k % 2 == 0:
                k += 1
            smoothed = np.convolve(edges, np.ones(k) / k, mode="valid")
            if len(smoothed) < 6:
                return None

            # 4. 三段平均值
            n = len(smoothed)
            top_avg = float(np.mean(smoothed[: n // 3]))
            mid_avg = float(np.mean(smoothed[n // 3 : 2 * n // 3]))
            bot_avg = float(np.mean(smoothed[2 * n // 3 :]))

            # S 值：中段相对两端的位置偏移（归一化到字符宽度）
            # S > 0：中段比两端更靠右 → Arial 的颈部凹陷（S形）
            # S < 0：中段比两端更靠左 → Comic Sans 的圆弧突出（C形）
            s_value = (mid_avg - (top_avg + bot_avg) / 2) / w

            if self.debug:
                print(
                    f"      a: S={s_value:.3f} "
                    f"(top={top_avg:.1f} mid={mid_avg:.1f} bot={bot_avg:.1f})"
                )

            # 一致性检查：三段差异过大说明裁到了错误字符
            spread = max(top_avg, mid_avg, bot_avg) - min(top_avg, mid_avg, bot_avg)
            if spread > w * 0.3:
                if self.debug:
                    print(f"      a: 三段差异过大({spread:.1f} > {w*0.3:.1f})，跳过")
                return None

            # 判定
            if s_value > 0.04:
                return False  # Arial (S形：中段凹陷）
            elif s_value < -0.04:
                return True   # Comic Sans (C形：中段突出)
            else:
                return None   # 特征不明显，无法判定

        except Exception as e:
            if self.debug:
                print(f"      ❌ a 检测异常: {e}")
            return None

    # ================================================================
    #  字母 t 检测
    # ================================================================

    def _detect_letter_t(self, char_img: np.ndarray) -> Optional[bool]:
        """
        检测单个字母 t 的字体。

        核心原理 — 底部右侧延伸：
            Arial 't':
                底部有明显的右弯钩，底端最右像素远远超过竖干中段
                → 底部扩散率 > 1.25，右边缘偏移显著
            Comic Sans 't':
                底部笔画宽度与竖干相近，无弯钩
                → 底部扩散率 ≈ 1.0，右边缘偏移极小

        检测方法（双重验证）：
            1. 宽度扩散率：底端宽度 / 竖干中段宽度
            2. 右边缘偏移：底端右像素 − 中段右像素（归一化到字符宽度）
            两者之一超过阈值 → Arial

        Returns:
            True: Comic Sans, False: Arial, None: 无法检测
        """
        try:
            binary = self._binarize(char_img)
            h, w = binary.shape
            if h < 10 or w < 3:
                return None

            # 字母 't' 形状检查：应该比宽更高
            ar = h / w
            if ar < 1.0:
                if self.debug:
                    print(f"      t: 宽高比异常({w}x{h}, ar={ar:.1f})，跳过")
                return None

            # 取下半部分（避开横杠区域）
            bottom_start = int(h * 0.55)
            bottom = binary[bottom_start:]
            bh = bottom.shape[0]
            if bh < 4:
                return None

            # 逐行统计宽度和右边缘位置
            widths = []
            right_edges = []
            for r in range(bh):
                whites = np.where(bottom[r] > 0)[0]
                if len(whites) > 0:
                    widths.append(whites[-1] - whites[0] + 1)
                    right_edges.append(whites[-1])

            if len(widths) < 4:
                return None

            # --- 指标1: 宽度扩散率 ---
            # 底端 20%
            tail_n = max(1, len(widths) // 5)
            bottom_width = float(np.mean(widths[-tail_n:]))
            # 中段 30%-60%
            m1 = len(widths) // 3
            m2 = 2 * len(widths) // 3
            stem_width = float(np.mean(widths[m1:m2])) if m2 > m1 else float(np.mean(widths))
            if stem_width < 1:
                return None

            flare_ratio = bottom_width / stem_width

            # --- 指标2: 右边缘偏移 ---
            bottom_right = float(np.mean(right_edges[-tail_n:]))
            mid_right = float(np.mean(right_edges[m1:m2])) if m2 > m1 else float(np.mean(right_edges))
            right_shift = (bottom_right - mid_right) / w if w > 0 else 0.0

            if self.debug:
                print(f"      t: flare={flare_ratio:.2f}, right_shift={right_shift:.3f}")

            # 过滤异常值：flare < 0.5 说明字符裁剪有误
            if flare_ratio < 0.5:
                if self.debug:
                    print(f"      t: flare异常低({flare_ratio:.2f})，跳过")
                return None

            # Arial: 底部有明显的右扩张（任一指标超阈值）
            if flare_ratio > 1.15 or right_shift > 0.10:
                return False  # Arial (有弯钩)

            # Comic Sans: flare 在 0.85-1.15 范围（底部宽度与竖干相近，无弯钩）
            if flare_ratio >= 0.85:
                return True   # Comic Sans (底部直，宽度正常)

            # flare 在 0.5-0.85 说明底部比竖干更窄，测量可能不准
            if self.debug:
                print(f"      t: flare模糊范围({flare_ratio:.2f})，跳过")
            return None  # 不做判定

        except Exception as e:
            if self.debug:
                print(f"      ❌ t 检测异常: {e}")
            return None

    # ================================================================
    #  辅助函数
    # ================================================================

    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """灰度 + 二值化（统一用 OTSU，放大后的图像更稳定）"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary


if __name__ == "__main__":
    detector = FontDetector(debug=True)
    print("字体检测器已初始化")
    print("检测字母: a (左侧S形凹陷), t (底部右弯钩)")
    print("Arial: a有双层结构, t有底部弯钩")
    print("Comic Sans: a是圆形单层, t底部直线")
