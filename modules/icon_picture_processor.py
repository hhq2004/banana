"""
任务2：Icon/Picture非基本图形处理模块

功能：
    - 处理icon、picture、logo、chart、function_graph等非基本图形
    - 使用RMBG-2.0进行背景移除
    - 将处理后的图片转换为base64
    - 生成XML片段

负责人：[已实现]
负责任务：任务2 - Icon、picture、函数图等非基本图形类

使用示例：
    from modules import IconPictureProcessor, ProcessingContext
    
    processor = IconPictureProcessor()
    context = ProcessingContext(image_path="test.png")
    context.elements = [...]  # 从SAM3获取的元素
    
    result = processor.process(context)
    # 处理后的元素会包含 base64 和 xml_fragment 字段

接口说明：
    输入：
        - context.elements: ElementInfo列表，筛选出需要处理的非基本图形
        - context.image_path: 原始图片路径，用于裁剪
        
    输出：
        - 更新 element.base64: 处理后的base64编码图片
        - 更新 element.has_transparency: 是否已去除背景
        - 更新 element.xml_fragment: 该元素的XML片段
"""

import os
import sys
import io
import base64
import yaml
from functools import lru_cache
from typing import Optional, List
from PIL import Image
import numpy as np
import cv2

# ONNX Runtime（可选依赖）
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("[IconPictureProcessor] Warning: onnxruntime not available, RMBG disabled")

# 超分模型（可选依赖）
try:
    import torch
    from spandrel import ModelLoader
    SPANDREL_AVAILABLE = True
except ImportError:
    SPANDREL_AVAILABLE = False
    print("[IconPictureProcessor] Warning: spandrel/torch not available, Upscale disabled")

from .base import BaseProcessor, ProcessingContext, ModelWrapper
from .data_types import ElementInfo, ProcessingResult, LayerLevel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from litserve_model.request import rmbg_request


# ======================== 配置加载（统一使用 config.read_config）========================
from config.read_config import load_config as _load_global_config


# ======================== RMBG-2.0 模型封装 ========================
class RMBGModel(ModelWrapper):
    """
    RMBG-2.0 背景移除模型封装
    
    基于 ONNX Runtime 实现，支持 CUDA 加速
    
    使用示例：
        model = RMBGModel(model_path)
        model.load()
        
        # 一行调用去除背景
        rgba_image = model.remove_background(pil_image)
    """
    
    # 模型固定输入尺寸
    INPUT_SIZE = (1024, 1024)
    
    def __init__(self, model_path: str = None, device: str = None):
        super().__init__()
        if model_path:
            self.model_path = model_path
        else:
            cfg = _load_global_config()
            self.model_path = (
                cfg.get("rmbg", {}).get("model_path")
                or self._get_default_path()
            )
        self._session = None
        self._input_name = None
        self._output_name = None
        self._device_spec = (str(device).strip() if device is not None else "auto").lower()
        self.device = device or "auto"
        self._active_provider = None
    
    def _get_default_path(self) -> str:
        """获取默认模型路径"""
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "rmbg", "model.onnx"
        )

    def _parse_device_spec(self) -> tuple:
        """解析用户指定的device字符串"""
        spec = self._device_spec
        if not spec or spec in {"auto", "default"}:
            return "auto", None
        if spec.startswith("cuda") or spec.startswith("gpu"):
            return "cuda", self._extract_device_index(spec)
        if spec.startswith("cpu"):
            return "cpu", None
        return "auto", None

    def _extract_device_index(self, spec: str) -> int:
        tail = ""
        if ":" in spec:
            tail = spec.split(":", 1)[1]
        elif spec.startswith("cuda"):
            tail = spec[4:]
        elif spec.startswith("gpu"):
            tail = spec[3:]
        tail = tail.strip()
        digits = []
        for ch in tail:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        return int("".join(digits)) if digits else 0

    def _build_provider_attempts(self, available_providers):
        """根据device配置构建provider加载策略"""
        attempts = []
        available_set = set(available_providers or [])

        def add_attempt(providers, description, options=None):
            attempts.append({
                "providers": providers,
                "description": description,
                "options": options,
            })

        mode, device_id = self._parse_device_spec()

        if mode == "cuda":
            if 'CUDAExecutionProvider' in available_set:
                cuda_options = [{"device_id": device_id or 0}]
                add_attempt(['CUDAExecutionProvider'], f"CUDA(device_id={device_id or 0})", cuda_options)
            else:
                print("[RMBGModel] Requested CUDA device but CUDAExecutionProvider is unavailable, falling back to CPU")
            add_attempt(['CPUExecutionProvider'], "CPU fallback")
            return attempts

        if mode == "cpu":
            add_attempt(['CPUExecutionProvider'], "CPU only")
            return attempts

        # auto 模式：优先GPU，再回退CPU
        if 'CUDAExecutionProvider' in available_set:
            add_attempt(['CUDAExecutionProvider'], "CUDA (auto)", [{"device_id": 0}])
        add_attempt(['CPUExecutionProvider'], "CPU fallback")
        return attempts
    
    def load(self):
        """
        加载RMBG-2.0 ONNX模型
        支持自动降级：CUDA失败时自动回退到CPU
        """
        if self._is_loaded:
            return
        
        if not ONNX_AVAILABLE:
            print("[RMBGModel] Warning: onnxruntime not available, using fallback mode")
            self._is_loaded = True
            return
        
        if not os.path.exists(self.model_path):
            print(f"[RMBGModel] Warning: Model file not found at {self.model_path}, using fallback mode")
            self._is_loaded = True
            return
        
        # 配置 ONNX Runtime 选项，屏蔽警告
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3  # ERROR only
        session_options.enable_profiling = False
        
        # 获取可用的 providers
        available_providers = ort.get_available_providers()

        provider_attempts = self._build_provider_attempts(available_providers)
        
        for attempt in provider_attempts:
            providers = [p for p in attempt['providers'] if p in available_providers]
            if not providers:
                continue

            provider_options = attempt.get('options')
            if provider_options:
                # 对齐有效provider的options
                aligned_options = []
                for provider, option in zip(attempt['providers'], provider_options):
                    if provider in providers:
                        aligned_options.append(option)
                provider_options = aligned_options or None
            
            try:
                print(f"[RMBGModel] Trying to load with {attempt['description']} ({providers})...")
                session_kwargs = {
                    'path_or_bytes': self.model_path,
                    'providers': providers,
                    'sess_options': session_options,
                }
                if provider_options is not None:
                    session_kwargs['provider_options'] = provider_options
                self._session = ort.InferenceSession(**session_kwargs)
                
                self._input_name = self._session.get_inputs()[0].name
                self._output_name = self._session.get_outputs()[0].name
                self._providers = providers
                self._active_provider = attempt['description']
                
                self._is_loaded = True
                print(f"[RMBGModel] Model loaded successfully with {attempt['description']}")
                return
                
            except Exception as e:
                print(f"[RMBGModel] Failed to load with {attempt['description']}: {e}")
                continue
        
        # 所有尝试都失败，使用 fallback 模式
        print("[RMBGModel] Warning: All loading attempts failed, using fallback mode (no background removal)")
        self._is_loaded = True
    
    def _preprocess(self, img: np.ndarray) -> tuple:
        """
        图片预处理：缩放、归一化、转CHW格式
        
        Args:
            img: RGB格式的numpy数组
            
        Returns:
            (preprocessed_image, original_size)
        """
        # RMBG-2.0 要求 BGR 格式
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]
        
        # 缩放到模型输入尺寸
        img_resized = cv2.resize(img_bgr, self.INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        
        # 归一化到 [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # 转 CHW 格式 (HWC -> CHW)
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        
        # 增加 batch 维度 (3, 1024, 1024) -> (1, 3, 1024, 1024)
        img_batch = np.expand_dims(img_transposed, axis=0)
        
        return img_batch, (h, w)
    
    def _postprocess(self, pred: np.ndarray, original_size: tuple) -> np.ndarray:
        """
        后处理：提取 alpha 通道并还原到原图尺寸
        
        Args:
            pred: 模型输出
            original_size: (height, width)
            
        Returns:
            alpha通道 (uint8, 0-255)
        """
        # 移除 batch 维度，提取 alpha 通道 (1, 1, 1024, 1024) -> (1024, 1024)
        alpha = pred[0, 0, :, :]
        
        # 缩放回原图尺寸
        alpha_resized = cv2.resize(alpha, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
        
        # 归一化到 [0, 255] 并转 8 位
        alpha_resized = (alpha_resized * 255).astype(np.uint8)
        
        return alpha_resized
    
    def predict(self, image: Image.Image) -> Image.Image:
        """
        背景移除推理
        支持自动降级：GPU推理失败时自动回退到CPU
        
        Args:
            image: PIL图像（RGB格式）
            
        Returns:
            去除背景后的RGBA图像
        """
        if not self._is_loaded:
            self.load()
        
        # 如果模型未成功加载，返回 fallback 结果
        if self._session is None:
            print("[RMBGModel] 难道是模型未正常加载？ Warning: Model not loaded, using fallback mode (no background removal)")
            return image.convert("RGBA")
        
        # 转换为 numpy 数组
        img = np.array(image)
        
        # 预处理
        img_input, original_size = self._preprocess(img)
        
        # 尝试推理（带错误处理）
        try:
            pred = self._session.run([self._output_name], {self._input_name: img_input})[0]
        except Exception as e:
            # GPU 推理失败，尝试重新加载为 CPU 模式
            if hasattr(self, '_providers') and 'CUDAExecutionProvider' in self._providers:
                print(f"[RMBGModel] GPU inference failed (OOM), switching to CPU...")
                print("[RMBGModel] Warning: Model not loaded, 这可不行啊")
                
                try:
                    # 释放当前 session
                    self._session = None
                    
                    # 重新创建 CPU-only session
                    session_options = ort.SessionOptions()
                    session_options.log_severity_level = 3
                    
                    self._session = ort.InferenceSession(
                        self.model_path,
                        providers=['CPUExecutionProvider'],
                        sess_options=session_options
                    )
                    self._providers = ['CPUExecutionProvider']
                    
                    # 重试推理
                    pred = self._session.run([self._output_name], {self._input_name: img_input})[0]
                    print("[RMBGModel] CPU inference successful")
                    
                except Exception as e2:
                    print(f"[RMBGModel] CPU inference also failed: {e2}")
                    print("[RMBGModel] Falling back to no background removal")
                    return image.convert("RGBA")
            else:
                print(f"[RMBGModel] Inference failed: {e}, using fallback (no background removal)")
                return image.convert("RGBA")
        
        # 后处理得到 alpha 通道
        alpha = self._postprocess(pred, original_size)
        
        # 合并 alpha 通道到原图（生成 RGBA 图片）
        img_rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        img_rgba[:, :, 3] = alpha
        
        # 转换为 PIL 图片
        return Image.fromarray(img_rgba)
    
    def remove_background(self, image: Image.Image) -> Image.Image:
        """predict的别名，更语义化"""
        return self.predict(image)
    
    def unload(self):
        """释放模型资源"""
        self._session = None
        self._is_loaded = False

# ======================== 超分模型 Realesrgan 封装 ========================
class UpscaleModel(ModelWrapper):
    """
    超分模型封装（基于 spandrel 模型加载）
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        fp16: bool = False,
    ):
        super().__init__()
        if model_path:
            self.model_path = model_path
        else:
            cfg = _load_global_config()
            self.model_path = (
                cfg.get("upscale", {}).get("model_path")
                or self._get_default_path()
            )
        self.device = device
        self.fp16 = fp16
        self.scale = 1
        self._model = None
        self._device: Optional["torch.device"] = None
        self._use_half = False

    def _resolve_device(self) -> "torch.device":
        if not SPANDREL_AVAILABLE:
            raise ImportError("spandrel/torch 未安装")
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            return torch.device("cpu")
        # allow explicit strings like "cpu" or "cuda:1"
        return torch.device(self.device)
    
    def  _get_default_path(self) -> str:
        """获取默认超分模型路径"""
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "RealESRGAN_x4plus_anime_6B.pth"
        )

    def load(self):
        """加载超分模型，失败时静默降级"""
        if self._is_loaded:
            return

        if not SPANDREL_AVAILABLE:
            print("[UpscaleModel] spandrel/torch 未安装，跳过超分")
            self._is_loaded = True
            return

        if not os.path.exists(self.model_path):
            print(f"[UpscaleModel] 模型文件不存在: {self.model_path}，跳过超分")
            self._is_loaded = True
            return

        try:
            self._device = self._resolve_device()
            descriptor = ModelLoader().load_from_file(str(self.model_path))
            self.scale = descriptor.scale
            self._model = descriptor.model
            self._model.eval().to(self._device)
            self._use_half = bool(self.fp16 and descriptor.supports_half and self._device.type == "cuda") 
            if self._use_half:
                self._model.half()
            self._is_loaded = True
            print("[UpscaleModel] 模型加载成功,在设备:", self._device)
        except Exception as e:
            print(f"[UpscaleModel] 加载或初始化失败，跳过超分: {e}")
            self._model = None
            self._device = None
            self._use_half = False
            self._is_loaded = True

    def upscale(self, image: Image.Image) -> Image.Image:
        if not self._is_loaded:
            self.load()
        if not SPANDREL_AVAILABLE:
            return image
        if self._model is None or self._device is None:
            return image

        rgb = image.convert("RGB")
        img = np.array(rgb, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self._device)
        tensor = tensor.half() if self._use_half else tensor.float()

        with torch.no_grad():
            out = self._model(tensor)

        out = out.float().clamp(0, 1)
        out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out = (out * 255.0).round().astype(np.uint8)
        return Image.fromarray(out, mode="RGB")

    def unload(self):
        """释放模型资源"""
        self._model = None
        self._device = None
        self._use_half = False
        self._is_loaded = False

# ======================== Icon/Picture处理器 ========================
class IconPictureProcessor(BaseProcessor):
    """
    Icon/Picture处理模块
    
    处理流程：
        1. 从context.elements中筛选需要处理的元素
        2. 根据元素bbox从原图裁剪
        3. 使用RMBG去除背景（icon类）或保留背景（picture类）
        4. 转换为base64编码
        5. 生成XML片段
        6. 更新元素信息
    """
    
    # 需要去背景的类型（抠图）
    RMBG_TYPES = {"icon", "logo", "symbol", "emoji", "button"}
    
    # 保留背景的类型（直接裁剪）
    KEEP_BG_TYPES = {"picture", "photo", "chart", "function_graph", "screenshot", "image", "diagram"}
    
    def __init__(self, config=None, model_path: str = None, device: str = None):
        super().__init__(config)
        self._rmbg_model: Optional[RMBGModel] = None
        self._model_path = model_path
        self._device = device
    
    # def load_model(self):
    #     """加载RMBG模型"""
    #     if self._rmbg_model is None:
    #         self._rmbg_model = RMBGModel(self._model_path, device=self._device)
    #     if not self._rmbg_model.is_loaded:
    #         self._rmbg_model.load()
    
    def process(self, context: ProcessingContext) -> ProcessingResult:
        """
        处理入口
        
        Args:
            context: 处理上下文
            
        Returns:
            ProcessingResult
        """
        self._log("开始处理Icon/Picture元素")
        
        # 加载模型
        # self.load_model()
        
        # 加载原图
        if not context.image_path or not os.path.exists(context.image_path):
            return ProcessingResult(
                success=False,
                error_message="图片路径无效"
            )
        
        original_image = Image.open(context.image_path).convert("RGB")
        cv2_image = cv2.imread(context.image_path)
        img_w, img_h = original_image.size
        
        # 筛选需要处理的元素
        elements_to_process = self._get_elements_to_process(context.elements)
        
        # 过滤掉 bbox 和原图一样大的元素（通常是误检测的背景）
        elements_to_process = self._filter_fullsize_elements(elements_to_process, img_w, img_h)
        
        self._log(f"需要处理的元素数量: {len(elements_to_process)}")
        
        processed_count = 0
        rmbg_count = 0
        keep_bg_count = 0
        
        for elem in elements_to_process:
            try:
                is_rmbg = self._process_element(elem, original_image, cv2_image)
                processed_count += 1
                if is_rmbg:
                    rmbg_count += 1
                else:
                    keep_bg_count += 1
            except Exception as e:
                elem.processing_notes.append(f"处理失败: {str(e)}")
                self._log(f"元素{elem.id}处理失败: {e}")
        
        self._log(f"处理完成: {processed_count}/{len(elements_to_process)}个元素 (抠图:{rmbg_count}, 保留背景:{keep_bg_count})")
        
        return ProcessingResult(
            success=True,
            elements=context.elements,
            canvas_width=context.canvas_width,
            canvas_height=context.canvas_height,
            metadata={
                'processed_count': processed_count,
                'total_to_process': len(elements_to_process),
                'rmbg_count': rmbg_count,
                'keep_bg_count': keep_bg_count
            }
        )
    
    def _get_elements_to_process(self, elements: List[ElementInfo]) -> List[ElementInfo]:
        """筛选需要处理的元素"""
        all_types = self.RMBG_TYPES | self.KEEP_BG_TYPES
        return [
            e for e in elements
            if e.element_type.lower() in all_types and e.base64 is None
        ]
    
    def _filter_fullsize_elements(self, elements: List[ElementInfo], img_w: int, img_h: int, 
                                   threshold: float = 0.95) -> List[ElementInfo]:
        """
        过滤掉 bbox 和原图差不多大的元素（通常是误检测的背景）
        
        Args:
            elements: 元素列表
            img_w: 原图宽度
            img_h: 原图高度
            threshold: 面积比例阈值，超过此比例的元素会被过滤（默认 0.95）
            
        Returns:
            过滤后的元素列表
        """
        filtered = []
        for elem in elements:
            bbox_w = elem.bbox.x2 - elem.bbox.x1
            bbox_h = elem.bbox.y2 - elem.bbox.y1
            bbox_area = bbox_w * bbox_h
            img_area = img_w * img_h
            
            ratio = bbox_area / img_area if img_area > 0 else 0
            
            if ratio >= threshold:
                self._log(f"过滤全图元素: {elem.element_type} (面积比例: {ratio:.2%})")
                continue
            
            filtered.append(elem)
        
        return filtered
    
    def _process_element(self, elem: ElementInfo, original_image: Image.Image, cv2_image: np.ndarray) -> bool:
        """
        处理单个元素
        
        Args:
            elem: 元素信息
            original_image: PIL原图
            cv2_image: OpenCV原图
            
        Returns:
            bool: 是否使用了RMBG抠图
        """
        elem_type = elem.element_type.lower()
        
        # 裁剪区域（带padding）
        # padding = 4
        padding = 10 if elem_type in self.RMBG_TYPES else 0
        img_w, img_h = original_image.size
        x1 = max(0, elem.bbox.x1 - padding)
        y1 = max(0, elem.bbox.y1 - padding)
        x2 = min(img_w, elem.bbox.x2 + padding)
        y2 = min(img_h, elem.bbox.y2 + padding)
        
        cropped = original_image.crop((x1, y1, x2, y2))
        
        is_rmbg = False
        
        # 根据类型决定是否去背景
        if elem_type in self.RMBG_TYPES:
            # 去除背景
            print("在process_element 调用了rmbg==去除背景-")
            processed = rmbg_request.call_rmbg_service(cropped)
            print("在process_element 调用了rmbg")
            # self._rmbg_model.remove_background(cropped) 
            elem.has_transparency = True
            is_rmbg = True
        else:
            # 保留背景
            processed = cropped.convert("RGBA")
            elem.has_transparency = False
        
        # 转base64
        elem.base64 = self._image_to_base64(processed)
        
        # 更新bbox（因为加了padding）
        elem.bbox.x1 = x1
        elem.bbox.y1 = y1
        elem.bbox.x2 = x2
        elem.bbox.y2 = y2
        
        # 生成XML片段
        self._generate_xml(elem)
        
        elem.processing_notes.append(f"IconPictureProcessor处理完成 (RMBG={is_rmbg})")
        
        return is_rmbg
    
    def _generate_xml(self, elem: ElementInfo):
        """
        生成图片元素的XML片段
        """
        x1 = elem.bbox.x1
        y1 = elem.bbox.y1
        width = elem.bbox.x2 - elem.bbox.x1
        height = elem.bbox.y2 - elem.bbox.y1
        
        # DrawIO 图片样式
        style = (
            "shape=image;verticalLabelPosition=bottom;verticalAlign=top;"
            "imageAspect=0;aspect=fixed;"
            f"image=data:image/png,{elem.base64};"
        )
        
        # DrawIO的id必须从2开始（0和1是保留的根元素）
        cell_id = elem.id + 2
        
        elem.xml_fragment = f'''<mxCell id="{cell_id}" parent="1" vertex="1" value="" style="{style}">
  <mxGeometry x="{x1}" y="{y1}" width="{width}" height="{height}" as="geometry"/>
</mxCell>'''
        
        # 设置层级
        elem.layer_level = LayerLevel.IMAGE.value
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """将PIL图像转换为base64"""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ======================== 图像复杂度分析 ========================
def calculate_image_complexity(image_arr: np.ndarray) -> tuple:
    """
    计算图像丰富度/复杂度
    用于自动判断元素是否应该作为图片处理
    
    Args:
        image_arr: BGR格式的numpy数组
        
    Returns:
        (laplacian_variance, std_deviation)
        - laplacian_variance: 拉普拉斯方差（纹理/边缘丰富度）
        - std_deviation: 标准差（对比度/颜色变化）
    """
    if image_arr.size == 0:
        return 0.0, 0.0
    
    gray = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
    
    # 拉普拉斯方差 (纹理/边缘丰富度)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 标准差 (对比度/颜色变化)
    std_dev = np.std(gray)
    
    return laplacian_var, std_dev


def is_complex_image(image_arr: np.ndarray, laplacian_threshold: float = 800, std_threshold: float = 50) -> bool:
    """
    判断图像是否为复杂图像（应作为picture处理）
    
    Args:
        image_arr: BGR格式的numpy数组
        laplacian_threshold: 拉普拉斯方差阈值
        std_threshold: 标准差阈值
        
    Returns:
        bool: 是否为复杂图像
    """
    l_var, s_dev = calculate_image_complexity(image_arr)
    return l_var > laplacian_threshold or s_dev > std_threshold


# ======================== 快捷函数 ========================
def process_icons_pictures(elements: List[ElementInfo], 
                           image_path: str) -> List[ElementInfo]:
    """
    快捷函数 - 处理所有icon和picture元素
    
    Args:
        elements: 元素列表
        image_path: 原始图片路径
        
    Returns:
        处理后的元素列表
        
    使用示例:
        elements = process_icons_pictures(elements, "test.png")
    """
    processor = IconPictureProcessor()
    context = ProcessingContext(
        image_path=image_path,
        elements=elements
    )
    
    result = processor.process(context)
    return result.elements