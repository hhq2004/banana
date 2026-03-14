import base64
import json
import os
import sys
import time

import cv2
import litserve as ls
import numpy as np

import hashlib
from pathlib import Path
from typing import Optional

"""
python litserve_model/main_server.py --services rmbg
"""


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modules.icon_picture_processor import RMBGModel

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
REQ_LOG = LOG_DIR / "litserve_requests.log"


def _log(service: str, message: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    line = f"[{ts}][pid={os.getpid()}][{service}] {message}\n"
    with REQ_LOG.open("a", encoding="utf-8") as f:
        f.write(line)

from config.read_config import load_config
    
# 图像与Base64互转!    
from PIL import Image
import io
def image_to_base64(img: Image.Image, format="PNG") -> str:
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")

def base64_to_image(b64_str: str, target_mode: Optional[str] = None) -> Image.Image:
    """Decode a base64 string back into an image, preserving mode when requested."""
    img_bytes = base64.b64decode(b64_str)
    buffer = io.BytesIO(img_bytes)
    with Image.open(buffer) as img:
        restored = img.copy()

    if target_mode and restored.mode != target_mode:
        restored = restored.convert(target_mode)

    return restored

class RMBGLitAPI(ls.LitAPI):
    def setup(self, device):
        config = load_config()
        # sam3_config = config.get('sam3', {})
        self.device = device
        _log("rmbg", f"setup device={self.device}")
        self.model = RMBGModel(device=self.device)
        self.model.load()

    def decode_request(self, request, **kwargs):
        # 解析传入的JSON请求

        if isinstance(request, str):
            request = json.loads(request)        

        # 还原为Image.Image对象
        image_b64 = request.get("image")
        if image_b64 is None:
            raise ValueError("Missing 'image' field in request")

        image = base64_to_image(image_b64, target_mode="RGBA")

        return {"image": image}
        
    def predict(self, x, **kwargs):        
        if isinstance(x, dict):
            x = [x]
        elif isinstance(x, str):
            x = [json.loads(x)]

        _log("rmbg", f"predict batch={len(x)} device={getattr(self, 'device', 'unknown')}")
        results = []
        for idx, item in enumerate(x):  # x is a list of requests
            image = item["image"]

            start = time.time()
            result = self.model.predict(image=image)
            elapsed = round(time.time() - start, 3)

            _log("rmbg", f"worker_pid={os.getpid()} idx={idx} elapsed={elapsed}s result_type={type(result).__name__}")

            results.append(result)
            #self.model._session = None 


        return results

    def encode_response(self, output, **kwargs):
        # 统一成 list 处理
        if isinstance(output, Image.Image):
            output = [output]

        encoded_results = []

        for item in output:
            if isinstance(item, Image.Image):
                encoded_results.append(image_to_base64(item))

            elif isinstance(item, list):
                encoded_results.append([
                    image_to_base64(img) for img in item
                ])

            else:
                # 非图片结果，原样返回
                encoded_results.append(item)

        return {"results": encoded_results}


