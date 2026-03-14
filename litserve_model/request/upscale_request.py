import base64
import io
import time

import requests
from PIL import Image
from typing import Optional

try:
    from .request_logging import log_request_metrics
    from .service_url import build_litserve_predict_url
except ImportError:  # When executed as a script
    from request_logging import log_request_metrics
    from service_url import build_litserve_predict_url

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

def call_upscale_service(
    image,
        url=None,
):
    if not url:
        url = build_litserve_predict_url("upscale")

    payload = {
        "image": image_to_base64(image)
    }
    start_ts = time.perf_counter()
    status = "success"
    error_details = None
    try:
        resp = requests.post(url, json=payload)
        resp.raise_for_status()

        data = resp.json()
        out_img = base64_to_image(data["results"][0], target_mode="RGBA")
        return out_img
    except Exception as exc:
        status = "error"
        error_details = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        raise
    finally:
        duration_ms = (time.perf_counter() - start_ts) * 1000
        metadata = {"url": url}
        if error_details:
            metadata.update(error_details)
        log_request_metrics(
            module="upscale_request",
            duration_ms=duration_ms,
            status=status,
            metadata=metadata,
        )

if __name__ == "__main__":
    img = Image.open("/home/youzirui/code_dir_yzr/ori_img/IMG2XML_litserve/input/test05.png")


    results = call_upscale_service(img)
    
    results.save("/home/youzirui/code_dir_yzr/ori_img/IMG2XML_litserve/output/test05_upscale.png")