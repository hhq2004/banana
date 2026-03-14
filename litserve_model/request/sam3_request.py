import base64
import binascii
import json
import pickle
import time

import requests

try:
    from .request_logging import log_request_metrics
    from .service_url import build_litserve_predict_url
except ImportError:  # When executed as a script
    from request_logging import log_request_metrics
    from service_url import build_litserve_predict_url


def _coerce_results_list(results_base64):
    """Normalize the litserve response field to a list for batch handling."""
    if results_base64 is None:
        return []
    if isinstance(results_base64, list):
        return results_base64
    if isinstance(results_base64, str):
        stripped = results_base64.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(results_base64)
            except json.JSONDecodeError:
                return [results_base64]
            if isinstance(parsed, list):
                return parsed
        return [results_base64]
    raise TypeError(f"Unsupported results_base64 type: {type(results_base64).__name__}")


def _decode_entry(encoded_entry, idx):
    if isinstance(encoded_entry, list):
        return [_decode_entry(item, idx) for item in encoded_entry]
    if not isinstance(encoded_entry, str):
        raise TypeError(f"Expected base64 string at index {idx}, got {type(encoded_entry).__name__}")
    try:
        decoded_bytes = base64.b64decode(encoded_entry)
    except binascii.Error as exc:
        raise ValueError(f"Invalid base64 payload at index {idx}") from exc
    return pickle.loads(decoded_bytes)

def call_sam3_service(
    image_path,
    prompts,
    score_threshold=0.5,
    min_area=100,
    url=None,
):
    if not url:
        url = build_litserve_predict_url("sam3")

    payload = {
        "image_path": image_path,
        "prompts": prompts,
        "score_threshold": score_threshold,
        "min_area": min_area
    }
    print(f"[SAM3_REQ] url={url} image={image_path} prompts={prompts} thr={score_threshold} min_area={min_area}", flush=True)
    start_ts = time.perf_counter()
    status = "success"
    error_details = None
    try:
        resp = requests.post(url, json=payload)
        print(f"[SAM3_REQ] response status={resp.status_code} len={len(resp.content)}", flush=True)
        resp.raise_for_status()

        data = resp.json()
        print(f"[SAM3_REQ] response keys={list(data.keys())}", flush=True)
        normalized_results = _coerce_results_list(data.get("results_base64"))
        print(f"[SAM3_REQ] normalized_results count={len(normalized_results)}", flush=True)

        decoded_results = []
        for idx, result_base64 in enumerate(normalized_results):
            decoded = _decode_entry(result_base64, idx)
            print(f"[SAM3_REQ] decoded[{idx}] type={type(decoded).__name__} len={len(decoded) if isinstance(decoded, (list, tuple)) else 'N/A'}", flush=True)
            decoded_results.append(decoded)
        
        # 如果只有一个结果，直接返回该结果
        if len(decoded_results) == 1:
            return decoded_results[0]
        
        return decoded_results
    except Exception as exc:
        status = "error"
        error_details = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        print(f"[SAM3_REQ] ERROR: {type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        duration_ms = (time.perf_counter() - start_ts) * 1000
        metadata = {"url": url}
        if error_details:
            metadata.update(error_details)
        log_request_metrics(
            module="sam3_request",
            duration_ms=duration_ms,
            status=status,
            metadata=metadata,
        )

if __name__ == "__main__":
    image_path = "/home/youzirui/code_dir_yzr/ori_img/IMG2XML_litserve/test_fig/test01.png"
    prompts = [
        
    ]

    results = call_sam3_service(image_path, prompts)
    print(len(results))