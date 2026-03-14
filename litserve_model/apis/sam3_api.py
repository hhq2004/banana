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

# modules/sam3_info_extractor.py
# 引用这里的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modules.sam3_info_extractor import SAM3Model

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
REQ_LOG = LOG_DIR / "litserve_requests.log"


def _log(service: str, message: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    line = f"[{ts}][pid={os.getpid()}][{service}] {message}\n"
    with REQ_LOG.open("a", encoding="utf-8") as f:
        f.write(line)

from config.read_config import load_config

class Sam3LitAPI(ls.LitAPI):
    def setup(self, device):
        config = load_config()
        sam3_config = config.get('sam3', {})
        self.device = device
        _log("sam3", f"setup device={device}")
        self.model = SAM3Model(
            checkpoint_path=sam3_config.get('checkpoint_path'),
            bpe_path=sam3_config.get('bpe_path_litserve'),
            device=device
        )
        self.model.load()

    def decode_request(self, request, **kwargs):
        # 解析传入的JSON请求
        # 假设请求包含图像路径和提示词列表
        # 处理request可能为字符串的情况
        if isinstance(request, str):
            request = json.loads(request)
        
        image_path = request.get("image_path", "/home/youzirui/code_dir_yzr/IMG2XML/test_fig/test01.png")
        prompts = request.get("prompts", [])
        score_threshold = request.get("score_threshold", 0.5)
        min_area = request.get("min_area", 100)
        
        # 如果没有提供prompts，则从配置文件获取默认值
        if not prompts:
            config = load_config()
            prompts = config.get('prompt_groups', {}).get('image', {}).get('prompts', [])
        
        return {"image_path": image_path, "prompts": prompts, "score_threshold": score_threshold, "min_area": min_area}
        
    def predict(self, x, **kwargs):
        self.model.clear_cache()

        if isinstance(x, dict):
            x = [x]
        elif isinstance(x, str):
            x = [json.loads(x)]

        results = []
        _log("sam3", f"predict batch={len(x)} device={getattr(self, 'device', 'unknown')}")
        for idx, item in enumerate(x):  # x is a list of requests
            image_path = item["image_path"]
            prompts =  item["prompts"]
            score_threshold = item.get("score_threshold", 0.5)
            min_area = item.get("min_area", 100)

            start = time.time()
            result = self.model.predict(image_path=image_path, prompts=prompts, score_threshold=score_threshold, min_area=min_area)
            elapsed = round(time.time() - start, 3)

            _log("sam3", f"worker_pid={os.getpid()} idx={idx} image={Path(image_path).name} prompts={len(prompts)} elapsed={elapsed}s result_len={len(result)}")

            # 将结果序列化为字节，然后编码为base64
            import pickle
            import base64
            serialized_result = pickle.dumps(result)
            base64_encoded = base64.b64encode(serialized_result).decode('utf-8')

            results.append(base64_encoded)
        
        return results

    def encode_response(self, output, **kwargs):
        return {"results_base64": output}