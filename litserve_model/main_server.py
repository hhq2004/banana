"""One-stop launcher for SAM3 / RMBG / Upscale LitServe processes.

Features
- Single command starts multiple LitServe instances on different ports.
- Per-service log files under logs/litserve_*.log and live stdout.
- Select which services to start via CLI flags.
"""

import os

# 减少 CUDA 显存碎片与峰值分配失败：
# 在导入任何可能使用 torch/cuda 的模块之前设置。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import multiprocessing as mp
import signal
import sys
from pathlib import Path
from typing import Tuple

import litserve as ls

from apis.sam3_api import Sam3LitAPI
from apis.rmbg_api import RMBGLitAPI
from apis.upscale_api import UpscaleLitAPI
from config.read_config import load_config


BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _litserve_cfg(cfg: dict) -> dict:
	return cfg.get("services", {}).get("litserve", {})


def _resolve_host_port(name: str, cfg: dict) -> Tuple[str, int]:
	c = _litserve_cfg(cfg)
	host = c.get("bind_host", c.get("host", "127.0.0.1"))
	if name == "sam3":
		return host, c.get("port_sam3", c.get("port", 8022))
	if name == "rmbg":
		return host, c.get("port_rmbg", 8023)
	if name == "upscale":
		return host, c.get("port_upscale", 8024)
	raise ValueError(name)


def _device(cfg: dict, name: str, fallback: str = "auto") -> str:
	c = _litserve_cfg(cfg)
	return c.get(f"devices_{name}", c.get("devices", fallback))


def _workers(cfg: dict, name: str, fallback: int = 2) -> int:
	c = _litserve_cfg(cfg)
	return int(c.get(f"workers_per_device_{name}", c.get("workers_per_device", fallback)))


def _timeout(cfg: dict, name: str, fallback: int = 600) -> int:
	c = _litserve_cfg(cfg)
	return int(c.get(f"timeout_{name}", c.get("timeout", fallback)))


def _batch(cfg: dict, name: str, fallback: int = 4) -> int:
	c = _litserve_cfg(cfg)
	return int(c.get(f"max_batch_size_{name}", c.get("max_batch_size", fallback)))


def run_server(name: str, cfg: dict, log_file: Path):
	def log(msg: str):
		line = f"[{name}] {msg}"
		print(line, flush=True)
		with log_file.open("a", encoding="utf-8") as f:
			f.write(line + "\n")

	if name == "sam3":
		api = Sam3LitAPI(max_batch_size=_batch(cfg, "sam3"))
		server = ls.LitServer(
			lit_api=[api],
			accelerator="cuda",
			devices=_device(cfg, "sam3"),
			workers_per_device=_workers(cfg, "sam3"),
			timeout=_timeout(cfg, "sam3"),
		)
	elif name == "rmbg":
		api = RMBGLitAPI(max_batch_size=_batch(cfg, "rmbg"))
		server = ls.LitServer(
			lit_api=[api],
			accelerator="cuda",
			devices=_device(cfg, "rmbg"),
			workers_per_device=_workers(cfg, "rmbg"),
			timeout=_timeout(cfg, "rmbg"),
		)
	elif name == "upscale":
		api = UpscaleLitAPI(max_batch_size=_batch(cfg, "upscale"))
		server = ls.LitServer(
			lit_api=[api],
			accelerator="cuda",
			devices=_device(cfg, "upscale"),
			workers_per_device=_workers(cfg, "upscale"),
			timeout=_timeout(cfg, "upscale"),
		)
	else:
		raise ValueError(f"Unknown service: {name}")

	host, port = _resolve_host_port(name, cfg)
	log(f"starting on {host}:{port}")
	server.run(host=host, port=port)


def start_process(name: str, cfg: dict) -> mp.Process:
	log_file = LOG_DIR / f"litserve_{name}.log"
	p = mp.Process(target=run_server, args=(name, cfg, log_file), daemon=False)
	p.start()
	return p


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Launch multiple LitServe services")
	parser.add_argument("--config", default=None, help="Path to config.yaml (default: auto-detect via CONFIG_PATH env)")
	parser.add_argument(
		"--services",
		nargs="+",
		choices=["sam3", "rmbg", "upscale", "all"],
		default=["all"],
		help="Which services to start",
	)
	return parser.parse_args()


def main():
	args = parse_args()
	cfg = load_config(args.config)

	services = args.services
	if "all" in services:
		services = ["sam3", "rmbg", "upscale"]

	procs = []
	for name in services:
		p = start_process(name, cfg)
		procs.append(p)
		print(f"[main] started {name} pid={p.pid}")

	def shutdown(sig, frame):
		print("[main] shutting down...", flush=True)
		for p in procs:
			if p.is_alive():
				p.terminate()
		sys.exit(0)

	signal.signal(signal.SIGINT, shutdown)
	signal.signal(signal.SIGTERM, shutdown)

	for p in procs:
		p.join()


if __name__ == "__main__":
	main()
