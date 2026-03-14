"""Command-line utility to summarize litserve request metrics logs."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from config.read_config import load_config

DEFAULT_LOG_FILE = Path(__file__).resolve().parents[1] / "logs" / "request_metrics.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze litserve request metrics.")
    parser.add_argument(
        "--log-file",
        default=str(DEFAULT_LOG_FILE),
        help="Path to the request metrics log file.",
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="Path to config.yaml (default: auto-detect via CONFIG_PATH env).",
    )
    return parser.parse_args()


def load_entries(log_file: str) -> List[Dict[str, Any]]:
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file not found: {log_file}")

    entries: List[Dict[str, Any]] = []
    with open(log_file, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def aggregate(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "total_ms": 0.0,
            "min_ms": None,
            "max_ms": 0.0,
            "success": 0,
            "errors": 0,
        }
    )

    for entry in entries:
        module = entry.get("module", "unknown")
        duration = float(entry.get("duration_ms", 0.0))
        status = entry.get("status", "success")

        module_stats = stats[module]
        module_stats["count"] += 1
        module_stats["total_ms"] += duration
        module_stats["min_ms"] = duration if module_stats["min_ms"] is None else min(module_stats["min_ms"], duration)
        module_stats["max_ms"] = max(module_stats["max_ms"], duration)
        if status == "success":
            module_stats["success"] += 1
        else:
            module_stats["errors"] += 1

    for module_stats in stats.values():
        count = module_stats["count"] or 1
        module_stats["avg_ms"] = module_stats["total_ms"] / count
        module_stats["success_rate"] = module_stats["success"] / count
    return stats


def load_litserve_config(config_file: str | None = None) -> Dict[str, Any]:
    config = load_config(config_file)
    return config.get("services", {}).get("litserve", {})


def format_duration(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def print_report(stats: Dict[str, Dict[str, Any]], litserve_cfg: Dict[str, Any]) -> None:
    print("LitServe configuration:")
    print(json.dumps(litserve_cfg, indent=2, ensure_ascii=False))
    print()

    if not stats:
        print("No request metrics found.")
        return

    header = (
        "Module",
        "Count",
        "Success",
        "Errors",
        "Success %",
        "Avg ms",
        "Min ms",
        "Max ms",
        "Total ms",
    )
    print(" | ".join(header))
    print("-" * 80)

    for module, module_stats in sorted(stats.items()):
        row = (
            module,
            str(module_stats["count"]),
            str(module_stats["success"]),
            str(module_stats["errors"]),
            f"{module_stats['success_rate'] * 100:.1f}",
            f"{module_stats['avg_ms']:.2f}",
            format_duration(module_stats["min_ms"]),
            format_duration(module_stats["max_ms"]),
            f"{module_stats['total_ms']:.2f}",
        )
        print(" | ".join(row))


def main() -> None:
    args = parse_args()

    try:
        entries = load_entries(args.log_file)
    except FileNotFoundError as exc:
        print(exc)
        return

    stats = aggregate(entries)

    try:
        litserve_cfg = load_litserve_config(args.config_file)
    except FileNotFoundError as exc:
        print(exc)
        litserve_cfg = {}

    print_report(stats, litserve_cfg)


if __name__ == "__main__":
    main()
