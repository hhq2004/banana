import argparse
import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> int:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8:backslashreplace"
    env.setdefault("LANG", "C.UTF-8")
    env.setdefault("LC_ALL", "C.UTF-8")
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="backslashreplace",
        bufsize=1,
        env=env,
    )
    assert p.stdout is not None
    try:
        for line in p.stdout:
            print(line, end="")
    finally:
        p.stdout.close()
    return p.wait()


def _parse_int_list(v: str) -> list[int]:
    s = (v or "").strip()
    if not s:
        return []
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: Azure OCR -> mask generation -> LaMa inpaint, with grid search over mask/inpaint expansion params."
    )
    parser.add_argument("--endpoint", default="http://localhost:5000", help="Azure OCR container endpoint")
    parser.add_argument("--input", required=True, help="Input image directory or single image file")
    parser.add_argument(
        "--out-root",
        default="",
        help="Optional unified output root; if set, outputs will be placed into '<out-root>/ocr' and '<out-root>/inpaint'",
    )
    parser.add_argument("--ocr-out", required=False, default="", help="Base output dir for OCR/masks")
    parser.add_argument("--inpaint-out", required=False, default="", help="Base output dir for inpaint results")

    parser.add_argument("--mask-mode", choices=["line", "word", "pixel"], default="word")
    parser.add_argument("--mask-dilate-grid", default="2", help="Comma-separated mask-dilate values")
    parser.add_argument("--mask-dilate-adaptive", action="store_true", help="Enable adaptive dilate per OCR block (requires updated run_azure_ocr_demo.py)")
    parser.add_argument("--mask-dilate-adaptive-mode", choices=["font", "bbox", "max"], default="font")
    parser.add_argument("--mask-dilate-min", type=int, default=1)
    parser.add_argument("--mask-dilate-max", type=int, default=6)
    parser.add_argument("--mask-blur", type=float, default=2.0, help="Gaussian blur radius for mask edges (default: 2.0)")
    parser.add_argument("--pixel-delta", type=int, default=18)
    parser.add_argument("--pixel-edge-aware", action="store_true")
    parser.add_argument("--pixel-edge-percentile", type=float, default=97.0)
    parser.add_argument("--pixel-edge-near", type=float, default=5.0)
    parser.add_argument("--pixel-edge-mid", type=float, default=10.0)
    parser.add_argument("--pixel-edge-exclude-dilate", type=int, default=6)
    parser.add_argument("--pixel-edge-min-area", type=int, default=200)
    parser.add_argument("--pixel-edge-edge-dilate", type=int, default=1)
    parser.add_argument("--pixel-edge-save-edge", action="store_true")
    parser.add_argument("--pixel-edge-grad-mode", choices=["gray", "rgb"], default="gray")
    parser.add_argument("--pixel-edge-metric", choices=["euclidean", "chessboard", "taxicab"], default="chessboard")
    parser.add_argument("--pixel-edge-stat", choices=["min", "p10"], default="min")
    parser.add_argument("--pixel-edge-debug", action="store_true")

    parser.add_argument(
        "--erase-sam3",
        action="store_true",
        help="Union SAM3 (arrow/icon/etc) detection masks into the final mask (OCR stage)",
    )
    parser.add_argument(
        "--sam3-prompts",
        default="",
        help="Comma-separated SAM3 prompts to erase (passed to OCR stage). Empty uses default set.",
    )
    parser.add_argument("--sam3-score-threshold", type=float, default=0.45)
    parser.add_argument("--sam3-min-area", type=int, default=50)
    parser.add_argument("--sam3-dilate", type=int, default=2)
    parser.add_argument(
        "--sam3-pythonpath",
        default="",
        help="Optional extra PYTHONPATH (colon-separated) to locate litserve_model during OCR stage (requires --erase-sam3)",
    )
    parser.add_argument(
        "--sam3-debug",
        action="store_true",
        help="Print SAM3 detection summary (prompt/score/bbox) during OCR stage (requires --erase-sam3)",
    )
    parser.add_argument(
        "--sam3-save-json",
        action="store_true",
        help="Save SAM3 raw detections as *_sam3_raw.json during OCR stage (requires --erase-sam3)",
    )
    parser.add_argument("--pixel-delta-small", type=int, default=48)
    parser.add_argument("--pixel-delta-mid", type=int, default=42)
    parser.add_argument("--mask-dilate-small", type=int, default=1)
    parser.add_argument("--mask-blur-small", type=float, default=1.0)
    parser.add_argument("--mask-dilate-mid", type=int, default=2)
    parser.add_argument("--mask-blur-mid", type=float, default=2.0)

    parser.add_argument("--expand-mask-grid", default="10", help="Comma-separated expand-mask values for inpaint (e.g. 6,8,10,12)")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N images (0 = no limit)")
    parser.add_argument("--adaptive", action="store_true", help="Enable adaptive expand-mask logic in inpaint")
    parser.add_argument("--auto-expand", action="store_true", help="Enable inpaint auto-expand (requires --adaptive)")
    parser.add_argument("--auto-expand-max", type=int, default=30)
    parser.add_argument("--auto-expand-delta", type=float, default=6.0)
    parser.add_argument("--auto-expand-min", type=int, default=1)
    parser.add_argument("--adaptive-edge-percentile", type=float, default=97.0)
    parser.add_argument("--adaptive-near-edge-px", type=float, default=2.0)
    parser.add_argument("--adaptive-far-edge-px", type=float, default=10.0)
    parser.add_argument("--adaptive-min-expand", type=int, default=1)
    parser.add_argument("--debug-vis", action="store_true", help="Pass through to inpaint debug visualization")
    parser.add_argument("--debug-vis-dir", default="", help="Directory for debug visualizations")

    parser.add_argument("--postprocess", action="store_true", help="Pass through to inpaint postprocess (scan rings + optional selective reinpaint)")
    parser.add_argument("--post-delta", type=float, default=6.0)
    parser.add_argument("--post-min-inner", type=int, default=2)
    parser.add_argument("--post-max-inner", type=int, default=30)
    parser.add_argument("--post-margin", type=int, default=64)
    parser.add_argument("--post-dilate", type=int, default=2)
    parser.add_argument("--post-dry-run", action="store_true")
    parser.add_argument("--post-debug-dir", default="")

    parser.add_argument("--post-noise-percentile", type=float, default=99.5)
    parser.add_argument("--post-noise-mode", choices=["auto", "resid_gray", "lab_outlier", "lab_resid", "lab_kmeans"], default="lab_outlier")
    parser.add_argument("--post-noise-sigma", type=float, default=6.0)
    parser.add_argument("--post-noise-edge", type=int, default=4)
    parser.add_argument("--post-noise-flat", type=float, default=8.0)
    parser.add_argument("--post-noise-kmeans-sep", type=float, default=8.0)
    parser.add_argument("--post-noise-kmeans-max-frac", type=float, default=0.25)
    parser.add_argument("--post-noise-z", type=float, default=6.0)
    parser.add_argument("--post-noise-min-area", type=int, default=6)
    parser.add_argument("--post-noise-vis-dir", default="")
    parser.add_argument("--post-no-noise-vis", action="store_true", help="Disable saving post noise visualization")
    parser.add_argument("--save-masked", action="store_true", help="Save the masked image before inpainting")

    parser.add_argument("--device", default="gpu", help="Device label passed to inpaint script")
    parser.add_argument(
        "--backend",
        default="simple_lama",
        choices=["simple_lama", "flux_fill_api", "flux_fill_local", "sdxl_inpaint_local"],
        help="Inpaint backend (simple_lama, flux_fill_api, flux_fill_local, sdxl_inpaint_local)",
    )
    parser.add_argument("--flux-prompt", default="", help="Prompt for flux_fill_api")
    parser.add_argument("--flux-steps", type=int, default=50, help="Steps for flux_fill_api")
    parser.add_argument("--flux-guidance", type=float, default=1.5, help="Guidance for flux_fill_api")
    parser.add_argument("--flux-safety-tolerance", type=int, default=2)
    parser.add_argument("--flux-output-format", default="png")
    parser.add_argument("--flux-endpoint", default="https://api.bfl.ai/v1/flux-pro-1.0-fill")
    parser.add_argument("--flux-poll-interval", type=float, default=0.5)
    parser.add_argument("--flux-poll-timeout", type=float, default=300.0)

    parser.add_argument(
        "--flux-model-dir",
        default="/home/wangyankun/models/flux_fill",
        help="Local FLUX.1-Fill-dev model directory for flux_fill_local backend",
    )
    parser.add_argument(
        "--flux-lora-path",
        default="/home/wangyankun/models/loras/removalV2.safetensors",
        help="Local LoRA .safetensors path for flux_fill_local backend",
    )
    parser.add_argument(
        "--flux-lora-scale",
        type=float,
        default=0.9,
        help="LoRA scale for flux_fill_local backend (default: 0.9)",
    )

    parser.add_argument(
        "--sdxl-model",
        default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        help="SDXL inpainting model id or local directory for sdxl_inpaint_local backend",
    )
    parser.add_argument(
        "--sdxl-negative-prompt",
        default="",
        help="Negative prompt for sdxl_inpaint_local backend",
    )
    parser.add_argument(
        "--sdxl-steps",
        type=int,
        default=30,
        help="Steps for sdxl_inpaint_local backend (default: 30)",
    )
    parser.add_argument(
        "--sdxl-guidance",
        type=float,
        default=6.0,
        help="Guidance scale for sdxl_inpaint_local backend (default: 6.0)",
    )
    parser.add_argument(
        "--sdxl-strength",
        type=float,
        default=0.65,
        help="Strength for sdxl_inpaint_local backend (default: 0.65). Lower reduces repainting.",
    )

    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Pass through to inpaint script to save debug assets into '<out_dir>/_debug/<stem>/'",
    )

    parser.add_argument("--skip-existing", action="store_true", help="Skip a run if its output folder already exists and is non-empty")
    parser.add_argument("--skip-missing-mask", action="store_true", help="Pass through to inpaint script")
    parser.add_argument(
        "--skip-ocr-if-exists",
        action="store_true",
        default=True,
        help="Skip OCR stage if masks already exist in the target mask dir (default: true)",
    )
    parser.add_argument(
        "--no-skip-ocr-if-exists",
        action="store_false",
        dest="skip_ocr_if_exists",
        help="Always run OCR stage even if masks exist",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"input not found: {input_path}", file=sys.stderr)
        return 2

    out_root_s = str(args.out_root or "").strip()
    ocr_out_s = str(args.ocr_out or "").strip()
    inpaint_out_s = str(args.inpaint_out or "").strip()
    if out_root_s:
        root = Path(out_root_s)
        if not ocr_out_s:
            ocr_out_s = str(root / "ocr")
        if not inpaint_out_s:
            inpaint_out_s = str(root / "inpaint")
    if not ocr_out_s or not inpaint_out_s:
        print("[FAIL] Must provide either --out-root or both --ocr-out and --inpaint-out", file=sys.stderr)
        return 2

    ocr_out = Path(ocr_out_s)
    inpaint_out = Path(inpaint_out_s)
    ocr_out.mkdir(parents=True, exist_ok=True)
    inpaint_out.mkdir(parents=True, exist_ok=True)

    mask_dilates = _parse_int_list(args.mask_dilate_grid)
    expand_masks = _parse_int_list(args.expand_mask_grid)
    if not mask_dilates:
        print("mask-dilate-grid is empty", file=sys.stderr)
        return 3
    if not expand_masks:
        print("expand-mask-grid is empty", file=sys.stderr)
        return 4

    base_dir = Path(__file__).parent
    ocr_py = base_dir / "run_azure_ocr_demo.py"
    inpaint_py = base_dir / "run_gen_inpaint_demo.py"

    if not ocr_py.exists():
        print(f"missing: {ocr_py}", file=sys.stderr)
        return 5
    if not inpaint_py.exists():
        print(f"missing: {inpaint_py}", file=sys.stderr)
        return 6

    for md in mask_dilates:
        name = f"mask-{args.mask_mode}_d{md}_b{args.mask_blur:g}_pd{args.pixel_delta}"
        if args.mask_dilate_adaptive:
            name += (
                f"_ad-{args.mask_dilate_adaptive_mode}"
                f"_{args.mask_dilate_min}-{args.mask_dilate_max}"
                f"_fr{args.mask_dilate_font_ref:g}"
                f"_br{args.mask_dilate_bbox_ratio:g}"
            )
        mask_dir = ocr_out / name
        mask_dir.mkdir(parents=True, exist_ok=True)

        has_any_mask = any(mask_dir.glob("*_mask.png"))
        if (args.skip_existing and any(mask_dir.iterdir())) or (args.skip_ocr_if_exists and has_any_mask):
            print(f"[SKIP] OCR mask exists: {mask_dir}")
        else:
            cmd_ocr = [
                sys.executable,
                str(ocr_py),
                "--endpoint",
                str(args.endpoint),
                "--input",
                str(input_path),
                "--output",
                str(mask_dir),
                "--mask-mode",
                str(args.mask_mode),
                "--save-mask",
                "--pixel-delta",
                str(args.pixel_delta),
                "--mask-dilate",
                str(md),
                "--mask-blur",
                str(args.mask_blur),
                "--limit",
                str(int(args.limit or 0)),
            ]
            if args.erase_sam3:
                cmd_ocr += [
                    "--erase-sam3",
                    "--sam3-prompts",
                    str(args.sam3_prompts or ""),
                    "--sam3-score-threshold",
                    str(args.sam3_score_threshold),
                    "--sam3-min-area",
                    str(args.sam3_min_area),
                    "--sam3-dilate",
                    str(args.sam3_dilate),
                    "--sam3-pythonpath",
                    str(args.sam3_pythonpath or ""),
                ]
                if args.sam3_debug:
                    cmd_ocr.append("--sam3-debug")
                if args.sam3_save_json:
                    cmd_ocr.append("--sam3-save-json")
            if args.pixel_edge_aware and args.mask_mode == "pixel":
                cmd_ocr += [
                    "--pixel-edge-aware",
                    "--pixel-edge-percentile",
                    str(args.pixel_edge_percentile),
                    "--pixel-edge-near",
                    str(args.pixel_edge_near),
                    "--pixel-edge-mid",
                    str(args.pixel_edge_mid),
                    "--pixel-edge-exclude-dilate",
                    str(args.pixel_edge_exclude_dilate),
                    "--pixel-edge-min-area",
                    str(args.pixel_edge_min_area),
                    "--pixel-edge-edge-dilate",
                    str(args.pixel_edge_edge_dilate),
                    "--pixel-edge-save-edge",
                    "--pixel-edge-grad-mode",
                    str(args.pixel_edge_grad_mode),
                    "--pixel-edge-metric",
                    str(args.pixel_edge_metric),
                    "--pixel-edge-stat",
                    str(args.pixel_edge_stat),
                    "--pixel-delta-small",
                    str(args.pixel_delta_small),
                    "--pixel-delta-mid",
                    str(args.pixel_delta_mid),
                    "--mask-dilate-small",
                    str(args.mask_dilate_small),
                    "--mask-blur-small",
                    str(args.mask_blur_small),
                    "--mask-dilate-mid",
                    str(args.mask_dilate_mid),
                    "--mask-blur-mid",
                    str(args.mask_blur_mid),
                ]
                if args.pixel_edge_debug:
                    cmd_ocr.append("--pixel-edge-debug")
            if args.mask_dilate_adaptive:
                cmd_ocr += [
                    "--mask-dilate-adaptive",
                    "--mask-dilate-adaptive-mode",
                    str(args.mask_dilate_adaptive_mode),
                    "--mask-dilate-min",
                    str(args.mask_dilate_min),
                    "--mask-dilate-max",
                    str(args.mask_dilate_max),
                    "--mask-dilate-font-ref",
                    str(args.mask_dilate_font_ref),
                    "--mask-dilate-bbox-ratio",
                    str(args.mask_dilate_bbox_ratio),
                ]
            print(f"\n=== OCR -> mask: mask_dilate={md} -> {mask_dir} ===")
            code = _run(cmd_ocr)
            if code != 0:
                print(f"[FAIL] OCR stage failed: exit={code}", file=sys.stderr)
                return 10

        for em in expand_masks:
            out_name = f"md{md}_em{em}" + ("_adaptive" if args.adaptive else "")
            out_dir = inpaint_out / out_name
            out_dir.mkdir(parents=True, exist_ok=True)

            if args.skip_existing and any(out_dir.iterdir()):
                print(f"[SKIP] inpaint exists: {out_dir}")
                continue

            cmd_inpaint = [
                sys.executable,
                str(inpaint_py),
                "--backend",
                str(args.backend),
            ]
            if input_path.is_file():
                mask_file = mask_dir / f"{input_path.stem}_mask.png"
                if not mask_file.exists():
                    print(f"[FAIL] expected mask not found: {mask_file}", file=sys.stderr)
                    return 12
                cmd_inpaint += [
                    "--image",
                    str(input_path),
                    "--mask",
                    str(mask_file),
                    "--output",
                    str(out_dir / f"{input_path.stem}.png"),
                ]
            else:
                cmd_inpaint += [
                    "--image-dir",
                    str(input_path),
                    "--mask-dir",
                    str(mask_dir),
                    "--limit",
                    str(int(args.limit or 0)),
                    "--out-dir",
                    str(out_dir),
                ]
            cmd_inpaint += [
                "--model",
                ("flux" if str(args.backend) in {"flux_fill_api", "flux_fill_local"} else (
                    "sdxl" if str(args.backend) == "sdxl_inpaint_local" else "lama"
                )),
                "--device",
                str(args.device),
                "--method-subdir",
                "--expand-mask",
                str(em),
            ]
            if args.backend == "flux_fill_api":
                if args.flux_prompt:
                    cmd_inpaint += ["--flux-prompt", str(args.flux_prompt)]
                cmd_inpaint += [
                    "--flux-steps", str(args.flux_steps),
                    "--flux-guidance", str(args.flux_guidance),
                    "--flux-safety-tolerance", str(args.flux_safety_tolerance),
                    "--flux-output-format", str(args.flux_output_format),
                    "--flux-endpoint", str(args.flux_endpoint),
                    "--flux-poll-interval", str(args.flux_poll_interval),
                    "--flux-poll-timeout", str(args.flux_poll_timeout),
                ]
            elif args.backend == "flux_fill_local":
                if args.flux_prompt:
                    cmd_inpaint += ["--flux-prompt", str(args.flux_prompt)]
                cmd_inpaint += [
                    "--flux-steps",
                    str(args.flux_steps),
                    "--flux-guidance",
                    str(args.flux_guidance),
                    "--flux-model-dir",
                    str(args.flux_model_dir),
                    "--flux-lora-path",
                    str(args.flux_lora_path),
                    "--flux-lora-scale",
                    str(args.flux_lora_scale),
                ]

            elif args.backend == "sdxl_inpaint_local":
                if args.flux_prompt:
                    cmd_inpaint += ["--flux-prompt", str(args.flux_prompt)]
                cmd_inpaint += [
                    "--sdxl-model",
                    str(args.sdxl_model),
                    "--sdxl-negative-prompt",
                    str(args.sdxl_negative_prompt),
                    "--sdxl-steps",
                    str(args.sdxl_steps),
                    "--sdxl-guidance",
                    str(args.sdxl_guidance),
                    "--sdxl-strength",
                    str(args.sdxl_strength),
                ]
            
            if args.skip_existing:
                cmd_inpaint.append("--skip-existing")
            if args.adaptive:
                cmd_inpaint.append("--adaptive")
                cmd_inpaint += [
                    "--adaptive-edge-percentile",
                    str(args.adaptive_edge_percentile),
                    "--adaptive-near-edge-px",
                    str(args.adaptive_near_edge_px),
                    "--adaptive-far-edge-px",
                    str(args.adaptive_far_edge_px),
                    "--adaptive-min-expand",
                    str(args.adaptive_min_expand),
                ]
            if args.auto_expand:
                cmd_inpaint += [
                    "--auto-expand",
                    "--auto-expand-max",
                    str(args.auto_expand_max),
                    "--auto-expand-delta",
                    str(args.auto_expand_delta),
                    "--auto-expand-min",
                    str(args.auto_expand_min),
                ]
            if args.debug_vis:
                cmd_inpaint.append("--debug-vis")
            if args.debug_vis_dir:
                cmd_inpaint += ["--debug-vis-dir", str(args.debug_vis_dir)]
            if args.save_masked:
                cmd_inpaint.append("--save-masked")
            if args.save_debug:
                cmd_inpaint.append("--save-debug")
            if args.postprocess:
                cmd_inpaint += [
                    "--postprocess",
                    "--post-delta",
                    str(args.post_delta),
                    "--post-min-inner",
                    str(args.post_min_inner),
                    "--post-max-inner",
                    str(args.post_max_inner),
                    "--post-margin",
                    str(args.post_margin),
                    "--post-dilate",
                    str(args.post_dilate),
                ]
                if args.post_dry_run:
                    cmd_inpaint.append("--post-dry-run")
                if args.post_debug_dir:
                    cmd_inpaint += ["--post-debug-dir", str(args.post_debug_dir)]
                cmd_inpaint += [
                    "--post-noise-mode",
                    str(args.post_noise_mode),
                    "--post-noise-percentile",
                    str(args.post_noise_percentile),
                    "--post-noise-sigma",
                    str(args.post_noise_sigma),
                    "--post-noise-edge",
                    str(args.post_noise_edge),
                    "--post-noise-flat",
                    str(args.post_noise_flat),
                    "--post-noise-kmeans-sep",
                    str(args.post_noise_kmeans_sep),
                    "--post-noise-kmeans-max-frac",
                    str(args.post_noise_kmeans_max_frac),
                    "--post-noise-z",
                    str(args.post_noise_z),
                    "--post-noise-min-area",
                    str(args.post_noise_min_area),
                ]
                if args.post_noise_vis_dir:
                    cmd_inpaint += ["--post-noise-vis-dir", str(args.post_noise_vis_dir)]
                if args.post_no_noise_vis:
                    cmd_inpaint.append("--post-no-noise-vis")
                if args.save_masked:
                    cmd_inpaint.append("--save-masked")
            if args.skip_missing_mask:
                cmd_inpaint.append("--skip-missing-mask")

            print(f"\n=== Inpaint: mask_dilate={md}, expand_mask={em}, adaptive={args.adaptive} -> {out_dir} ===")
            code = _run(cmd_inpaint)
            if code != 0:
                print(f"[FAIL] inpaint stage failed: exit={code}", file=sys.stderr)
                return 11

    print("\nAll done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
