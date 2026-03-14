import argparse
import base64
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from azure import AzureOCR


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _method_subdir_name(
    mask_mode: str,
    remove_method: str,
    inpaint_method: str,
    inpaint_radius: int,
    pixel_delta: int,
    mask_dilate: int,
    mask_dilate_adaptive: bool,
    mask_dilate_adaptive_mode: str,
    mask_dilate_min: int,
    mask_dilate_max: int,
    mask_dilate_font_ref: float,
    mask_dilate_bbox_ratio: float,
    mask_blur: float,
    bg_sample_off: int,
    erase_sam3: bool,
    sam3_prompts: str,
    sam3_score_threshold: float,
    sam3_min_area: int,
    sam3_dilate: int,
) -> str:
    parts = [f"mask-{mask_mode}", f"rm-{remove_method}"]
    if mask_mode == "pixel":
        parts.append(f"pd{pixel_delta}")
    if mask_dilate:
        parts.append(f"d{int(mask_dilate)}")
    if mask_dilate_adaptive:
        parts.append(
            f"ad-{mask_dilate_adaptive_mode}-"
            f"{int(mask_dilate_min)}-{int(mask_dilate_max)}-"
            f"fr{float(mask_dilate_font_ref):g}-br{float(mask_dilate_bbox_ratio):g}"
        )
    if remove_method == "cv2":
        parts.append(f"ip-{inpaint_method}")
        parts.append(f"r{int(inpaint_radius)}")
    else:
        if bg_sample_off:
            parts.append(f"bg{int(bg_sample_off)}")
        if mask_blur:
            parts.append(f"b{float(mask_blur):g}")
    if erase_sam3:
        parts.append(f"es-{sam3_prompts or ''}-{float(sam3_score_threshold):g}-{int(sam3_min_area)}-{int(sam3_dilate)}")
    return "_".join(parts)


def _adaptive_dilate_radius(
    base: int,
    font_size_px: Optional[float],
    bbox_h: Optional[float],
    r_min: int,
    r_max: int,
    font_ref: float,
    bbox_ratio: float,
    mode: str,
) -> int:
    if mode == "bbox":
        if not bbox_h or bbox_h <= 0:
            return _clamp(int(base), int(r_min), int(r_max))
        r = int(round(float(base) + float(bbox_h) * float(bbox_ratio)))
        return _clamp(r, int(r_min), int(r_max))

    if mode == "max":
        r0 = int(base)
        if bbox_h and bbox_h > 0:
            r0 = max(r0, int(round(float(base) + float(bbox_h) * float(bbox_ratio))))
        if font_size_px and font_size_px > 0:
            scale = float(font_ref) / float(font_size_px)
            r0 = max(r0, int(round(float(base) * scale)))
        return _clamp(int(r0), int(r_min), int(r_max))

    # mode == "font" (default)
    if not font_size_px or font_size_px <= 0:
        return _clamp(int(base), int(r_min), int(r_max))
    scale = float(font_ref) / float(font_size_px)
    r = int(round(float(base) * scale))
    return _clamp(r, int(r_min), int(r_max))


def _load_font(size: int = 16) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _safe_get_pixel(img: Image.Image, x: int, y: int) -> Optional[Tuple[int, int, int]]:
    w, h = img.size
    if x < 0 or y < 0 or x >= w or y >= h:
        return None, None
    p = img.getpixel((x, y))
    if isinstance(p, int):
        return (p, p, p)
    if len(p) >= 3:
        return (int(p[0]), int(p[1]), int(p[2]))
    return None


def _avg_color(pixels: Iterable[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    px = list(pixels)
    if not px:
        return (255, 255, 255)
    r = sum(p[0] for p in px) / len(px)
    g = sum(p[1] for p in px) / len(px)
    b = sum(p[2] for p in px) / len(px)
    return (int(r), int(g), int(b))


def _median_color(pixels: Iterable[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    px = list(pixels)
    if not px:
        return (255, 255, 255)
    rs = [p[0] for p in px]
    gs = [p[1] for p in px]
    bs = [p[2] for p in px]
    return (
        int(statistics.median(rs)),
        int(statistics.median(gs)),
        int(statistics.median(bs)),
    )


def _estimate_bg_color(img: Image.Image, pts: List[Tuple[int, int]], off: int) -> Tuple[int, int, int]:
    w, h = img.size
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    off = int(off) if int(off) > 0 else 1
    xmin2 = _clamp(int(xmin - off), 0, w - 1)
    xmax2 = _clamp(int(xmax + off), 0, w - 1)
    ymin2 = _clamp(int(ymin - off), 0, h - 1)
    ymax2 = _clamp(int(ymax + off), 0, h - 1)

    px: List[Tuple[int, int, int]] = []

    def add_sample(x: int, y: int) -> None:
        p = _safe_get_pixel(img, x, y)
        if p is not None:
            px.append(p)

    # Sample along the expanded bbox perimeter to better match local background.
    steps = 25
    for i in range(steps + 1):
        t = i / steps
        x = int(round(xmin2 + (xmax2 - xmin2) * t))
        add_sample(x, ymin2)
        add_sample(x, ymax2)
        y = int(round(ymin2 + (ymax2 - ymin2) * t))
        add_sample(xmin2, y)
        add_sample(xmax2, y)

    # Prefer median to reduce influence from nearby text/lines.
    return _median_color(px)


def _build_mask(size: Tuple[int, int], polygons: List[List[Tuple[int, int]]]) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for pts in polygons:
        if len(pts) >= 3:
            draw.polygon(pts, fill=255)
    return mask


def _hist_median(hist: List[int]) -> int:
    total = sum(hist)
    if total <= 0:
        return 128
    half = total / 2
    acc = 0
    for i, c in enumerate(hist):
        acc += c
        if acc >= half:
            return i
    return 128


def _build_polygon_mask(size: Tuple[int, int], pts: List[Tuple[int, int]]) -> Image.Image:
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    if len(pts) >= 3:
        d.polygon(pts, fill=255)
    return m


def _build_pixel_mask(
    img: Image.Image,
    polygons: List[List[Tuple[int, int]]],
    delta: int,
) -> Image.Image:
    w, h = img.size
    out = Image.new("L", (w, h), 0)
    for pts in polygons:
        if len(pts) < 3:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        pad = 2
        xmin2 = _clamp(int(xmin - pad), 0, w - 1)
        xmax2 = _clamp(int(xmax + pad), 0, w - 1)
        ymin2 = _clamp(int(ymin - pad), 0, h - 1)
        ymax2 = _clamp(int(ymax + pad), 0, h - 1)
        if xmax2 <= xmin2 or ymax2 <= ymin2:
            continue

        roi = img.crop((xmin2, ymin2, xmax2 + 1, ymax2 + 1)).convert("L")
        med = _hist_median(roi.histogram())
        d = int(delta) if int(delta) > 0 else 18

        m = roi.point(lambda p, med=med, d=d: 255 if abs(int(p) - int(med)) >= d else 0)
        poly_local = [(p[0] - xmin2, p[1] - ymin2) for p in pts]
        poly_m = _build_polygon_mask(roi.size, poly_local)
        m = Image.composite(m, Image.new("L", roi.size, 0), poly_m)
        out.paste(m, (xmin2, ymin2), m)

    return out


def _dilate_mask(mask: Image.Image, radius: int) -> Image.Image:
    if radius <= 0:
        return mask
    # MaxFilter size must be odd.
    size = max(1, radius) * 2 + 1
    return mask.filter(ImageFilter.MaxFilter(size=size))


def _blur_and_binarize_mask(mask: Image.Image, blur_radius: float) -> Image.Image:
    br = float(blur_radius or 0.0)
    if br <= 0:
        return mask
    m = mask.filter(ImageFilter.GaussianBlur(radius=br))
    return m.point(lambda p: 255 if int(p) >= 128 else 0)


def _build_sam3_union_mask(
    *,
    image_path: Path,
    size: Tuple[int, int],
    prompts: List[str],
    score_threshold: float,
    min_area: int,
    dilate: int,
    pythonpath: str,
) -> Tuple[Optional[Image.Image], Optional[List[Dict[str, object]]]]:
    # Import Image at function start so it's always available
    from PIL import Image
    
    try:
        extra = str(pythonpath or "").strip()
        if extra:
            for part in [p.strip() for p in extra.split(":") if p.strip()]:
                if part not in sys.path:
                    sys.path.insert(0, part)
        # Ensure the parent of litserve_model is in sys.path
        root = Path(__file__).resolve().parent
        lm_parent = None
        if (root / "litserve_model").is_dir():
            lm_parent = root
        elif (root.parent / "litserve_model").is_dir():
            lm_parent = root.parent

        if lm_parent:
            lm_path = str(lm_parent)
            req_path = str(lm_parent / "litserve_model" / "request")
            sam3_root = str(lm_parent / "litserve_model" / "sam3")
            
            # 1. Add paths to sys.path
            for p in [lm_path, req_path, sam3_root]:
                if p not in sys.path:
                    sys.path.insert(0, p)

            # 2. Mock 'config.read_config' if it's missing (needed by service_url.py)
            import types
            if "config" not in sys.modules:
                cfg_mod = types.ModuleType("config")
                sys.modules["config"] = cfg_mod
                
                read_cfg_mod = types.ModuleType("config.read_config")
                read_cfg_mod.get_config = lambda k, default=None: {
                    "services.litserve.host": "127.0.0.1",
                    "services.litserve.port_sam3": 8022,
                    "services.litserve.scheme": "http"
                }.get(k, default)
                read_cfg_mod.load_config = lambda *a, **k: None
                sys.modules["config.read_config"] = read_cfg_mod
                cfg_mod.read_config = read_cfg_mod

            # 3. Mock 'request_logging' if it's missing (needed by sam3_request.py)
            if "request_logging" not in sys.modules:
                rl_mod = types.ModuleType("request_logging")
                rl_mod.log_request_metrics = lambda *a, **k: None
                sys.modules["request_logging"] = rl_mod

        from litserve_model.request import sam3_request
    except Exception as e:
        print(f"[SAM3_MASK_FAIL] import sam3_request failed: {e}", file=sys.stderr)
        # Return an empty mask so the pipeline can continue (mask will be OCR-only).
        from PIL import Image as PILImage
        return PILImage.new("L", size, 0), []

    try:
        raw = sam3_request.call_sam3_service(
            str(image_path),
            list(prompts or []),
            score_threshold=float(score_threshold),
            min_area=int(min_area),
        )
        print(f"[SAM3_BUILD] call_sam3_service returned {len(raw) if raw else 0} items", flush=True)
    except Exception as e:
        print(f"[SAM3_MASK_FAIL] call_sam3_service failed: {e}", file=sys.stderr)
        from PIL import Image as PILImage
        return PILImage.new("L", size, 0), []

    if not raw:
        print("[SAM3_BUILD] raw is empty, returning empty mask", flush=True)
        from PIL import Image as PILImage
        return PILImage.new("L", size, 0), []

    import numpy as np

    w, h = int(size[0]), int(size[1])
    union = np.zeros((h, w), dtype=np.uint8)
    for i, item in enumerate(raw):
        m = item.get("mask")
        prompt = item.get("prompt")
        score = item.get("score")
        if m is None:
            continue
        try:
            m_np = np.array(m, dtype=np.uint8)
            # 打印每个 mask 的统计信息
            if i < 5 or i == len(raw) - 1:
                print(f"[SAM3_BUILD] item[{i}] prompt={prompt} score={score:.3f} shape={m_np.shape} dtype={m_np.dtype} range=[{m_np.min()},{m_np.max()}] sum={m_np.sum()}", flush=True)
        except Exception as e:
            print(f"[SAM3_BUILD] item[{i}] np.array conversion failed: {e}", flush=True)
            continue
        
        if m_np.ndim > 2:
            m_np = m_np.squeeze()
        
        if m_np.shape[0] != h or m_np.shape[1] != w:
            try:
                # 修复潜在的归一化问题：如果 mask 是 0-1 之间的 float，先转为 0/255
                m_img = Image.fromarray((m_np > 0).astype(np.uint8) * 255, mode="L")
                m_img = m_img.resize((w, h), resample=Image.NEAREST)
                m_np = np.array(m_img, dtype=np.uint8)
            except Exception as e:
                print(f"[SAM3_BUILD] item[{i}] resize failed: {e}", flush=True)
                continue
        
        # 确保二值化
        m_binary = (m_np > 0).astype(np.uint8) * 255
        union = np.maximum(union, m_binary)

    print(f"[SAM3_BUILD] final union mask: shape={union.shape} range=[{union.min()},{union.max()}] nonzero={np.count_nonzero(union)}", flush=True)

    if int(dilate) > 0:
        union = _dilate_mask_np(union, int(dilate))

    return Image.fromarray(union, mode="L"), raw


def _dilate_mask_np(mask_np: "object", r: int) -> "object":
    import numpy as np

    if r <= 0:
        return mask_np
    try:
        from scipy import ndimage

        k = np.ones((int(r), int(r)), dtype=np.uint8)
        out = ndimage.binary_dilation(mask_np > 0, structure=k).astype(np.uint8) * 255
        return out
    except Exception:
        return mask_np


def _analyze_one(
    ocr: AzureOCR,
    img_path: Path,
    out_dir: Path,
    seq_index: int,
    font: ImageFont.ImageFont,
    save_mask: bool,
    remove_text: bool,
    mask_mode: str,
    mask_dilate: int,
    mask_dilate_adaptive: bool,
    mask_dilate_adaptive_mode: str,
    mask_dilate_min: int,
    mask_dilate_max: int,
    mask_dilate_font_ref: float,
    mask_dilate_bbox_ratio: float,
    mask_blur: float,
    remove_method: str,
    inpaint_radius: int,
    inpaint_method: str,
    pixel_delta: int,
    bg_sample_off: int,
    adaptive_edge_aware: bool,
    adaptive_edge_percentile: float,
    adaptive_near_edge_px: float,
    adaptive_pixel_delta_small: int,
    adaptive_mask_dilate_small: int,
    adaptive_mask_blur_small: float,
    adaptive_edge_debug: bool,
    adaptive_edge_stat: str,
    adaptive_edge_metric: str,
    adaptive_mid_edge_px: float,
    adaptive_pixel_delta_mid: int,
    adaptive_mask_dilate_mid: int,
    adaptive_mask_blur_mid: float,
    adaptive_edge_exclude_dilate: int,
    adaptive_edge_min_area: int,
    adaptive_edge_save_edge: bool,
    adaptive_edge_grad_mode: str,
    adaptive_edge_edge_dilate: int,
    erase_sam3: bool,
    sam3_prompts: str,
    sam3_score_threshold: float,
    sam3_min_area: int,
    sam3_dilate: int,
    sam3_pythonpath: str,
    sam3_debug: bool,
    sam3_save_json: bool,
) -> None:
    # Create per-image subfolder: <seq>_<uuid>/
    img_subdir = out_dir / f"{seq_index:03d}_{img_path.stem}"
    img_subdir.mkdir(parents=True, exist_ok=True)
    
    vis_path = img_subdir / "vis.png"
    mask_path = img_subdir / "mask.png"
    no_text_path = img_subdir / "no_text.png"

    send_path = img_path
    tmp_path = None
    if img_path.suffix.lower() == ".webp":
        try:
            with Image.open(img_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
                    tmp_path = Path(f.name)
                img.save(tmp_path, format="PNG")
                send_path = tmp_path
        except Exception as e:
            print(f"[FAIL] {img_path.name}: webp->png convert failed: {e}", file=sys.stderr)
            return

    try:
        result = ocr.analyze_image(str(send_path))
    except Exception as e:
        print(f"[FAIL] {img_path.name}: {e}", file=sys.stderr)
        return
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    print(f"\n=== {img_path.name} ({result.image_width}x{result.image_height}) ===")
    for i, block in enumerate(result.text_blocks, start=1):
        meta = (
            f"font={block.font_name} weight={block.font_weight} style={block.font_style} "
            f"color={block.font_color} bg={block.background_color} size_px={block.font_size_px} "
            f"bold={block.is_bold} italic={block.is_italic} subscript_like={block.has_size_variation}"
        )
        txt = (block.text or "").replace("\n", " ").strip()
        print(f"[{i:03d}] {txt} | {meta}")

    polygons: List[List[Tuple[int, int]]] = []
    poly_meta: List[Tuple[List[Tuple[int, int]], Optional[float], Optional[float]]] = []
    if mask_mode == "word" and getattr(result, "word_polygons", None):
        for poly in result.word_polygons:
            if len(poly) >= 3:
                pts = [(int(round(float(x))), int(round(float(y)))) for (x, y) in poly]
                polygons.append(pts)
    else:
        for block in result.text_blocks:
            poly = block.polygon or []
            if len(poly) >= 3:
                pts = [(int(round(float(x))), int(round(float(y)))) for (x, y) in poly]
                polygons.append(pts)
                try:
                    fs = float(getattr(block, "font_size_px", 0) or 0)
                except Exception:
                    fs = 0
                try:
                    ys = [p[1] for p in pts]
                    bbox_h = float(max(ys) - min(ys)) if ys else 0.0
                except Exception:
                    bbox_h = 0.0
                poly_meta.append((pts, fs if fs > 0 else None, bbox_h if bbox_h > 0 else None))

    try:
        with Image.open(img_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_vis = img.copy()
            draw = ImageDraw.Draw(img_vis)

            for i, block in enumerate(result.text_blocks, start=1):
                poly = block.polygon or []
                if len(poly) >= 4:
                    pts = [(float(x), float(y)) for (x, y) in poly[:4]]
                    pts_closed = pts + [pts[0]]
                    draw.line(pts_closed, width=2, fill=(255, 0, 0))
                    x0, y0 = pts[0]
                    label = str(i)
                    draw.text((x0 + 2, y0 + 2), label, fill=(255, 0, 0), font=font)

            img_vis.save(vis_path)
            print(f"[VIS] saved: {vis_path}")

            if save_mask or remove_text:
                if mask_mode == "pixel":
                    if adaptive_edge_aware:
                        import numpy as np
                        from scipy import ndimage

                        # Build fat and thin masks.
                        fat_delta = int(pixel_delta)
                        fat_dilate = int(mask_dilate)
                        fat_blur = float(mask_blur or 0.0)

                        thin_delta = int(adaptive_pixel_delta_small)
                        thin_dilate = int(adaptive_mask_dilate_small)
                        thin_blur = float(adaptive_mask_blur_small or 0.0)

                        mid_delta = int(adaptive_pixel_delta_mid)
                        mid_dilate = int(adaptive_mask_dilate_mid)
                        mid_blur = float(adaptive_mask_blur_mid or 0.0)

                        fat = _build_pixel_mask(img, polygons, fat_delta)
                        fat = _dilate_mask(fat, fat_dilate)
                        fat = _blur_and_binarize_mask(fat, fat_blur)

                        thin = _build_pixel_mask(img, polygons, thin_delta)
                        thin = _dilate_mask(thin, thin_dilate)
                        thin = _blur_and_binarize_mask(thin, thin_blur)

                        mid = _build_pixel_mask(img, polygons, mid_delta)
                        mid = _dilate_mask(mid, mid_dilate)
                        mid = _blur_and_binarize_mask(mid, mid_blur)

                        # Compute strong-edge map and distance.
                        grad_mode = (adaptive_edge_grad_mode or "gray").strip().lower()
                        if grad_mode not in {"gray", "rgb"}:
                            grad_mode = "gray"
                        if grad_mode == "rgb":
                            # Capture pure chroma boundaries where grayscale contrast is weak.
                            rgb = np.array(img, dtype=np.float32)
                            gx = ndimage.sobel(rgb, axis=1, mode="reflect")
                            gy = ndimage.sobel(rgb, axis=0, mode="reflect")
                            # gx/gy are HxWx3, compute per-channel magnitude then combine.
                            g = np.sqrt((gx * gx + gy * gy).sum(axis=2))
                            grad = g
                        else:
                            gray = np.array(img.convert("L"), dtype=np.float32)
                            gx = ndimage.sobel(gray, axis=1, mode="reflect")
                            gy = ndimage.sobel(gray, axis=0, mode="reflect")
                            grad = np.hypot(gx, gy)
                        fat_a = (np.array(fat, dtype=np.uint8) > 0)
                        thin_a = (np.array(thin, dtype=np.uint8) > 0)
                        mid_a = (np.array(mid, dtype=np.uint8) > 0)

                        # IMPORTANT: exclude edges inside text/icon regions themselves.
                        # Otherwise every component contains strong gradients => dist_to_edge=0.
                        # Build an exclusion region slightly larger than the fat mask.
                        exclude = fat_a
                        try:
                            it = int(max(0, adaptive_edge_exclude_dilate))
                            if it > 0:
                                exclude = ndimage.binary_dilation(exclude, iterations=it)
                        except Exception:
                            exclude = fat_a

                        grad_out = grad.copy()
                        grad_out[exclude] = 0.0

                        ep = float(adaptive_edge_percentile)
                        if not (0.0 < ep < 100.0):
                            ep = 97.0
                        valid = grad_out[~exclude]
                        if valid.size <= 0:
                            thr = float(np.percentile(grad_out, ep))
                        else:
                            thr = float(np.percentile(valid, ep))
                        edge = (grad_out >= thr)

                        # Slightly thicken edges so small/thin boundaries (icons, color-block borders)
                        # survive min-area filtering and yield more stable distance transforms.
                        try:
                            ed_it = int(max(0, adaptive_edge_edge_dilate))
                        except Exception:
                            ed_it = 0
                        if ed_it > 0:
                            try:
                                edge = ndimage.binary_dilation(edge, iterations=ed_it)
                            except Exception:
                                pass

                        # Remove tiny edge fragments (icons/text strokes) so they don't act as "structural" edges.
                        # Keep only edge connected components with enough pixels.
                        try:
                            min_edge_area = int(max(0, adaptive_edge_min_area))
                        except Exception:
                            min_edge_area = 0
                        if min_edge_area > 0:
                            try:
                                elab, enum = ndimage.label(edge)
                                if int(enum) > 0:
                                    sizes = np.bincount(elab.ravel())
                                    keep = (sizes >= int(min_edge_area))
                                    keep[0] = False
                                    edge = keep[elab]
                            except Exception:
                                pass
                        metric = (adaptive_edge_metric or "euclidean").strip().lower()
                        if not bool(np.any(edge)):
                            # No structural edges left (e.g. after exclude/min-area filtering).
                            # SciPy distance_transform_* returns -1 for all pixels if there is no background (no zeros).
                            # In this case, treat everything as "far from edge" so we default to FAT.
                            dist_to_edge = np.full(gray.shape, float(max(gray.shape) * 2), dtype=np.float32)
                        else:
                            if metric in {"chessboard", "taxicab"}:
                                # Integer grid distances; easier to threshold (e.g. near=2).
                                dist_to_edge = ndimage.distance_transform_cdt(~edge, metric=metric).astype(np.float32)
                            else:
                                metric = "euclidean"
                                dist_to_edge = ndimage.distance_transform_edt(~edge).astype(np.float32)

                        near_thr = float(max(0.0, adaptive_near_edge_px))
                        mid_thr = float(max(near_thr, float(adaptive_mid_edge_px)))

                        want_save_edge = bool(adaptive_edge_debug or adaptive_edge_save_edge)
                        if want_save_edge:
                            try:
                                excl_ratio = float(exclude.mean())
                            except Exception:
                                excl_ratio = 0.0
                            try:
                                edge_ratio = float(edge.mean())
                            except Exception:
                                edge_ratio = 0.0
                            if adaptive_edge_debug:
                                print(
                                    f"[OCR_ADAPT][DBG] grad={grad_mode} metric={metric} stat={adaptive_edge_stat} perc={adaptive_edge_percentile:g} "
                                    f"thr={thr:g} edge_ratio={edge_ratio:.4f} excl_ratio={excl_ratio:.4f} "
                                    f"excl_dilate={int(max(0, adaptive_edge_exclude_dilate))} edge_dilate={int(max(0, adaptive_edge_edge_dilate))} "
                                    f"min_edge_area={int(max(0, adaptive_edge_min_area))} "
                                    f"near={near_thr:g} mid={mid_thr:g}"
                                )
                                if not bool(np.any(edge)):
                                    print("[OCR_ADAPT][DBG] WARNING: edge_map is empty after filtering; defaulting dist_to_edge to large values")
                            try:
                                edge_img = Image.fromarray((edge.astype(np.uint8) * 255), mode="L")
                                edge_path = out_dir / f"{img_path.stem}_edge_map.png"
                                edge_img.save(edge_path)

                                d = dist_to_edge.astype(np.float32)
                                dmax = float(np.percentile(d, 99.0)) if d.size else 1.0
                                if not (dmax > 0.0):
                                    dmax = 1.0
                                dn = np.clip(d / dmax, 0.0, 1.0)
                                dist_img = Image.fromarray((dn * 255.0).astype(np.uint8), mode="L")
                                dist_path = out_dir / f"{img_path.stem}_edge_dist.png"
                                dist_img.save(dist_path)
                                if adaptive_edge_debug:
                                    print(f"[OCR_ADAPT][DBG] saved: {edge_path}")
                                    print(f"[OCR_ADAPT][DBG] saved: {dist_path}")
                            except Exception as e:
                                if adaptive_edge_debug:
                                    print(f"[OCR_ADAPT][DBG] save edge debug failed: {e}")

                            try:
                                enum = 0
                                try:
                                    _, enum = ndimage.label(edge)
                                except Exception:
                                    enum = 0
                                summary_path = out_dir / "edge_summary.csv"
                                is_new = (not summary_path.exists())
                                with summary_path.open("a", encoding="utf-8") as f:
                                    if is_new:
                                        f.write(
                                            "image,grad_mode,metric,stat,percentile,thr,edge_ratio,excl_ratio,excl_dilate,edge_dilate,min_edge_area,edge_cc\n"
                                        )
                                    f.write(
                                        f"{img_path.name},{grad_mode},{metric},{adaptive_edge_stat},{adaptive_edge_percentile:g},{thr:g},{edge_ratio:.6f},{excl_ratio:.6f},{int(max(0, adaptive_edge_exclude_dilate))},{int(max(0, adaptive_edge_edge_dilate))},{int(max(0, adaptive_edge_min_area))},{int(enum)}\n"
                                    )
                            except Exception:
                                pass
                        labeled, num = ndimage.label(fat_a)

                        out_a = np.zeros_like(fat_a, dtype=bool)

                        for cid in range(1, int(num) + 1):
                            comp = (labeled == cid)
                            if not np.any(comp):
                                continue

                            # Use component BOUNDARY for distance stats; interior pixels can be far when mask is thick.
                            try:
                                boundary = comp & (~ndimage.binary_erosion(comp, iterations=1))
                            except Exception:
                                boundary = comp
                            dvals = dist_to_edge[boundary]
                            if dvals.size <= 0:
                                dvals = dist_to_edge[comp]

                            if dvals.size:
                                d_min = float(np.min(dvals.astype(np.float32)))
                                d_p10 = float(np.percentile(dvals.astype(np.float32), 10.0))
                            else:
                                d_min = 0.0
                                d_p10 = 0.0

                            stat = (adaptive_edge_stat or "p10").strip().lower()
                            if stat not in {"min", "p10"}:
                                stat = "p10"
                            d_comp = d_min if stat == "min" else d_p10
                            d_use = float(d_comp)
                            if d_use <= near_thr:
                                chosen = thin_a
                                mode_s = "thin"
                                pd0, d0, b0 = thin_delta, thin_dilate, thin_blur
                            elif d_use <= mid_thr:
                                chosen = mid_a
                                mode_s = "mid"
                                pd0, d0, b0 = mid_delta, mid_dilate, mid_blur
                            else:
                                chosen = fat_a
                                mode_s = "fat"
                                pd0, d0, b0 = fat_delta, fat_dilate, fat_blur
                            out_a |= (chosen & comp)
                            if adaptive_edge_debug:
                                try:
                                    bpx = int(boundary.sum())
                                except Exception:
                                    bpx = 0
                                print(
                                    f"[OCR_ADAPT][DBG] comp={cid:03d} px={int(comp.sum())} bpx={bpx} d_min={d_min:.2f} d_p10={d_p10:.2f} -> d={float(d_comp):.2f}"
                                )
                            print(
                                f"[OCR_ADAPT] comp={cid:03d} px={int(comp.sum())} d_comp={d_comp:.2f} -> {mode_s} "
                                f"(pd={int(pd0)}, d={int(d0)}, b={float(b0):g})"
                            )

                        mask = Image.fromarray((out_a.astype(np.uint8) * 255), mode="L")
                    else:
                        mask = _build_pixel_mask(img, polygons, pixel_delta)
                else:
                    if mask_dilate_adaptive and poly_meta:
                        mask = Image.new("L", img.size, 0)
                        for pts, fs, bbox_h in poly_meta:
                            pm = _build_polygon_mask(img.size, pts)
                            r = _adaptive_dilate_radius(
                                base=mask_dilate,
                                font_size_px=fs,
                                bbox_h=bbox_h,
                                r_min=mask_dilate_min,
                                r_max=mask_dilate_max,
                                font_ref=mask_dilate_font_ref,
                                bbox_ratio=mask_dilate_bbox_ratio,
                                mode=mask_dilate_adaptive_mode,
                            )
                            pm = _dilate_mask(pm, r)
                            mask = ImageChops.lighter(mask, pm)
                    else:
                        mask = _build_mask(img.size, polygons)
                        mask = _dilate_mask(mask, mask_dilate)
                if mask_mode == "pixel" and not adaptive_edge_aware:
                    mask = _dilate_mask(mask, mask_dilate)

                # Ensure blur affects saved mask as well (not only remove_text feather).
                if mask_blur and mask_blur > 0:
                    mask = _blur_and_binarize_mask(mask, mask_blur)

                # SAM3 logic
                if erase_sam3:
                    try:
                        from PIL import Image as PILImage
                        prompts = [p.strip() for p in str(sam3_prompts or "").split(",") if p.strip()]
                        if not prompts:
                            prompts = [
                                "icon",
                                "picture",
                                "logo",
                                "chart",
                                "function_graph",
                                "arrow",
                                "line",
                                "connector",
                            ]
                        sam3_mask, sam3_raw = _build_sam3_union_mask(
                            image_path=img_path,
                            size=img.size,
                            prompts=prompts,
                            score_threshold=float(sam3_score_threshold),
                            min_area=int(sam3_min_area),
                            dilate=int(sam3_dilate),
                            pythonpath=str(sam3_pythonpath or ""),
                        )
                        if sam3_raw is None:
                            sam3_raw = []
                        
                        if sam3_debug:
                            try:
                                by_prompt: Dict[str, int] = {}
                                for it in sam3_raw:
                                    p = str(it.get("prompt") or "")
                                    by_prompt[p] = by_prompt.get(p, 0) + 1
                                total = int(sum(by_prompt.values()))
                                print(
                                    f"[SAM3] prompts={len(prompts)} thr={float(sam3_score_threshold):g} min_area={int(sam3_min_area)} dilate={int(sam3_dilate)} -> {total} detections"
                                )
                                for k in sorted(by_prompt.keys()):
                                    print(f"[SAM3]   prompt='{k}': {by_prompt[k]}")
                                
                                items = list(sam3_raw)
                                try:
                                    items.sort(key=lambda d: float(d.get("score") or 0.0), reverse=True)
                                except Exception:
                                    pass
                                for j, it in enumerate(items[:30], start=1):
                                    bbox = it.get("bbox")
                                    score = it.get("score")
                                    prompt = it.get("prompt")
                                    try:
                                        bb = [int(v) for v in (bbox or [])][:4]
                                    except Exception:
                                        bb = bbox
                                    print(f"[SAM3]   #{j:02d} prompt={prompt} score={score} bbox={bb}")
                            except Exception as e:
                                print(f"[SAM3_DEBUG_FAIL] {img_path.name}: {e}", file=sys.stderr)

                        if sam3_save_json:
                            try:
                                sam3_json_path = out_dir / f"{img_path.stem}_sam3_raw.json"
                                with sam3_json_path.open("w", encoding="utf-8") as f:
                                    json.dump(sam3_raw, f, ensure_ascii=False)
                                print(f"[SAM3_JSON] saved: {sam3_json_path}")
                            except Exception as e:
                                print(f"[SAM3_JSON_FAIL] {img_path.name}: {e}", file=sys.stderr)

                        if sam3_mask is not None:
                            sam3_path = img_subdir / "sam3_mask.png"
                            sam3_mask.save(sam3_path)
                            print(f"[SAM3_MASK] saved: {sam3_path}")
                            from PIL import ImageChops
                            mask = ImageChops.lighter(mask, sam3_mask)
                    except Exception as e:
                        print(f"[SAM3_MASK_FAIL] {img_path.name}: {e}", file=sys.stderr)

                if save_mask:
                    mask.save(mask_path)
                    print(f"[MASK] saved: {mask_path}")

                if remove_text:
                    if remove_method == "cv2":
                        try:
                            import numpy as np
                            import cv2

                            img_np = np.array(img)
                            mask_np = np.array(mask)
                            if mask_np.ndim == 3:
                                mask_np = mask_np[:, :, 0]
                            mask_np = (mask_np > 0).astype("uint8") * 255

                            radius = int(inpaint_radius) if int(inpaint_radius) > 0 else 3
                            algo = cv2.INPAINT_TELEA
                            if inpaint_method == "ns":
                                algo = cv2.INPAINT_NS
                            out_np = cv2.inpaint(img_np, mask_np, radius, algo)
                            img_no_text = Image.fromarray(out_np)
                            img_no_text.save(no_text_path)
                            print(f"[NO_TEXT] saved: {no_text_path}")
                        except Exception as e:
                            print(
                                f"[NO_TEXT_FAIL] {img_path.name}: cv2 inpaint failed ({e}). "
                                f"Try --remove-method fill, or install dependencies: pip install opencv-python numpy",
                                file=sys.stderr,
                            )
                    else:
                        img_filled = img.copy()
                        draw2 = ImageDraw.Draw(img_filled)
                        for pts in polygons:
                            if len(pts) >= 3:
                                fill = _estimate_bg_color(img, pts, bg_sample_off)
                                draw2.polygon(pts, fill=fill)

                        # Feather edges to avoid obvious blocks.
                        feather = mask
                        if mask_blur and mask_blur > 0:
                            feather = mask.filter(ImageFilter.GaussianBlur(radius=float(mask_blur)))
                        img_no_text = Image.composite(img_filled, img, feather)
                        img_no_text.save(no_text_path)
                        print(f"[NO_TEXT] saved: {no_text_path}")

    except Exception as e:
        print(f"[VIS_FAIL] {img_path.name}: {e}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Azure Document Intelligence OCR on a folder and visualize polygons")
    parser.add_argument(
        "--input",
        default=str(Path(__file__).parent / "图例"),
        help="Input image folder or file (default: ./图例)",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "ocr_out"),
        help="Output folder for visualization images (default: ./ocr_out)",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:5000",
        help="Azure Document Intelligence container endpoint (default: http://localhost:5000)",
    )
    parser.add_argument(
        "--save-mask",
        action="store_true",
        help="Save text mask image (default: false)",
    )
    parser.add_argument(
        "--remove-text",
        action="store_true",
        help="Generate image with text regions filled by estimated background color (default: false)",
    )
    parser.add_argument(
        "--mask-mode",
        choices=["line", "word", "pixel"],
        default="word",
        help="Mask granularity: word uses Azure word polygons if available; pixel refines within line boxes; otherwise use line (default: word)",
    )

    parser.add_argument(
        "--erase-sam3",
        action="store_true",
        help="Union SAM3 detection masks (arrow/icon/etc) into the final mask",
    )
    parser.add_argument(
        "--sam3-prompts",
        default="",
        help="Comma-separated SAM3 prompts to erase. If empty and --erase-sam3 is set, uses a default set for icon/picture/logo/chart/function_graph + arrow/line/connector",
    )
    parser.add_argument("--sam3-score-threshold", type=float, default=0.45)
    parser.add_argument("--sam3-min-area", type=int, default=50)
    parser.add_argument("--sam3-dilate", type=int, default=2)
    parser.add_argument(
        "--sam3-pythonpath",
        default="",
        help="Optional extra PYTHONPATH (colon-separated) to locate litserve_model when using --erase-sam3",
    )
    parser.add_argument(
        "--sam3-debug",
        action="store_true",
        help="Print SAM3 detection summary (prompt/score/bbox) when --erase-sam3 is enabled",
    )
    parser.add_argument(
        "--sam3-save-json",
        action="store_true",
        help="Save SAM3 raw detections as *_sam3_raw.json when --erase-sam3 is enabled",
    )
    parser.add_argument(
        "--mask-dilate",
        type=int,
        default=2,
        help="Dilate mask radius in pixels to cover thin strokes (default: 2; set 0 to disable)",
    )
    parser.add_argument(
        "--mask-dilate-adaptive",
        action="store_true",
        help="Adaptive dilate per OCR block based on font_size_px (smaller text -> larger dilate) (default: false)",
    )
    parser.add_argument(
        "--mask-dilate-adaptive-mode",
        choices=["font", "bbox", "max"],
        default="font",
        help="Adaptive mode: font uses font_size_px; bbox uses polygon bbox height; max takes max of both (default: font)",
    )
    parser.add_argument(
        "--mask-dilate-min",
        type=int,
        default=1,
        help="Adaptive dilate min radius (default: 1)",
    )
    parser.add_argument(
        "--mask-dilate-max",
        type=int,
        default=6,
        help="Adaptive dilate max radius (default: 6)",
    )
    parser.add_argument(
        "--mask-dilate-font-ref",
        type=float,
        default=24.0,
        help="Adaptive dilate reference font size in px (default: 24.0)",
    )
    parser.add_argument(
        "--mask-dilate-bbox-ratio",
        type=float,
        default=0.12,
        help="Adaptive bbox mode padding ratio: r ~= base + bbox_h * ratio (default: 0.12)",
    )
    parser.add_argument(
        "--mask-blur",
        type=float,
        default=2.0,
        help="Mask feather blur radius for smooth edges (default: 2.0; set 0 to disable)",
    )
    parser.add_argument(
        "--pixel-delta",
        type=int,
        default=18,
        help="pixel mask sensitivity against local median (default: 18; smaller removes more)",
    )
    parser.add_argument(
        "--pixel-edge-aware",
        action="store_true",
        help="Pixel mask edge-aware mode: for each connected component, if it is close to strong edges/lines, use smaller (pixel-delta/mask-dilate/mask-blur) to avoid over-masking near borders (default: false)",
    )
    parser.add_argument("--pixel-edge-percentile", type=float, default=97.0)
    parser.add_argument("--pixel-edge-near", type=float, default=5.0)
    parser.add_argument("--pixel-edge-mid", type=float, default=10.0)
    parser.add_argument(
        "--pixel-edge-exclude-dilate",
        type=int,
        default=6,
        help="Dilate fat-mask by N pixels when excluding gradients for edge detection (prevents treating other fill-block boundaries as structural edges)",
    )
    parser.add_argument(
        "--pixel-edge-min-area",
        type=int,
        default=200,
        help="Minimum connected-component size (in pixels) for an edge fragment to be kept as a structural edge",
    )
    parser.add_argument(
        "--pixel-edge-edge-dilate",
        type=int,
        default=1,
        help="Dilate the structural edge map by N pixels before min-area filtering and distance transform (helps keep thin/small boundaries)",
    )
    parser.add_argument(
        "--pixel-edge-save-edge",
        action="store_true",
        help="Save structural edge_map/dist_to_edge images and write edge_summary.csv (does not require --pixel-edge-debug)",
    )
    parser.add_argument(
        "--pixel-edge-grad-mode",
        choices=["gray", "rgb"],
        default="gray",
        help="Gradient source for structural edge detection. 'rgb' captures color-block boundaries better; 'gray' is more conservative (default: gray)",
    )
    parser.add_argument(
        "--pixel-edge-stat",
        choices=["min", "p10"],
        default="min",
        help="Which distance statistic to use for near-edge decision: min is more aggressive (more thin), p10 is more robust (default: min)",
    )
    parser.add_argument(
        "--pixel-edge-metric",
        choices=["euclidean", "chessboard", "taxicab"],
        default="euclidean",
        help="Distance metric used to compute distance-to-edge. euclidean may yield fractional minima (e.g. 2.24). chessboard/taxicab are integer grid distances (default: euclidean)",
    )
    parser.add_argument(
        "--pixel-edge-debug",
        action="store_true",
        help="Print debug stats for pixel edge-aware selection (edge threshold/ratios and per-component distance stats)",
    )
    parser.add_argument(
        "--pixel-delta-small",
        type=int,
        default=48,
        help="Edge-thin pixel-delta override for components near edges. Note: for pixel mask, larger delta => thinner mask; smaller delta => fatter mask.",
    )
    parser.add_argument("--pixel-delta-mid", type=int, default=42)
    parser.add_argument("--mask-dilate-small", type=int, default=1)
    parser.add_argument("--mask-blur-small", type=float, default=1.0)
    parser.add_argument("--mask-dilate-mid", type=int, default=2)
    parser.add_argument("--mask-blur-mid", type=float, default=2.0)
    parser.add_argument(
        "--bg-sample-off",
        type=int,
        default=6,
        help="fill method background sampling offset in pixels (default: 6; smaller reduces gradient artifacts)",
    )
    parser.add_argument(
        "--remove-method",
        choices=["fill", "cv2"],
        default="fill",
        help="Text removal method: fill (fast) or cv2 (inpaint, better for textured backgrounds; requires opencv-python+numpy) (default: fill)",
    )
    parser.add_argument(
        "--inpaint-radius",
        type=int,
        default=3,
        help="cv2 inpaint radius (default: 3)",
    )
    parser.add_argument(
        "--inpaint-method",
        choices=["telea", "ns"],
        default="telea",
        help="cv2 inpaint algorithm: telea (default) or ns (Navier-Stokes)",
    )
    parser.add_argument(
        "--method-subdir",
        action="store_true",
        help="Write outputs into a per-method subdirectory under --output (default: false)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N images (0 = no limit, default: 0)",
    )
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    if not in_dir.exists():
        print(f"Input path not found: {in_dir}", file=sys.stderr)
        return 2
    if in_dir.is_file() and in_dir.suffix.lower() not in IMAGE_EXTS:
        print(f"Input file is not a supported image: {in_dir}", file=sys.stderr)
        return 2

    _ensure_dir(out_dir)

    if args.method_subdir:
        sub = _method_subdir_name(
            mask_mode=args.mask_mode,
            remove_method=args.remove_method,
            inpaint_method=args.inpaint_method,
            inpaint_radius=args.inpaint_radius,
            pixel_delta=args.pixel_delta,
            mask_dilate=args.mask_dilate,
            mask_dilate_adaptive=args.mask_dilate_adaptive,
            mask_dilate_adaptive_mode=args.mask_dilate_adaptive_mode,
            mask_dilate_min=args.mask_dilate_min,
            mask_dilate_max=args.mask_dilate_max,
            mask_dilate_font_ref=args.mask_dilate_font_ref,
            mask_dilate_bbox_ratio=args.mask_dilate_bbox_ratio,
            mask_blur=args.mask_blur,
            bg_sample_off=args.bg_sample_off,
            erase_sam3=bool(args.erase_sam3),
            sam3_prompts=str(args.sam3_prompts or ""),
            sam3_score_threshold=float(args.sam3_score_threshold),
            sam3_min_area=int(args.sam3_min_area),
            sam3_dilate=int(args.sam3_dilate),
        )
        out_dir = out_dir / sub
        _ensure_dir(out_dir)

    try:
        ocr = AzureOCR(endpoint=args.endpoint)
    except Exception as e:
        print(f"Failed to init AzureOCR: {e}", file=sys.stderr)
        return 3

    font = _load_font(16)

    if in_dir.is_file():
        paths = [in_dir]
    else:
        paths = [p for p in sorted(in_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not paths:
        print(f"No images found in: {in_dir}", file=sys.stderr)
        return 4

    # Apply limit if specified
    limit = int(args.limit or 0)
    if limit > 0:
        paths = paths[:limit]
        print(f"Found {len(paths)} images (limited to {limit})")
    else:
        print(f"Found {len(paths)} images")
    for idx, p in enumerate(paths, 1):
        _analyze_one(
            ocr,
            p,
            out_dir,
            idx,
            font,
            save_mask=args.save_mask,
            remove_text=args.remove_text,
            mask_mode=args.mask_mode,
            mask_dilate=args.mask_dilate,
            mask_dilate_adaptive=args.mask_dilate_adaptive,
            mask_dilate_adaptive_mode=args.mask_dilate_adaptive_mode,
            mask_dilate_min=args.mask_dilate_min,
            mask_dilate_max=args.mask_dilate_max,
            mask_dilate_font_ref=args.mask_dilate_font_ref,
            mask_dilate_bbox_ratio=args.mask_dilate_bbox_ratio,
            mask_blur=args.mask_blur,
            remove_method=args.remove_method,
            inpaint_radius=args.inpaint_radius,
            inpaint_method=args.inpaint_method,
            pixel_delta=args.pixel_delta,
            bg_sample_off=args.bg_sample_off,
            adaptive_edge_aware=bool(args.pixel_edge_aware),
            adaptive_edge_percentile=float(args.pixel_edge_percentile),
            adaptive_near_edge_px=float(args.pixel_edge_near),
            adaptive_pixel_delta_small=int(args.pixel_delta_small),
            adaptive_mask_dilate_small=int(args.mask_dilate_small),
            adaptive_mask_blur_small=float(args.mask_blur_small),
            adaptive_edge_debug=bool(args.pixel_edge_debug),
            adaptive_edge_stat=str(args.pixel_edge_stat),
            adaptive_edge_metric=str(args.pixel_edge_metric),
            adaptive_mid_edge_px=float(args.pixel_edge_mid),
            adaptive_pixel_delta_mid=int(args.pixel_delta_mid),
            adaptive_mask_dilate_mid=int(args.mask_dilate_mid),
            adaptive_mask_blur_mid=float(args.mask_blur_mid),
            adaptive_edge_exclude_dilate=int(args.pixel_edge_exclude_dilate),
            adaptive_edge_min_area=int(args.pixel_edge_min_area),
            adaptive_edge_save_edge=bool(args.pixel_edge_save_edge),
            adaptive_edge_grad_mode=str(args.pixel_edge_grad_mode),
            adaptive_edge_edge_dilate=int(args.pixel_edge_edge_dilate),
            erase_sam3=bool(args.erase_sam3),
            sam3_prompts=str(args.sam3_prompts or ""),
            sam3_score_threshold=float(args.sam3_score_threshold),
            sam3_min_area=int(args.sam3_min_area),
            sam3_dilate=int(args.sam3_dilate),
            sam3_pythonpath=str(args.sam3_pythonpath or ""),
            sam3_debug=bool(args.sam3_debug),
            sam3_save_json=bool(args.sam3_save_json),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
