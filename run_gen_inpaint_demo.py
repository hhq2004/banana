import argparse
import base64
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


_FLUX_LOCAL_PIPE_CACHE: dict[tuple[str, str, str], "object"] = {}
_SDXL_LOCAL_PIPE_CACHE: dict[tuple[str, str], "object"] = {}


def _run(cmd: list[str]) -> int:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=False, bufsize=1)
    assert p.stdout is not None
    try:
        for line in p.stdout:
            print(line, end="")
    finally:
        p.stdout.close()
    return p.wait()


def _iopaint_device(device: str) -> str:
    d = (device or "").lower()
    if d in {"gpu", "cuda"}:
        return "cuda"
    return "cpu"


def _load_and_expand_mask_for_iopaint(mask_path: Path, expand_mask: int) -> "object":
    from PIL import Image, ImageFilter

    mask = Image.open(mask_path).convert("L")
    if expand_mask and expand_mask > 0:
        mask = mask.point(lambda p: 255 if p >= 127 else 0)
        size = int(expand_mask) * 2 + 1
        mask = mask.filter(ImageFilter.MaxFilter(size=size))
    return mask


def _iopaint_run(
    image: Path,
    mask: Path,
    output: Path,
    model: str,
    device: str,
    expand_mask: int,
    adaptive: bool,
    config: str,
) -> None:
    exe = shutil.which("iopaint")
    if not exe:
        raise RuntimeError("iopaint is not installed or not on PATH. Install: pip install iopaint")

    with tempfile.TemporaryDirectory(prefix="iopaint-mask-single-") as td:
        tmp_mask_dir = Path(td)
        tmp_mask = tmp_mask_dir / f"{image.stem}.png"
        if adaptive:
            m = _create_adaptive_mask(image, mask, expand_mask)
        else:
            m = _load_and_expand_mask_for_iopaint(mask, expand_mask)
        m.save(tmp_mask)

        cmd = [
            exe,
            "run",
            f"--model={model}",
            f"--device={_iopaint_device(device)}",
            f"--image={str(image)}",
            f"--mask={str(tmp_mask)}",
            f"--output={str(output)}",
        ]
        if config:
            cmd.append(f"--config={config}")
        code = _run(cmd)
        if code != 0:
            raise RuntimeError(f"iopaint failed with exit code {code}")


def _iopaint_run_batch(
    image_dir: Path,
    mask_dir: Path,
    out_dir: Path,
    model: str,
    device: str,
    images: list[Path],
    expand_mask: int,
    adaptive: bool,
    config: str,
) -> None:
    exe = shutil.which("iopaint")
    if not exe:
        raise RuntimeError("iopaint is not installed or not on PATH. Install: pip install iopaint")
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="iopaint-mask-") as td:
        tmp_mask_dir = Path(td)
        for img in images:
            src = mask_dir / f"{img.stem}_mask.png"
            if not src.exists():
                # Try new structure: find folder containing this UUID
                for sub in mask_dir.iterdir():
                    if sub.is_dir() and sub.name.endswith(f"_{img.stem}"):
                        candidate = sub / "mask.png"
                        if candidate.exists():
                            src = candidate
                            break
            if not src.exists():
                continue
            dst = tmp_mask_dir / f"{img.stem}.png"
            if adaptive:
                m = _create_adaptive_mask(img, src, expand_mask)
            else:
                m = _load_and_expand_mask_for_iopaint(src, expand_mask)
            m.save(dst)
        cmd = [
            exe,
            "run",
            f"--model={model}",
            f"--device={_iopaint_device(device)}",
            f"--image={str(image_dir)}",
            f"--mask={str(tmp_mask_dir)}",
            f"--output={str(out_dir)}",
        ]
        if config:
            cmd.append(f"--config={config}")
        code = _run(cmd)
        if code != 0:
            raise RuntimeError(f"iopaint failed with exit code {code}")


def _iter_images(image_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        files = [p for p in image_dir.rglob("*") if p.is_file()]
    else:
        files = [p for p in image_dir.iterdir() if p.is_file()]
    return [p for p in sorted(files) if p.suffix.lower() in IMAGE_EXTS]


def _method_dir(base_out_dir: Path, model: str, device: str, hd: bool, method_subdir: bool) -> Path:
    if not method_subdir:
        base_out_dir.mkdir(parents=True, exist_ok=True)
        return base_out_dir
    sub = f"model-{model}_dev-{device}" + ("_hd" if hd else "")
    out_dir = base_out_dir / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _prepare_mask_bw(mask_path: Path, expand_mask: int, adaptive: bool, image_path: Path) -> "object":
    from PIL import Image
    import numpy as np

    if adaptive:
        m = _create_adaptive_mask(image_path, mask_path, int(expand_mask))
        if isinstance(m, tuple):
            m = m[0]
        mask = m
    else:
        mask = Image.open(mask_path).convert("L")
        if int(expand_mask) > 0:
            try:
                from scipy import ndimage

                ma = (np.array(mask, dtype=np.uint8) > 128)
                ma = ndimage.binary_dilation(ma, iterations=int(expand_mask))
                mask = Image.fromarray((ma.astype(np.uint8) * 255), mode="L")
            except Exception:
                pass

    # Ensure strict black/white mask for API
    mask = mask.point(lambda p: 255 if p >= 127 else 0).convert("L")
    return mask


def _prefill_image_with_mask_inpaint(image_path: Path, mask: "object", method: str = "telea") -> "object":
    """使用 OpenCV Inpainting (Telea/NS) 基于边缘填充 mask 区域，保留原图纹理"""
    import cv2
    import numpy as np
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img, dtype=np.uint8)
    mask_np = np.array(mask.convert("L"), dtype=np.uint8)
    # mask 需要是二值图，255 表示需要填充的区域
    mask_binary = (mask_np > 127).astype(np.uint8) * 255

    if method.lower() == "telea":
        inpainted = cv2.inpaint(img_np, mask_binary, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    else:
        inpainted = cv2.inpaint(img_np, mask_binary, inpaintRadius=3, flags=cv2.INPAINT_NS)

    return Image.fromarray(inpainted, mode="RGB")


def _save_montage(original_path: Path, mask_l: "object", result_path: Path, montage_path: Path) -> None:
    from PIL import Image

    orig = Image.open(original_path).convert("RGB")
    res = Image.open(result_path).convert("RGB")
    m = mask_l.convert("L")
    if m.size != orig.size:
        m = m.resize(orig.size, resample=Image.NEAREST)

    # Red overlay where mask is white
    red = Image.new("RGB", orig.size, (255, 0, 0))
    alpha = m.point(lambda p: 160 if p >= 127 else 0).convert("L")
    overlay = Image.composite(red, orig, alpha)
    mask_vis = Image.blend(orig, overlay, 0.6)

    w, h = orig.size
    gap = 8
    canvas = Image.new("RGB", (w * 3 + gap * 2, h), (255, 255, 255))
    canvas.paste(orig, (0, 0))
    canvas.paste(mask_vis, (w + gap, 0))
    canvas.paste(res, (w * 2 + gap * 2, 0))
    montage_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(montage_path)


def _save_debug_bundle(
    *,
    original_path: Path,
    mask_l: "object",
    result_path: Path,
    debug_dir: Path,
    prefilled: "object" = None,
) -> None:
    from PIL import Image

    debug_dir.mkdir(parents=True, exist_ok=True)

    orig = Image.open(original_path).convert("RGB")
    res = Image.open(result_path).convert("RGB")
    m = mask_l.convert("L")
    if m.size != orig.size:
        m = m.resize(orig.size, resample=Image.NEAREST)

    # raw assets
    orig.save(debug_dir / "original.png")
    m.save(debug_dir / "mask.png")
    res.save(debug_dir / "result.png")

    # overlay visualization
    red = Image.new("RGB", orig.size, (255, 0, 0))
    alpha = m.point(lambda p: 160 if p >= 127 else 0).convert("L")
    overlay = Image.composite(red, orig, alpha)
    mask_vis = Image.blend(orig, overlay, 0.6)
    mask_vis.save(debug_dir / "mask_vis.png")

    if prefilled is not None:
        try:
            prefilled.convert("RGB").save(debug_dir / "prefilled.png")
        except Exception:
            pass


def _flux_fill_api_inpaint(
    image_path: Path,
    mask_path: Path,
    output_path: Path,
    *,
    expand_mask: int = 0,
    adaptive: bool = False,
    save_masked: bool = False,
    save_debug: bool = False,
    prompt: str = "",
    steps: int = 50,
    guidance: float = 1.5,
    safety_tolerance: int = 2,
    output_format: str = "png",
    poll_interval: float = 0.5,
    poll_timeout: float = 300.0,
    endpoint: str = "https://api.bfl.ai/v1/flux-pro-1.0-fill",
) -> None:
    try:
        import requests
    except Exception as e:
        raise RuntimeError(f"requests is required for flux_fill_api backend ({e}). Install: pip install requests")

    api_key = str(os.environ.get("BFL_API_KEY", "")).strip()
    if not api_key:
        raise RuntimeError("BFL_API_KEY env var is required for flux_fill_api backend")

    mask_bw = _prepare_mask_bw(mask_path=mask_path, expand_mask=int(expand_mask), adaptive=bool(adaptive), image_path=image_path)

    # Prefill the image to hide original text from the model
    prefilled_img = _prefill_image_with_mask(image_path, mask_bw, fill_color=(255, 255, 255))

    import io
    if not prompt:
        prompt = (
            "Clean technical diagram background. "
            "Seamlessly fill the masked area with the surrounding color, texture, and patterns. "
            "Maintain the original geometric structure, lines, and shading. "
            "Strictly NO text, NO letters, NO words, NO symbols, NO characters."
        )
    
    # Use prefilled image bytes
    img_buf = io.BytesIO()
    prefilled_img.save(img_buf, format="PNG")
    image_bytes = img_buf.getvalue()
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    
    mask_buf = io.BytesIO()
    mask_bw.save(mask_buf, format="PNG")
    mask_bytes = mask_buf.getvalue()
    mask_b64 = base64.b64encode(mask_bytes).decode("utf-8")

    payload = {
        "prompt": str(prompt),
        "image": img_b64,
        "mask": mask_b64,
        "steps": int(steps),
        "guidance": float(guidance),
        "output_format": str(output_format),
        "safety_tolerance": int(safety_tolerance),
    }

    headers = {
        "x-key": api_key,
        "Content-Type": "application/json",
        "accept": "application/json",
    }

    r = requests.post(str(endpoint), headers=headers, json=payload, timeout=120)
    if r.status_code not in (200, 202):
        raise RuntimeError(f"FLUX Fill request failed: {r.status_code} - {r.text}")
    try:
        j = r.json()
    except Exception:
        raise RuntimeError(f"FLUX Fill request returned non-JSON: {r.text}")

    polling_url = j.get("polling_url") or j.get("pollingUrl") or j.get("url")
    if not polling_url:
        raise RuntimeError(f"FLUX Fill response missing polling_url: keys={list(j.keys())}")

    t0 = time.time()
    last_status = ""
    while True:
        if (time.time() - t0) > float(poll_timeout):
            raise RuntimeError(f"FLUX Fill poll timeout after {poll_timeout}s (last_status={last_status})")
        time.sleep(float(max(0.1, poll_interval)))

        pr = requests.get(str(polling_url), headers={"x-key": api_key, "accept": "application/json"}, timeout=60)
        if pr.status_code != 200:
            continue
        try:
            pj = pr.json()
        except Exception:
            continue
        status = str(pj.get("status", ""))
        if status and status != last_status:
            print(f"  flux poll status: {status}")
            last_status = status

        if status in {"Ready", "Succeeded", "Success"}:
            res = pj.get("result") or {}
            sample = None
            if isinstance(res, dict):
                sample = res.get("sample") or res.get("output")
            if not sample:
                sample = pj.get("sample")
            if not sample:
                raise RuntimeError(f"FLUX Fill ready but missing result.sample: keys={list(pj.keys())}")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            dr = requests.get(str(sample), timeout=120)
            if dr.status_code != 200:
                raise RuntimeError(f"FLUX Fill download failed: {dr.status_code}")
            output_path.write_bytes(dr.content)
            if save_debug:
                try:
                    dbg = output_path.parent / "_debug" / output_path.stem
                    _save_debug_bundle(
                        original_path=image_path,
                        mask_l=mask_bw,
                        result_path=output_path,
                        debug_dir=dbg,
                        prefilled=prefilled_img,
                    )
                    print(f"  saved debug bundle: {dbg}")
                except Exception as e:
                    print(f"  debug bundle failed: {e}")
            return
        if status in {"Error", "Failed", "Failure"}:
            raise RuntimeError(f"FLUX Fill failed: {pj}")


def _get_flux_fill_local_pipe(*, model_dir: str, device: str, lora_path: str, lora_scale: float) -> "object":
    key = (str(model_dir), str(lora_path), str(device))
    cached = _FLUX_LOCAL_PIPE_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        import torch
        import gc
        from diffusers import FluxFillPipeline
    except Exception as e:
        raise RuntimeError(
            f"diffusers/torch is required for flux_fill_local backend ({e}). Install: pip install diffusers transformers accelerate peft safetensors"
        )

    def _mem_info(stage: str):
        try:
            import psutil
            vm = psutil.virtual_memory()
            print(f"  [MEM] {stage}: RAM used={vm.used/1e9:.2f}GB total={vm.total/1e9:.2f}GB available={vm.available/1e9:.2f}GB")
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    mem_alloc = torch.cuda.memory_allocated(i) / 1e9
                    mem_reserved = torch.cuda.memory_reserved(i) / 1e9
                    print(f"  [MEM] {stage}: GPU:{i} alloc={mem_alloc:.2f}GB reserved={mem_reserved:.2f}GB")
        except Exception:
            pass

    _mem_info("Before pipeline loading")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    torch_dtype = torch.float16
    # CRITICAL: Use device_map to load directly to GPU, avoiding CPU RAM spike.
    # But we need to handle LoRA carefully to avoid NaN latents.
    print(f"  [DEBUG] Loading FluxFillPipeline with device_map='cuda', dtype={torch_dtype}")
    
    pipe = FluxFillPipeline.from_pretrained(
        str(model_dir),
        torch_dtype=torch_dtype,
        device_map="cuda",
    )
    _mem_info("After from_pretrained (device_map='cuda')")

    dev = (device or "").lower()
    use_cuda = dev in {"gpu", "cuda"}

    # NOTE: With device_map='cuda', the pipeline is already on GPU.
    # No need to call pipe.to("cuda") again.
    print(f"  [DEBUG] Pipeline loaded via device_map='cuda', use_cuda={use_cuda}")
    
    _mem_info("After device_map setup")

    print(f"  [DEBUG] Enabling memory optimizations (attention slicing, vae slicing/tiling)")
    try:
        pipe.enable_attention_slicing()
        print(f"  [DEBUG] Attention slicing enabled")
    except Exception as e:
        print(f"  [WARN] attention_slicing failed: {e}")
    try:
        pipe.enable_vae_slicing()
        print(f"  [DEBUG] VAE slicing enabled")
    except Exception as e:
        print(f"  [WARN] vae_slicing failed: {e}")
    try:
        pipe.enable_vae_tiling()
        print(f"  [DEBUG] VAE tiling enabled")
    except Exception as e:
        print(f"  [WARN] vae_tiling failed: {e}")

    _mem_info("After optimizations")

    lp = str(lora_path or "").strip()
    if lp:
        print(f"  [DEBUG] Loading LoRA weights from {lp}")
        try:
            # Load LoRA with same dtype as base model to avoid dtype mismatch
            pipe.load_lora_weights(lp, adapter_name="removal_lora")
            print(f"  [DEBUG] LoRA weights loaded successfully as adapter 'removal_lora'")
        except Exception as e:
            raise RuntimeError(f"failed to load LoRA weights: {lp} ({e})")

        scale = float(lora_scale)
        print(f"  [DEBUG] Setting LoRA adapter 'removal_lora' with scale={scale}")
        
        # CRITICAL: Ensure LoRA weights are on GPU and in float16 to avoid NaN
        if hasattr(pipe, 'transformer') and hasattr(pipe.transformer, 'lora_layers'):
            print(f"  [DEBUG] Checking LoRA layer devices and dtypes...")
            for name, layer in pipe.transformer.lora_layers.items():
                for param_name, param in layer.named_parameters():
                    if param.device.type != 'cuda':
                        print(f"  [WARN] Moving LoRA param {name}.{param_name} from {param.device} to cuda")
                        param.data = param.data.cuda()
                    if param.dtype != torch.float16:
                        print(f"  [WARN] Converting LoRA param {name}.{param_name} from {param.dtype} to float16")
                        param.data = param.data.to(torch.float16)
            print(f"  [DEBUG] LoRA layer check complete")
        
        # Use set_adapters to activate LoRA without fusing (fusing can cause NaN with device_map)
        try:
            # Check what adapters are available
            if hasattr(pipe, 'get_active_adapters'):
                active = pipe.get_active_adapters()
                print(f"  [DEBUG] Active adapters before: {active}")
            
            pipe.set_adapters(["removal_lora"], adapter_weights=[scale])
            print(f"  [DEBUG] LoRA adapter 'removal_lora' activated with scale={scale}")
            
            if hasattr(pipe, 'get_active_adapters'):
                active = pipe.get_active_adapters()
                print(f"  [DEBUG] Active adapters after: {active}")
        except Exception as e:
            print(f"  [WARN] set_adapters failed: {e}, trying fuse_lora")
            try:
                pipe.fuse_lora(lora_scale=scale)
                print(f"  [DEBUG] LoRA fused with scale={scale}")
            except Exception as e2:
                print(f"  [WARN] fuse_lora also failed: {e2}")
        
        _mem_info("After LoRA loading")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # CRITICAL: Clear PyTorch CPU memory pool to reduce RSS
    try:
        if hasattr(torch, '_C') and hasattr(torch._C, '_host_emptyCache'):
            torch._C._host_emptyCache()
            print(f"  [DEBUG] PyTorch CPU cache cleared via _host_emptyCache")
    except Exception as e:
        print(f"  [DEBUG] _host_emptyCache not available: {e}")
    
    # CRITICAL: Force release temporary CPU memory that may be held by Python allocator
    # This prevents accumulation of CPU RAM across multiple pipeline loads.
    try:
        import ctypes
        ctypes.CDLL('libc.so.6').malloc_trim(0)
        print(f"  [DEBUG] malloc_trim called to release CPU memory to OS")
    except Exception as e:
        print(f"  [DEBUG] malloc_trim not available: {e}")
    
    # Print detailed memory breakdown
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        print(f"  [MEM] Process RSS: {mem_info.rss/1e9:.2f}GB, VMS: {mem_info.vms/1e9:.2f}GB")
        
        # Check system caches
        vm = psutil.virtual_memory()
        if hasattr(vm, 'buffers') and hasattr(vm, 'cached'):
            print(f"  [MEM] System buffers: {vm.buffers/1e9:.2f}GB, cached: {vm.cached/1e9:.2f}GB")
    except Exception as e:
        print(f"  [DEBUG] Detailed memory check failed: {e}")
    
    _mem_info("After explicit memory cleanup")
    
    _mem_info("Before caching pipe")
    _FLUX_LOCAL_PIPE_CACHE[key] = pipe
    _mem_info("After caching pipe - initialization complete")
    print(f"  [DEBUG] FluxFillPipeline initialized and cached successfully")
    return pipe


def _get_sdxl_inpaint_local_pipe(*, model: str, device: str) -> "object":
    key = (str(model), str(device))
    cached = _SDXL_LOCAL_PIPE_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        import torch
        from diffusers import StableDiffusionXLInpaintPipeline
    except Exception as e:
        raise RuntimeError(
            f"diffusers/torch is required for sdxl_inpaint_local backend ({e}). Install: pip install diffusers transformers accelerate safetensors"
        )

    dev = (device or "").lower()
    use_cuda = dev in {"gpu", "cuda"}
    torch_dtype = torch.float16 if use_cuda else torch.float32

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(str(model), torch_dtype=torch_dtype)
    if hasattr(pipe, "safety_checker"):
        try:
            pipe.safety_checker = None
        except Exception:
            pass

    if use_cuda:
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")

    try:
        pipe.enable_model_cpu_offload()
    except Exception:
        pass
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass
    try:
        pipe.enable_vae_tiling()
    except Exception:
        pass

    _SDXL_LOCAL_PIPE_CACHE[key] = pipe
    return pipe


def _sdxl_inpaint_local_inpaint(
    image_path: Path,
    mask_path: Path,
    output_path: Path,
    *,
    expand_mask: int = 0,
    adaptive: bool = False,
    save_masked: bool = False,
    save_debug: bool = False,
    prompt: str = "",
    negative_prompt: str = "",
    steps: int = 30,
    guidance: float = 6.0,
    strength: float = 0.65,
    model: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    device: str = "gpu",
    prefill_mode: str = "telea",
) -> None:
    import torch
    import numpy as np
    from PIL import Image

    mask_bw = _prepare_mask_bw(mask_path=mask_path, expand_mask=int(expand_mask), adaptive=bool(adaptive), image_path=image_path)
    
    # 根据 prefill_mode 选择填充策略：
    # - white: 纯白填充（技术图表，需要白底）
    # - telea: 边缘填充（自然图片，需要纹理线索）
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    
    if prefill_mode == "telea":
        prefilled_img = _prefill_image_with_mask_inpaint(image_path, mask_bw, method="telea")
    elif prefill_mode == "original":
        # 不使用 prefill，直接用原图（SDXL 会看到 mask 并自己决定填充什么）
        prefilled_img = img
    else:
        # 默认纯白填充
        white = Image.new("RGB", img.size, (255, 255, 255))
        prefilled_img = Image.composite(white, img, mask_bw)

    # SDXL 需要输入尺寸对齐到 8 的倍数 (VAE 下采样 8x)
    # 使用 padding 对齐，推理后再裁剪回原始尺寸，保持比例
    import math
    ALIGN = 64  # SDXL 通常使用 64 对齐以获得最佳效果
    target_w = int(math.ceil(orig_w / ALIGN) * ALIGN)
    target_h = int(math.ceil(orig_h / ALIGN) * ALIGN)
    pad_w = target_w - orig_w
    pad_h = target_h - orig_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    pad_right = pad_w - pad_left
    pad_bottom = pad_h - pad_top

    if target_w != orig_w or target_h != orig_h:
        new_img = Image.new("RGB", (target_w, target_h), (255, 255, 255))
        new_img.paste(prefilled_img, (pad_left, pad_top))
        prefilled_img = new_img

        new_mask = Image.new("L", (target_w, target_h), 0)
        new_mask.paste(mask_bw, (pad_left, pad_top))
        mask_bw = new_mask

        print(f"  [DEBUG] SDXL padded input: orig=({orig_w},{orig_h}) target=({target_w},{target_h}) pad_left={pad_left} pad_top={pad_top}")

    if not prompt:
        prompt = (
            "Clean technical diagram background. "
            "Seamlessly fill the masked area with the surrounding color and texture. "
            "Maintain the original geometric structure and lines. "
            "Strictly NO text, NO letters, NO words, NO symbols, NO characters."
        )
    if not negative_prompt:
        negative_prompt = "text, letters, words, watermark, signature, logo, symbols, artifacts, blurry"

    pipe = _get_sdxl_inpaint_local_pipe(model=str(model), device=str(device))

    device_type = "cuda" if ("cuda" in str(device).lower() or "gpu" in str(device).lower()) else "cpu"
    amp_dtype = torch.float16 if device_type == "cuda" else None

    with torch.inference_mode():
        if device_type == "cuda":
            autocast_ctx = torch.autocast(device_type=device_type, dtype=amp_dtype)
        else:
            autocast_ctx = torch.autocast(device_type=device_type, enabled=False)

        with autocast_ctx:
            out = pipe(
                prompt=str(prompt),
                negative_prompt=str(negative_prompt or ""),
                image=prefilled_img,
                mask_image=mask_bw,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                strength=float(strength),
            )

    img = out.images[0] if hasattr(out, "images") else out[0]
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.array(img))

    # 如果进行了 padding，裁剪回原始尺寸
    if pad_w or pad_h:
        # 计算缩放比例（SDXL 输出可能与输入尺寸不同）
        out_w, out_h = img.size
        scale_x = out_w / target_w
        scale_y = out_h / target_h

        # 计算裁剪区域
        left = int(round(pad_left * scale_x))
        top = int(round(pad_top * scale_y))
        right = int(round((pad_left + orig_w) * scale_x))
        bottom = int(round((pad_top + orig_h) * scale_y))

        # 确保在有效范围内
        left = max(0, min(left, out_w))
        top = max(0, min(top, out_h))
        right = max(0, min(right, out_w))
        bottom = max(0, min(bottom, out_h))

        img = img.crop((left, top, right, bottom))

        # 如果裁剪后尺寸仍不匹配，强制 resize
        if img.size != (orig_w, orig_h):
            img = img.resize((orig_w, orig_h), Image.LANCZOS)
            print(f"  [DEBUG] SDXL resized output to original size: ({orig_w},{orig_h})")
        else:
            print(f"  [DEBUG] SDXL cropped output to original size: ({orig_w},{orig_h})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)

    if save_debug:
        try:
            dbg = output_path.parent / "_debug" / output_path.stem
            _save_debug_bundle(
                original_path=image_path,
                mask_l=mask_bw,
                result_path=output_path,
                debug_dir=dbg,
                prefilled=prefilled_img,
            )
            print(f"  saved debug bundle: {dbg}")
        except Exception as e:
            print(f"  debug bundle failed: {e}")


def _flux_fill_local_inpaint(
    image_path: Path,
    mask_path: Path,
    output_path: Path,
    *,
    expand_mask: int = 0,
    adaptive: bool = False,
    save_masked: bool = False,
    save_debug: bool = False,
    prompt: str = "",
    steps: int = 50,
    guidance: float = 1.5,
    model_dir: str = "",
    lora_path: str = "",
    lora_scale: float = 0.9,
    device: str = "gpu",
) -> None:
    import torch
    import numpy as np
    from PIL import Image
    import math

    mask_bw = _prepare_mask_bw(mask_path=mask_path, expand_mask=int(expand_mask), adaptive=bool(adaptive), image_path=image_path)
    mask_bw_full = mask_bw
    # 使用 OpenCV Telea 边缘填充保留原图纹理，避免纯白填充导致的过度生成
    prefilled_img = _prefill_image_with_mask_inpaint(image_path, mask_bw, method="telea")
    prefilled_full = prefilled_img

    # Patch-based inpaint to avoid global blur:
    # FLUX decode is typically 1024px; scaling the whole image back to a larger size makes everything look soft.
    # For technical diagrams, we inpaint only around the mask bbox (with margin), then paste back to the original.
    patch_x0 = patch_y0 = 0
    patch_x1 = patch_y1 = 0
    use_patch = False
    try:
        m0 = mask_bw
        if m0.size != prefilled_img.size:
            m0 = m0.resize(prefilled_img.size, resample=Image.NEAREST)
        m_np = (np.array(m0, dtype=np.uint8) > 127)
        if np.any(m_np):
            ys, xs = np.where(m_np)
            x0 = int(xs.min())
            x1 = int(xs.max()) + 1
            y0 = int(ys.min())
            y1 = int(ys.max()) + 1

            orig_w0, orig_h0 = prefilled_img.size
            mask_area = float((x1 - x0) * (y1 - y0))
            img_area = float(orig_w0 * orig_h0)
            max_side = int(max(orig_w0, orig_h0))

            # Heuristic: only enable on large images where full-frame scaling is visible,
            # and when the mask is not covering most of the canvas.
            if max_side > 1100 and (mask_area / max(1.0, img_area)) < 0.60:
                margin = 128
                patch_x0 = max(0, x0 - margin)
                patch_y0 = max(0, y0 - margin)
                patch_x1 = min(orig_w0, x1 + margin)
                patch_y1 = min(orig_h0, y1 + margin)
                if (patch_x1 - patch_x0) >= 64 and (patch_y1 - patch_y0) >= 64:
                    use_patch = True
    except Exception:
        use_patch = False

    if use_patch:
        prefilled_img = prefilled_img.crop((patch_x0, patch_y0, patch_x1, patch_y1))
        mask_bw = mask_bw.crop((patch_x0, patch_y0, patch_x1, patch_y1))

    # FLUX packed latents assume a grid that is aligned to a fixed stride (effectively 32px at the pixel level).
    # For non-square inputs, Flux may pad internally; when we decode latents manually, we must match that padding.
    # To make this deterministic, we pad image+mask to a square size aligned to 32, then crop back.
    # NOTE: Use a moderate canvas size (1024 or 64-aligned) to preserve fine details and avoid over-generation.
    orig_w, orig_h = prefilled_img.size
    pad_left = pad_top = 0
    pad_w = pad_h = 0
    # Align to nearest 64 to satisfy FLUX pack factor (2) and VAE scale (8): 2*8=16, but 64 gives headroom.
    ALIGN = 64
    target_w = int(math.ceil(orig_w / ALIGN) * ALIGN)
    target_h = int(math.ceil(orig_h / ALIGN) * ALIGN)
    # Keep original aspect ratio; pad separately for w and h
    if target_w != orig_w or target_h != orig_h:
        pad_w = target_w - orig_w
        pad_h = target_h - orig_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        pad_right = pad_w - pad_left
        pad_bottom = pad_h - pad_top

        new_img = Image.new("RGB", (target_w, target_h), (255, 255, 255))
        new_img.paste(prefilled_img, (pad_left, pad_top))
        prefilled_img = new_img

        new_mask = Image.new("L", (target_w, target_h), 0)
        new_mask.paste(mask_bw, (pad_left, pad_top))
        mask_bw = new_mask

        print(
            f"  [DEBUG] padded input: orig=({orig_w},{orig_h}) target=({target_w},{target_h}) pad_left={pad_left} pad_top={pad_top}"
        )

    if not prompt:
        prompt = (
            "Clean technical diagram background. "
            "Seamlessly fill the masked area with the surrounding color, texture, and patterns. "
            "Maintain the original geometric structure, lines, and shading. "
            "Strictly NO text, NO letters, NO words, NO symbols, NO characters."
        )

    if not model_dir:
        model_dir = "/home/wangyankun/models/flux_fill"
    if not lora_path:
        lora_path = "/home/wangyankun/models/loras/removalV2.safetensors"

    pipe = _get_flux_fill_local_pipe(
        model_dir=str(model_dir),
        device=str(device),
        lora_path=str(lora_path),
        lora_scale=float(lora_scale),
    )

    # Keep VAE in half precision to reduce peak memory. We already sanitize NaNs/INFs.
    # For debugging stability issues, you can temporarily switch this back to float32.

    device_type = "cuda" if "cuda" in str(device).lower() or "gpu" in str(device).lower() else "cpu"
    # Keep AMP dtype consistent with pipeline weights (we load FLUX/LoRA in float16).
    # Using bfloat16 here can trigger dtype mismatches like: BFloat16 input vs Half bias.
    amp_dtype = torch.float16 if device_type == "cuda" else None
    
    with torch.inference_mode():
        if device_type == "cuda":
            autocast_ctx = torch.autocast(device_type=device_type, dtype=amp_dtype)
        else:
            autocast_ctx = torch.autocast(device_type=device_type, enabled=False)

        with autocast_ctx:
            # Step 1: Run inference getting latents as output
            print(f"  [DEBUG] Starting inference: prompt='{prompt[:50]}...' steps={steps} guidance={guidance}")
            out = pipe(
                prompt=str(prompt),
                image=prefilled_img,
                mask_image=mask_bw,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                output_type="latent",
            )

        latents = out.images if hasattr(out, "images") else out[0]
        try:
            del out
        except Exception:
            pass
        if device_type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        print(f"  [DEBUG] Latents shape: {latents.shape}, dtype: {latents.dtype}")

        # Diagnose NaN/Inf prevalence. If a large fraction of latents is invalid, the decoded image becomes gray.
        latents_f = latents.float()
        nan_mask = torch.isnan(latents_f)
        inf_mask = torch.isinf(latents_f)
        nan_count = int(nan_mask.sum().item())
        inf_count = int(inf_mask.sum().item())
        total = int(latents_f.numel())
        if nan_count or inf_count:
            frac = float(nan_count + inf_count) / float(max(1, total))
            print(f"  [WARN] invalid latents detected: nan={nan_count} inf={inf_count} total={total} frac={frac:.6f}")

        try:
            finite = latents_f[~(nan_mask | inf_mask)]
            if finite.numel() > 0:
                print(
                    f"  [DEBUG] Latents finite stats: min={finite.min().item():.6f} max={finite.max().item():.6f} mean={finite.mean().item():.6f}"
                )
        except Exception:
            pass

        if nan_count or inf_count:
            print("  [WARN] Cleaning invalid latents with nan_to_num")
            latents = torch.nan_to_num(latents, nan=0.0, posinf=0.0, neginf=0.0)

        # Step 2: Manual VAE decode
        # FLUX uses a Flow-Transformer, its latents are (batch, seq_len, channels)
        # We need to unpack them back to (batch, channels, height, width) before VAE decode
        # Keep half precision to reduce memory.
        if device_type == "cuda":
            # Force float16 on CUDA to match VAE weights/bias.
            if latents.dtype != torch.float16:
                latents = latents.to(dtype=torch.float16)
        else:
            if latents.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                latents = latents.to(dtype=torch.float32)
        
        print(f"  [DEBUG] Unpacking latents: shape={latents.shape}")
        
        # NOTE: use padded size for unpacking; we will crop back to original size later.
        scaling_factor = getattr(pipe.vae.config, "scaling_factor", 0.3611)
        shift_factor = getattr(pipe.vae.config, "shift_factor", 0.1159)

        # Deterministic unpack based only on packed latents shape.
        # packed latents: [B, L, 64] where 64 = 16 (vae channels) * 2 * 2 (pack factor)
        batch_size, seq_len, channels = latents.shape
        if channels != 64:
            raise RuntimeError(f"unexpected packed latent channels={channels}, expected 64")

        grid = int(math.isqrt(int(seq_len)))
        if grid * grid != int(seq_len):
            raise RuntimeError(f"packed seq_len={seq_len} is not a perfect square; cannot deterministically unpack")

        # Unpack: [B, L, 64] -> [B, grid, grid, 16, 2, 2] -> [B, 16, grid*2, grid*2]
        latents_vae = latents.view(batch_size, grid, grid, 16, 2, 2)
        latents_vae = latents_vae.permute(0, 3, 1, 4, 2, 5).reshape(batch_size, 16, grid * 2, grid * 2)
        latents_vae = (latents_vae / scaling_factor) + shift_factor

        if device_type == "cuda" and latents_vae.dtype != torch.float16:
            latents_vae = latents_vae.to(dtype=torch.float16)

        print(f"  [DEBUG] Final VAE input shape (deterministic): {latents_vae.shape}")
        vae_output = pipe.vae.decode(latents_vae, return_dict=False)[0]
        try:
            del latents_vae
        except Exception:
            pass
        if device_type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        print(f"  [DEBUG] VAE output shape: {vae_output.shape}, dtype: {vae_output.dtype}")


        
        # Step 3: Map to [0, 1] and handle NaNs
        image_tensor = (vae_output / 2 + 0.5).clamp(0, 1)

        # Safety: if decoding produced padded spatial size, crop/resize back to original.
        # This avoids aspect ratio changes when padding is present.
        # First crop away padding to get back to original aspect.
        target_w = int(math.ceil(orig_w / 64) * 64)
        target_h = int(math.ceil(orig_h / 64) * 64)
        if pad_w or pad_h:
            # `pad_top/pad_left/orig_h/orig_w` are in the padded input canvas coordinate system.
            # The decoded image might have a different resolution, so we must scale coordinates.
            _, _, dec_h, dec_w = image_tensor.shape
            sy = float(dec_h) / float(target_h)
            sx = float(dec_w) / float(target_w)

            top = int(round(pad_top * sy))
            left = int(round(pad_left * sx))
            h = int(round(orig_h * sy))
            w = int(round(orig_w * sx))

            # Clamp to valid range.
            top = max(0, min(top, dec_h))
            left = max(0, min(left, dec_w))
            bottom = max(0, min(top + h, dec_h))
            right = max(0, min(left + w, dec_w))

            image_tensor = image_tensor[:, :, top:bottom, left:right]

        # Safety: enforce final size equals original.
        _, _, out_h, out_w = image_tensor.shape
        if out_h != orig_h or out_w != orig_w:
            image_tensor = torch.nn.functional.interpolate(
                image_tensor,
                size=(int(orig_h), int(orig_w)),
                mode="bilinear",
                align_corners=False,
            )
        
        nan_mask = torch.isnan(image_tensor)
        if nan_mask.any():
            nan_count = torch.sum(nan_mask).item()
            print(f"  [WARN] {nan_count} NaNs detected in VAE output! Replacing with 0")
            image_tensor = torch.nan_to_num(image_tensor, nan=0.0)

        # Step 4: Convert to PIL
        image_np = image_tensor.cpu().permute(0, 2, 3, 1).float().numpy()
        
        # Final check on pixel values before uint8 conversion
        p_min, p_max = image_np.min(), image_np.max()
        p_mean = image_np.mean()
        print(f"  [DEBUG] Pixel stats: min={p_min:.4f}, max={p_max:.4f}, mean={p_mean:.4f}")
        
        if p_max <= 0.0:
            print("  [ERROR] Output image is completely black (max value 0)!")

        image_np = (image_np * 255).round().astype(np.uint8)
        res_img = Image.fromarray(image_np[0])

    mask_comp = mask_bw
    if pad_w or pad_h:
        mask_comp = mask_comp.crop((pad_left, pad_top, pad_left + orig_w, pad_top + orig_h))
    if mask_comp.size != (orig_w, orig_h):
        mask_comp = mask_comp.resize((orig_w, orig_h), resample=Image.NEAREST)

    from PIL import ImageFilter

    orig_full = Image.open(image_path).convert("RGB")
    if use_patch:
        orig_img = orig_full.crop((patch_x0, patch_y0, patch_x1, patch_y1))
    else:
        orig_img = orig_full
    if orig_img.size != (orig_w, orig_h):
        orig_img = orig_img.resize((orig_w, orig_h), resample=Image.LANCZOS)

    if res_img.size != (orig_w, orig_h):
        res_img = res_img.resize((orig_w, orig_h), resample=Image.LANCZOS)

    # FLUX 解码结果通常在 1024 量级；放大回原图会变软。
    # 对结果进行适度锐化（只会在 mask 区域生效，因为最终会按 mask 与原图合成）
    try:
        res_img = res_img.filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=3))
    except Exception:
        pass
    res_img = Image.composite(res_img, orig_img, mask_comp)

    if use_patch:
        final_img = orig_full.copy()
        final_img.paste(res_img, (int(patch_x0), int(patch_y0)))
    else:
        final_img = res_img

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_img.save(output_path)

    if save_debug:
        try:
            dbg = output_path.parent / "_debug" / output_path.stem
            _save_debug_bundle(
                original_path=image_path,
                mask_l=mask_bw,
                result_path=output_path,
                debug_dir=dbg,
                prefilled=prefilled_img,
            )
            print(f"  saved debug bundle: {dbg}")
        except Exception as e:
            print(f"  debug bundle failed: {e}")


def _estimate_auto_expand(
    image_path: Path,
    mask_path: Path,
    max_expand: int,
    delta_abs: float,
    min_expand: int,
) -> int:
    from PIL import Image
    import numpy as np
    from scipy import ndimage

    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    img = np.array(image, dtype=np.float32)
    m = (np.array(mask, dtype=np.uint8) > 128)
    if not np.any(m):
        return int(min_expand)

    gray = img.mean(axis=2)
    local_mean = ndimage.uniform_filter(gray, size=5, mode="reflect")
    local_mean2 = ndimage.uniform_filter(gray * gray, size=5, mode="reflect")
    local_var = np.maximum(local_mean2 - local_mean * local_mean, 0.0)
    complexity = np.sqrt(local_var)

    distance_map = ndimage.distance_transform_edt(~m)

    prev = None
    stable = 0
    chosen = int(min_expand)
    max_expand = int(max(1, max_expand))

    for d in range(1, max_expand + 1):
        ring = (distance_map >= d - 1) & (distance_map < d)
        if not np.any(ring):
            continue
        cur = float(np.mean(complexity[ring]))
        if prev is not None:
            if abs(cur - prev) >= float(delta_abs):
                chosen = d
                break
            stable += 1
        prev = cur
        chosen = d

    lo = int(min_expand)
    hi = int(max_expand)
    if hi < lo:
        hi = lo
    if chosen < lo:
        return lo
    if chosen > hi:
        return hi
    return int(chosen)


def _save_auto_expand_debug_vis(
    image_path: Path,
    mask_path: Path,
    out_path: Path,
    chosen_expand: int,
    complexity: "object",
    distance_map: "object",
) -> None:
    from PIL import Image
    import numpy as np

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as im:
        im = im.convert("RGB")
        w, h = im.size

    with Image.open(mask_path) as m:
        m = m.convert("L")
        mask_u8 = np.array(m, dtype=np.uint8)

    comp = np.array(complexity, dtype=np.float32)
    if comp.shape[0] != h or comp.shape[1] != w:
        return

    c0 = float(np.percentile(comp, 5))
    c1 = float(np.percentile(comp, 95))
    if c1 <= c0:
        c1 = c0 + 1.0
    comp01 = np.clip((comp - c0) / (c1 - c0), 0.0, 1.0)
    comp_u8 = (comp01 * 255.0).astype(np.uint8)

    dm = np.array(distance_map, dtype=np.float32)
    ring = (dm >= float(max(0, chosen_expand - 1))) & (dm < float(max(1, chosen_expand)))

    img = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    overlay = img.copy()
    overlay[ring] = (255, 0, 0)

    mask_rgb = np.dstack([mask_u8] * 3)
    comp_rgb = np.dstack([comp_u8] * 3)

    top = np.concatenate([img, overlay], axis=1)
    bottom = np.concatenate([mask_rgb, comp_rgb], axis=1)
    canvas = np.concatenate([top, bottom], axis=0)
    Image.fromarray(canvas).save(out_path)


def _save_post_noise_vis(
    image: "object",
    mask: "object",
    out_path: Path,
    *,
    mode: str,
    percentile: float,
    sigma: float,
    edge_exclude: int,
    flat: float,
    kmeans_sep: float,
    kmeans_max_frac: float,
    z_thresh: float,
    min_area: int,
) -> None:
    from PIL import Image
    import numpy as np
    from scipy import ndimage

    img = image.convert("RGB")
    m = mask.convert("L")
    if m.size != img.size:
        m = m.resize(img.size, resample=Image.NEAREST)

    img_np = np.array(img, dtype=np.float32)

    m_np = (np.array(m, dtype=np.uint8) > 128)
    if not np.any(m_np):
        return

    inner = m_np
    ee = int(max(0, edge_exclude))
    if ee > 0:
        try:
            inner = ndimage.binary_erosion(m_np, iterations=ee)
        except Exception:
            inner = m_np
    if not np.any(inner):
        inner = m_np

    def _rgb_to_lab(rgb: "object") -> "object":
        # rgb: float32, 0..255
        import numpy as np

        r = rgb[:, :, 0] / 255.0
        g = rgb[:, :, 1] / 255.0
        b = rgb[:, :, 2] / 255.0

        def inv_gamma(u: "object") -> "object":
            return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)

        r = inv_gamma(r)
        g = inv_gamma(g)
        b = inv_gamma(b)

        # sRGB D65
        X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
        Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
        Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

        # Reference white
        Xn, Yn, Zn = 0.95047, 1.0, 1.08883
        x = X / Xn
        y = Y / Yn
        z = Z / Zn

        eps = 216.0 / 24389.0
        k = 24389.0 / 27.0

        def f(t: "object") -> "object":
            return np.where(t > eps, np.cbrt(t), (k * t + 16.0) / 116.0)

        fx = f(x)
        fy = f(y)
        fz = f(z)

        L = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b2 = 200.0 * (fy - fz)
        return L, a, b2

    mode_s = (mode or "").strip().lower() or "lab_outlier"
    noise = None

    if mode_s == "auto":
        noise = _detect_post_noise_mask(
            image,
            mask,
            mode="auto",
            percentile=float(percentile),
            sigma=float(sigma),
            edge_exclude=int(edge_exclude),
            flat=float(flat),
            kmeans_sep=float(kmeans_sep),
            kmeans_max_frac=float(kmeans_max_frac),
            z_thresh=float(z_thresh),
            min_area=int(min_area),
        )
        overlay = img_np.astype(np.uint8).copy()
        if noise is not None and np.any(noise):
            overlay[noise] = (255, 0, 0)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(out_path)
        return
    # Optional: restrict to flat regions (avoid edges/gradients causing false positives)
    flat_thr = float(flat)
    if flat_thr > 0:
        try:
            gray = img_np.mean(axis=2)
            gx = ndimage.sobel(gray, axis=1, mode="reflect")
            gy = ndimage.sobel(gray, axis=0, mode="reflect")
            gmag = np.sqrt(gx * gx + gy * gy)
            inner = inner & (gmag <= flat_thr)
        except Exception:
            pass

    def _lab_kmeans_noise2(
        aa: "object", bb: "object", inner_m: "object", sep: float, max_frac: float
    ) -> "object":
        import numpy as np

        idx = np.flatnonzero(inner_m)
        if idx.size <= 0:
            return None
        max_n = 6000
        if idx.size > max_n:
            step = int(np.ceil(idx.size / float(max_n)))
            idx_s = idx[::step]
        else:
            idx_s = idx
        pts = np.stack([aa.reshape(-1)[idx_s], bb.reshape(-1)[idx_s]], axis=1).astype(np.float32)
        if pts.shape[0] < 24:
            return None
        c0 = np.median(pts, axis=0)
        d0 = np.sum((pts - c0) ** 2, axis=1)
        c1 = pts[int(np.argmax(d0))]
        for _ in range(12):
            d_to_0 = np.sum((pts - c0) ** 2, axis=1)
            d_to_1 = np.sum((pts - c1) ** 2, axis=1)
            lab = (d_to_1 < d_to_0)
            if not np.any(lab) or np.all(lab):
                break
            c0_new = np.mean(pts[~lab], axis=0)
            c1_new = np.mean(pts[lab], axis=0)
            if float(np.sum((c0_new - c0) ** 2) + np.sum((c1_new - c1) ** 2)) < 1e-4:
                c0, c1 = c0_new, c1_new
                break
            c0, c1 = c0_new, c1_new
        sep2 = float(np.sum((c0 - c1) ** 2))
        if sep2 < float(max(0.0, sep)) ** 2:
            return None

        aa_flat = aa.reshape(-1)[idx]
        bb_flat = bb.reshape(-1)[idx]
        pts_all = np.stack([aa_flat, bb_flat], axis=1).astype(np.float32)
        d_to_0 = np.sum((pts_all - c0) ** 2, axis=1)
        d_to_1 = np.sum((pts_all - c1) ** 2, axis=1)
        lab_all = (d_to_1 < d_to_0)
        n1 = int(np.sum(lab_all))
        n0 = int(lab_all.size - n1)
        if n0 == 0 or n1 == 0:
            return None
        frac1 = float(n1) / float(n0 + n1)
        frac0 = 1.0 - frac1
        # Keep only if the "minor" cluster is truly small; otherwise it's likely just a normal color region.
        mf = float(max_frac)
        if mf > 0:
            if min(frac0, frac1) > mf:
                return None
        if n1 < n0:
            bad = lab_all
        else:
            bad = ~lab_all
        out = np.zeros_like(inner_m, dtype=bool).reshape(-1)
        out[idx] = bad
        return out.reshape(inner_m.shape)

    if mode_s == "resid_gray":
        gray = img_np.mean(axis=2)
        s = float(sigma)
        if s <= 0:
            s = 6.0
        blur = ndimage.gaussian_filter(gray, sigma=s, mode="reflect")
        resid = np.abs(gray - blur)
        vals = resid[inner]
        if vals.size <= 0:
            return
        p = float(percentile)
        if p < 50.0:
            p = 50.0
        if p > 100.0:
            p = 100.0
        thr = float(np.percentile(vals, p))
        thr = max(thr, 1.5)
        noise = (resid >= thr) & inner
    elif mode_s == "lab_resid":
        # Lab residual against a blurred baseline (more stable on gradients)
        _, aa, bb = _rgb_to_lab(img_np)
        s = float(sigma)
        if s <= 0:
            s = 6.0
        aa_blur = ndimage.gaussian_filter(aa, sigma=s, mode="reflect")
        bb_blur = ndimage.gaussian_filter(bb, sigma=s, mode="reflect")
        resid = np.sqrt((aa - aa_blur) ** 2 + (bb - bb_blur) ** 2)
        vals = resid[inner]
        if vals.size <= 0:
            return
        p = float(percentile)
        if p < 50.0:
            p = 50.0
        if p > 100.0:
            p = 100.0
        thr = float(np.percentile(vals, p))
        thr = max(thr, float(max(0.0, z_thresh)))
        noise = (resid >= thr) & inner
    elif mode_s == "lab_kmeans":
        _, aa, bb = _rgb_to_lab(img_np)
        try:
            labeled, num = ndimage.label(inner)
        except Exception:
            labeled, num = (None, 0)
        if labeled is None or num <= 1:
            noise = _lab_kmeans_noise(aa, bb, inner, float(kmeans_sep), float(kmeans_max_frac))
        else:
            acc = np.zeros_like(inner, dtype=bool)
            for cid in range(1, int(num) + 1):
                comp = labeled == cid
                if not np.any(comp):
                    continue
                nm = _lab_kmeans_noise(aa, bb, comp, float(kmeans_sep), float(kmeans_max_frac))
                if nm is not None and np.any(nm):
                    acc |= nm
            noise = acc if np.any(acc) else None
    else:
        # Lab a/b robust outlier detection (targets chroma blotches)
        _, aa, bb = _rgb_to_lab(img_np)
        a_vals = aa[inner]
        b_vals = bb[inner]
        if a_vals.size <= 0 or b_vals.size <= 0:
            return

        med_a = float(np.median(a_vals))
        med_b = float(np.median(b_vals))

        mad_a = float(np.median(np.abs(a_vals - med_a)))
        mad_b = float(np.median(np.abs(b_vals - med_b)))
        # avoid zero
        mad_a = max(mad_a, 1e-3)
        mad_b = max(mad_b, 1e-3)

        za = np.abs(aa - med_a) / mad_a
        zb = np.abs(bb - med_b) / mad_b
        score = np.sqrt(za * za + zb * zb)
        svals = score[inner]
        if svals.size <= 0:
            return

        zt = float(z_thresh)
        if zt <= 0:
            zt = 6.0

        # Optional percentile gate: keeps only the most extreme outliers
        p = float(percentile)
        if p < 50.0:
            p = 50.0
        if p > 100.0:
            p = 100.0
        thr_p = float(np.percentile(svals, p))
        thr = max(zt, thr_p)
        noise = (score >= thr) & inner

    if noise is not None:
        # Clean: open then remove tiny blobs
        try:
            noise = ndimage.binary_opening(noise, structure=np.ones((3, 3)), iterations=1)
        except Exception:
            pass
        labeled, num = ndimage.label(noise)
        if num > 0 and int(min_area) > 1:
            keep = np.zeros_like(noise, dtype=bool)
            for i in range(1, num + 1):
                comp = labeled == i
                if int(np.sum(comp)) >= int(min_area):
                    keep |= comp
            noise = keep

    overlay = img_np.astype(np.uint8).copy()
    if noise is not None and np.any(noise):
        overlay[noise] = (255, 0, 0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(out_path)


def _detect_post_noise_mask(
    image: "object",
    mask: "object",
    *,
    mode: str,
    percentile: float,
    sigma: float,
    edge_exclude: int,
    flat: float,
    kmeans_sep: float,
    kmeans_max_frac: float,
    z_thresh: float,
    min_area: int,
) -> "object":
    from PIL import Image
    import numpy as np
    from scipy import ndimage

    img = image.convert("RGB")
    m = mask.convert("L")
    if m.size != img.size:
        m = m.resize(img.size, resample=Image.NEAREST)

    m_np = (np.array(m, dtype=np.uint8) > 128)
    if not np.any(m_np):
        return None

    inner = m_np
    ee = int(max(0, edge_exclude))
    if ee > 0:
        try:
            inner = ndimage.binary_erosion(m_np, iterations=ee)
        except Exception:
            inner = m_np
    if not np.any(inner):
        inner = m_np

    img_np = np.array(img, dtype=np.float32)

    # Reuse the visualization function but return the boolean mask.
    # We inline the core to avoid disk I/O.
    mode_s = (mode or "").strip().lower() or "lab_outlier"
    noise = None

    def _lab_kmeans_noise2(
        aa: "object", bb: "object", inner_m: "object", sep: float, max_frac: float
    ) -> "object":
        import numpy as np

        idx = np.flatnonzero(inner_m)
        if idx.size <= 0:
            return None
        max_n = 6000
        if idx.size > max_n:
            step = int(np.ceil(idx.size / float(max_n)))
            idx_s = idx[::step]
        else:
            idx_s = idx
        pts = np.stack([aa.reshape(-1)[idx_s], bb.reshape(-1)[idx_s]], axis=1).astype(np.float32)
        if pts.shape[0] < 24:
            return None
        c0 = np.median(pts, axis=0)
        d0 = np.sum((pts - c0) ** 2, axis=1)
        c1 = pts[int(np.argmax(d0))]
        for _ in range(12):
            d_to_0 = np.sum((pts - c0) ** 2, axis=1)
            d_to_1 = np.sum((pts - c1) ** 2, axis=1)
            lab = (d_to_1 < d_to_0)
            if not np.any(lab) or np.all(lab):
                break
            c0_new = np.mean(pts[~lab], axis=0)
            c1_new = np.mean(pts[lab], axis=0)
            if float(np.sum((c0_new - c0) ** 2) + np.sum((c1_new - c1) ** 2)) < 1e-4:
                c0, c1 = c0_new, c1_new
                break
            c0, c1 = c0_new, c1_new
        sep2 = float(np.sum((c0 - c1) ** 2))
        if sep2 < float(max(0.0, sep)) ** 2:
            return None

        aa_flat = aa.reshape(-1)[idx]
        bb_flat = bb.reshape(-1)[idx]
        pts_all = np.stack([aa_flat, bb_flat], axis=1).astype(np.float32)
        d_to_0 = np.sum((pts_all - c0) ** 2, axis=1)
        d_to_1 = np.sum((pts_all - c1) ** 2, axis=1)
        lab_all = (d_to_1 < d_to_0)
        n1 = int(np.sum(lab_all))
        n0 = int(lab_all.size - n1)
        if n0 == 0 or n1 == 0:
            return None
        frac1 = float(n1) / float(n0 + n1)
        frac0 = 1.0 - frac1
        mf = float(max_frac)
        if mf > 0:
            if min(frac0, frac1) > mf:
                return None
        if n1 < n0:
            bad = lab_all
        else:
            bad = ~lab_all
        out = np.zeros_like(inner_m, dtype=bool).reshape(-1)
        out[idx] = bad
        return out.reshape(inner_m.shape)

    def _compute_noise_for_mode(m_s: str) -> "object":
        # Returns a boolean mask or None (no cleaning/min_area here).
        nonlocal inner
        if m_s == "resid_gray":
            gray = img_np.mean(axis=2)
            s = float(sigma)
            if s <= 0:
                s = 6.0
            blur = ndimage.gaussian_filter(gray, sigma=s, mode="reflect")
            resid = np.abs(gray - blur)
            vals = resid[inner]
            if vals.size <= 0:
                return None
            p = float(percentile)
            p = min(100.0, max(50.0, p))
            thr = float(np.percentile(vals, p))
            thr = max(thr, 1.5)
            nm = (resid >= thr) & inner
            return nm if np.any(nm) else None

        if m_s == "lab_resid":
            r = img_np[:, :, 0] / 255.0
            g = img_np[:, :, 1] / 255.0
            b = img_np[:, :, 2] / 255.0

            def inv_gamma(u: "object") -> "object":
                return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)

            r2 = inv_gamma(r)
            g2 = inv_gamma(g)
            b2 = inv_gamma(b)
            X = r2 * 0.4124564 + g2 * 0.3575761 + b2 * 0.1804375
            Y = r2 * 0.2126729 + g2 * 0.7151522 + b2 * 0.0721750
            Z = r2 * 0.0193339 + g2 * 0.1191920 + b2 * 0.9503041
            Xn, Yn, Zn = 0.95047, 1.0, 1.08883
            x = X / Xn
            y = Y / Yn
            z = Z / Zn
            eps = 216.0 / 24389.0
            k = 24389.0 / 27.0

            def f(t: "object") -> "object":
                return np.where(t > eps, np.cbrt(t), (k * t + 16.0) / 116.0)

            fx = f(x)
            fy = f(y)
            fz = f(z)
            aa = 500.0 * (fx - fy)
            bb = 200.0 * (fy - fz)
            s = float(sigma)
            if s <= 0:
                s = 6.0
            aa_blur = ndimage.gaussian_filter(aa, sigma=s, mode="reflect")
            bb_blur = ndimage.gaussian_filter(bb, sigma=s, mode="reflect")
            resid = np.sqrt((aa - aa_blur) ** 2 + (bb - bb_blur) ** 2)
            vals = resid[inner]
            if vals.size <= 0:
                return None
            p = float(percentile)
            p = min(100.0, max(50.0, p))
            thr = float(np.percentile(vals, p))
            thr = max(thr, float(max(0.0, z_thresh)))
            nm = (resid >= thr) & inner
            return nm if np.any(nm) else None

        if m_s == "lab_kmeans":
            r = img_np[:, :, 0] / 255.0
            g = img_np[:, :, 1] / 255.0
            b = img_np[:, :, 2] / 255.0

            def inv_gamma(u: "object") -> "object":
                return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)

            r2 = inv_gamma(r)
            g2 = inv_gamma(g)
            b2 = inv_gamma(b)
            X = r2 * 0.4124564 + g2 * 0.3575761 + b2 * 0.1804375
            Y = r2 * 0.2126729 + g2 * 0.7151522 + b2 * 0.0721750
            Z = r2 * 0.0193339 + g2 * 0.1191920 + b2 * 0.9503041
            Xn, Yn, Zn = 0.95047, 1.0, 1.08883
            x = X / Xn
            y = Y / Yn
            z = Z / Zn
            eps = 216.0 / 24389.0
            k = 24389.0 / 27.0

            def f(t: "object") -> "object":
                return np.where(t > eps, np.cbrt(t), (k * t + 16.0) / 116.0)

            fx = f(x)
            fy = f(y)
            fz = f(z)
            aa = 500.0 * (fx - fy)
            bb = 200.0 * (fy - fz)
            try:
                labeled, num = ndimage.label(inner)
            except Exception:
                labeled, num = (None, 0)
            if labeled is None or int(num) <= 1:
                nm = _lab_kmeans_noise2(aa, bb, inner, float(kmeans_sep), float(kmeans_max_frac))
                return nm if (nm is not None and np.any(nm)) else None
            acc = np.zeros_like(inner, dtype=bool)
            for cid in range(1, int(num) + 1):
                comp = labeled == cid
                if not np.any(comp):
                    continue
                nm = _lab_kmeans_noise2(aa, bb, comp, float(kmeans_sep), float(kmeans_max_frac))
                if nm is not None and np.any(nm):
                    acc |= nm
            return acc if np.any(acc) else None

        # lab_outlier
        r = img_np[:, :, 0] / 255.0
        g = img_np[:, :, 1] / 255.0
        b = img_np[:, :, 2] / 255.0

        def inv_gamma(u: "object") -> "object":
            return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)

        r2 = inv_gamma(r)
        g2 = inv_gamma(g)
        b2 = inv_gamma(b)
        X = r2 * 0.4124564 + g2 * 0.3575761 + b2 * 0.1804375
        Y = r2 * 0.2126729 + g2 * 0.7151522 + b2 * 0.0721750
        Z = r2 * 0.0193339 + g2 * 0.1191920 + b2 * 0.9503041
        Xn, Yn, Zn = 0.95047, 1.0, 1.08883
        x = X / Xn
        y = Y / Yn
        z = Z / Zn
        eps = 216.0 / 24389.0
        k = 24389.0 / 27.0

        def f(t: "object") -> "object":
            return np.where(t > eps, np.cbrt(t), (k * t + 16.0) / 116.0)

        fx = f(x)
        fy = f(y)
        fz = f(z)
        aa = 500.0 * (fx - fy)
        bb = 200.0 * (fy - fz)
        a_vals = aa[inner]
        b_vals = bb[inner]
        if a_vals.size <= 0 or b_vals.size <= 0:
            return None
        med_a = float(np.median(a_vals))
        med_b = float(np.median(b_vals))
        mad_a = float(np.median(np.abs(a_vals - med_a)))
        mad_b = float(np.median(np.abs(b_vals - med_b)))
        mad_a = max(mad_a, 1e-3)
        mad_b = max(mad_b, 1e-3)
        za = np.abs(aa - med_a) / mad_a
        zb = np.abs(bb - med_b) / mad_b
        score = np.sqrt(za * za + zb * zb)
        svals = score[inner]
        if svals.size <= 0:
            return None
        zt = float(z_thresh)
        if zt <= 0:
            zt = 6.0
        p = float(percentile)
        p = min(100.0, max(50.0, p))
        thr_p = float(np.percentile(svals, p))
        thr = max(zt, thr_p)
        nm = (score >= thr) & inner
        return nm if np.any(nm) else None

    if mode_s == "auto":
        acc = np.zeros_like(inner, dtype=bool)
        for m_s in ("lab_kmeans", "lab_resid", "resid_gray", "lab_outlier"):
            nm = _compute_noise_for_mode(m_s)
            if nm is not None and np.any(nm):
                acc |= nm
        noise = acc if np.any(acc) else None
    elif mode_s == "resid_gray":
        noise = _compute_noise_for_mode("resid_gray")
    elif mode_s == "lab_resid":
        noise = _compute_noise_for_mode("lab_resid")
    elif mode_s == "lab_kmeans":
        noise = _compute_noise_for_mode("lab_kmeans")
    else:
        noise = _compute_noise_for_mode("lab_outlier")

    if mode_s == "auto":
        # auto already computed via union above; do not fall through to legacy per-mode code below
        pass
    elif mode_s == "resid_gray":
        gray = img_np.mean(axis=2)
        s = float(sigma)
        if s <= 0:
            s = 6.0
        blur = ndimage.gaussian_filter(gray, sigma=s, mode="reflect")
        resid = np.abs(gray - blur)
        vals = resid[inner]
        if vals.size <= 0:
            return None
        p = float(percentile)
        p = min(100.0, max(50.0, p))
        thr = float(np.percentile(vals, p))
        thr = max(thr, 1.5)
        noise = (resid >= thr) & inner
    elif mode_s == "lab_resid":
        # Lab residual against a blurred baseline (more stable than global outlier on gradients)
        r = img_np[:, :, 0] / 255.0
        g = img_np[:, :, 1] / 255.0
        b = img_np[:, :, 2] / 255.0

        def inv_gamma(u: "object") -> "object":
            return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)

        r = inv_gamma(r)
        g = inv_gamma(g)
        b = inv_gamma(b)

        X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
        Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
        Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

        Xn, Yn, Zn = 0.95047, 1.0, 1.08883
        x = X / Xn
        y = Y / Yn
        z = Z / Zn

        eps = 216.0 / 24389.0
        k = 24389.0 / 27.0

        def f(t: "object") -> "object":
            return np.where(t > eps, np.cbrt(t), (k * t + 16.0) / 116.0)

        fx = f(x)
        fy = f(y)
        fz = f(z)
        aa = 500.0 * (fx - fy)
        bb = 200.0 * (fy - fz)

        s = float(sigma)
        if s <= 0:
            s = 6.0
        aa_blur = ndimage.gaussian_filter(aa, sigma=s, mode="reflect")
        bb_blur = ndimage.gaussian_filter(bb, sigma=s, mode="reflect")
        resid = np.sqrt((aa - aa_blur) ** 2 + (bb - bb_blur) ** 2)
        vals = resid[inner]
        if vals.size <= 0:
            return
        p = float(percentile)
        p = min(100.0, max(50.0, p))
        thr = float(np.percentile(vals, p))
        thr = max(thr, float(max(0.0, z_thresh)))
        noise = (resid >= thr) & inner
    elif mode_s == "lab_kmeans":
        r = img_np[:, :, 0] / 255.0
        g = img_np[:, :, 1] / 255.0
        b = img_np[:, :, 2] / 255.0

        def inv_gamma(u: "object") -> "object":
            return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)

        r = inv_gamma(r)
        g = inv_gamma(g)
        b = inv_gamma(b)

        X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
        Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
        Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

        Xn, Yn, Zn = 0.95047, 1.0, 1.08883
        x = X / Xn
        y = Y / Yn
        z = Z / Zn

        eps = 216.0 / 24389.0
        k = 24389.0 / 27.0

        def f(t: "object") -> "object":
            return np.where(t > eps, np.cbrt(t), (k * t + 16.0) / 116.0)

        fx = f(x)
        fy = f(y)
        fz = f(z)
        aa = 500.0 * (fx - fy)
        bb = 200.0 * (fy - fz)
        try:
            labeled, num = ndimage.label(inner)
        except Exception:
            labeled, num = (None, 0)
        if labeled is None or int(num) <= 1:
            noise = _lab_kmeans_noise(aa, bb, inner, float(kmeans_sep), float(kmeans_max_frac))
        else:
            acc = np.zeros_like(inner, dtype=bool)
            for cid in range(1, int(num) + 1):
                comp = labeled == cid
                if not np.any(comp):
                    continue
                nm = _lab_kmeans_noise(aa, bb, comp, float(kmeans_sep), float(kmeans_max_frac))
                if nm is not None and np.any(nm):
                    acc |= nm
            noise = acc if np.any(acc) else None
    else:
        # Lab outlier
        r = img_np[:, :, 0] / 255.0
        g = img_np[:, :, 1] / 255.0
        b = img_np[:, :, 2] / 255.0

        def inv_gamma(u: "object") -> "object":
            return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)

        r = inv_gamma(r)
        g = inv_gamma(g)
        b = inv_gamma(b)

        X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
        Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
        Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

        Xn, Yn, Zn = 0.95047, 1.0, 1.08883
        x = X / Xn
        y = Y / Yn
        z = Z / Zn

        eps = 216.0 / 24389.0
        k = 24389.0 / 27.0

        def f(t: "object") -> "object":
            return np.where(t > eps, np.cbrt(t), (k * t + 16.0) / 116.0)

        fx = f(x)
        fy = f(y)
        fz = f(z)
        aa = 500.0 * (fx - fy)
        bb = 200.0 * (fy - fz)

        a_vals = aa[inner]
        b_vals = bb[inner]
        if a_vals.size <= 0 or b_vals.size <= 0:
            return None

        med_a = float(np.median(a_vals))
        med_b = float(np.median(b_vals))
        mad_a = float(np.median(np.abs(a_vals - med_a)))
        mad_b = float(np.median(np.abs(b_vals - med_b)))
        mad_a = max(mad_a, 1e-3)
        mad_b = max(mad_b, 1e-3)

        za = np.abs(aa - med_a) / mad_a
        zb = np.abs(bb - med_b) / mad_b
        score = np.sqrt(za * za + zb * zb)
        svals = score[inner]
        if svals.size <= 0:
            return None
        zt = float(z_thresh)
        if zt <= 0:
            zt = 6.0
        p = float(percentile)
        p = min(100.0, max(50.0, p))
        thr_p = float(np.percentile(svals, p))
        thr = max(zt, thr_p)
        noise = (score >= thr) & inner

    if noise is None or not np.any(noise):
        return None

    # Clean
    try:
        noise = ndimage.binary_opening(noise, structure=np.ones((3, 3)), iterations=1)
    except Exception:
        pass
    labeled, num = ndimage.label(noise)
    if num > 0 and int(min_area) > 1:
        keep = np.zeros_like(noise, dtype=bool)
        for i in range(1, num + 1):
            comp = labeled == i
            if int(np.sum(comp)) >= int(min_area):
                keep |= comp
        noise = keep
    if not np.any(noise):
        return None
    return noise


def _postprocess_components(
    image: "object",
    mask: "object",
    *,
    delta_abs: float,
    min_inner: int,
    max_inner: int,
    margin: int,
    dilate: int,
    noise_mask: "object",
    dry_run: bool,
    debug_dir: str,
) -> "object":
    from PIL import Image, ImageFilter
    import numpy as np
    from scipy import ndimage

    img = image.convert("RGB")
    out_img = img.copy()
    img_np = np.array(img, dtype=np.float32)
    gray = img_np.mean(axis=2)
    local_mean = ndimage.uniform_filter(gray, size=5, mode="reflect")
    local_mean2 = ndimage.uniform_filter(gray * gray, size=5, mode="reflect")
    local_var = np.maximum(local_mean2 - local_mean * local_mean, 0.0)
    complexity = np.sqrt(local_var)

    m = mask.convert("L")
    if m.size != out_img.size:
        try:
            m = m.resize(out_img.size, resample=Image.NEAREST)
            print(f"  postprocess: resized mask {mask.size} -> {out_img.size}")
        except Exception:
            # If resize fails, continue with original; downstream will likely error and be caught by caller.
            pass
    m_np = (np.array(m, dtype=np.uint8) > 128)
    if not np.any(m_np):
        return img

    labeled, num = ndimage.label(m_np)
    if num <= 0:
        return img

    w, h = out_img.size

    dbg_base = None
    if debug_dir:
        dbg_base = Path(debug_dir)
        dbg_base.mkdir(parents=True, exist_ok=True)

    for cid in range(1, num + 1):
        comp0 = labeled == cid
        if not np.any(comp0):
            continue

        comp = comp0
        dpx = int(max(0, dilate))
        if dpx > 0:
            try:
                comp = ndimage.binary_dilation(comp0, iterations=dpx)
            except Exception:
                comp = comp0

        dist_in = ndimage.distance_transform_edt(comp)
        dmax = int(min(float(dist_in.max()), float(max_inner)))
        if dmax <= 0:
            continue

        # Baseline: boundary rings are usually the most reliable context transition zone.
        # We use mean complexity on the first one or two rings as a reference.
        base_vals = []
        for bd in (1, 2):
            ring_b = comp & (dist_in >= (bd - 1)) & (dist_in < bd)
            if np.any(ring_b):
                base_vals.append(float(np.mean(complexity[ring_b])))
        base = float(np.mean(base_vals)) if base_vals else 0.0

        inner_by_noise = None
        if noise_mask is not None:
            try:
                inner_by_noise = (noise_mask & comp)
            except Exception:
                inner_by_noise = None

        trigger = None
        if inner_by_noise is not None and np.any(inner_by_noise):
            print(f"  post[comp={cid:03d}] noise pixels={int(np.sum(inner_by_noise))}")
        else:
            prev = None
            curve = []
            for d in range(1, dmax + 1):
                ring = comp & (dist_in >= (d - 1)) & (dist_in < d)
                if not np.any(ring):
                    continue
                cur = float(np.mean(complexity[ring]))
                curve.append((d, cur))
                if d >= int(min_inner):
                    if prev is not None and abs(cur - prev) >= float(delta_abs):
                        trigger = d
                        break
                    if base_vals and (cur - base) >= float(delta_abs):
                        trigger = d
                        break
                prev = cur

            if trigger is None:
                if curve:
                    print(f"  post[comp={cid:03d}] base={base:.2f} dmax={dmax} -> no trigger (last={curve[-1][1]:.2f})")
                else:
                    print(f"  post[comp={cid:03d}] base={base:.2f} dmax={dmax} -> no trigger (empty curve)")
            else:
                cur_v = None
                for dd, vv in curve:
                    if dd == trigger:
                        cur_v = vv
                        break
                cur_s = f"{cur_v:.2f}" if cur_v is not None else "?"
                print(
                    f"  post[comp={cid:03d}] base={base:.2f} dmax={dmax} -> trigger d={trigger} (mean={cur_s}, delta={delta_abs})"
                )

        if dbg_base is not None:
            ys, xs = np.where(comp)
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())
            pad = int(max(4, margin))
            y0p, y1p = max(0, y0 - pad), min(h - 1, y1 + pad)
            x0p, x1p = max(0, x0 - pad), min(w - 1, x1 + pad)

            comp_u8 = (comp.astype(np.uint8) * 255)
            comp_rgb = np.dstack([comp_u8] * 3)
            comp_crop = comp_rgb[y0p : y1p + 1, x0p : x1p + 1]
            comp_im = Image.fromarray(comp_crop.astype(np.uint8))
            comp_im.save(dbg_base / f"comp_{cid:03d}_mask.png")

            if trigger is not None:
                inner = comp & (dist_in >= float(trigger))
                inner_u8 = (inner.astype(np.uint8) * 255)
                inner_rgb = np.dstack([inner_u8] * 3)
                inner_crop = inner_rgb[y0p : y1p + 1, x0p : x1p + 1]
                Image.fromarray(inner_crop.astype(np.uint8)).save(dbg_base / f"comp_{cid:03d}_inner.png")

        if (inner_by_noise is None or not np.any(inner_by_noise)) and trigger is None:
            continue

        if dry_run:
            continue

        if inner_by_noise is not None and np.any(inner_by_noise):
            inner = inner_by_noise
        else:
            inner = comp & (dist_in >= float(trigger))
        if not np.any(inner):
            continue

        inner_px = int(np.sum(inner))
        if inner_px < 64:
            print(f"  post[comp={cid:03d}] skip reinpaint (inner too small): px={inner_px}")
            continue

        ys, xs = np.where(inner)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        pad = int(max(8, margin))
        y0p, y1p = max(0, y0 - pad), min(h - 1, y1 + pad)
        x0p, x1p = max(0, x0 - pad), min(w - 1, x1 + pad)

        patch = out_img.crop((x0p, y0p, x1p + 1, y1p + 1))
        inner_patch = inner[y0p : y1p + 1, x0p : x1p + 1]
        mask_patch = Image.fromarray((inner_patch.astype(np.uint8) * 255), mode="L")
        mask_patch = mask_patch.filter(ImageFilter.GaussianBlur(radius=1))

        try:
            from simple_lama_inpainting import SimpleLama
        except Exception:
            continue

        lama = SimpleLama()
        print(
            f"  post[comp={cid:03d}] reinpaint patch=({x0p},{y0p})-({x1p},{y1p}) px={int(np.sum(inner_patch))}"
        )
        repaired = lama(patch.convert("RGB"), mask_patch)
        if repaired.size != patch.size:
            repaired = repaired.resize(patch.size)
        if mask_patch.size != patch.size:
            mask_patch = mask_patch.resize(patch.size)
        blended = Image.composite(repaired.convert("RGB"), patch.convert("RGB"), mask_patch.convert("L"))
        out_img.paste(blended, (x0p, y0p))

    return out_img


def _create_adaptive_mask(
    image_path: Path,
    mask_path: Path,
    base_expand: int = 10,
    *,
    edge_percentile: float = 97.0,
    near_edge_px: float = 2.0,
    far_edge_px: float = 10.0,
    min_expand: int = 1,
    return_debug: bool = False,
):
    """创建像素级自适应的mask，每个位置根据周围复杂度有不同的扩展大小"""
    from PIL import Image
    import numpy as np
    from scipy import ndimage
    
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    
    img_array = np.array(image, dtype=np.uint8)
    mask_array = np.array(mask, dtype=np.uint8) > 128  # 二值化mask
    
    # Distance to nearest masked pixel (outside->mask). Used to expand the mask.
    distance_to_mask = ndimage.distance_transform_edt(~mask_array)

    # Edge-aware expansion: keep expansion small near strong structure edges/lines.
    gray = img_array.mean(axis=2).astype(np.float32)
    gx = ndimage.sobel(gray, axis=1, mode="reflect")
    gy = ndimage.sobel(gray, axis=0, mode="reflect")
    grad = np.hypot(gx, gy)
    # Robust threshold: focus on strongest edges.
    ep = float(edge_percentile)
    if not (0.0 < ep < 100.0):
        ep = 97.0
    edge_thr = float(np.percentile(grad, ep))
    edge_map = (grad >= edge_thr)
    # Distance to nearest edge pixel.
    dist_to_edge = ndimage.distance_transform_edt(~edge_map)

    max_expand = int(max(1, base_expand))
    min_expand_i = int(max(0, min_expand))
    if min_expand_i <= 0:
        min_expand_i = 1
    # If a component is very close to an edge line, limit expansion to min_expand.
    near_edge_px_f = float(max(0.0, near_edge_px))
    # Past this distance, allow full expansion.
    far_edge_px_f = float(max(near_edge_px_f + 1e-3, far_edge_px))

    # Component-wise strategy:
    # - split mask into connected components
    # - choose ONE expansion radius per component based on its distance to strong edges
    # - dilate that component uniformly by the chosen radius
    try:
        labeled, num = ndimage.label(mask_array)
    except Exception:
        labeled, num = None, 0

    expanded = np.zeros_like(mask_array, dtype=bool)
    comp_debug: list[dict] = []
    if labeled is None or num <= 0:
        expanded = mask_array.copy()
    else:
        for i in range(1, int(num) + 1):
            comp = (labeled == i)
            if not np.any(comp):
                continue

            # Use a robust near-edge statistic for the component: 10th percentile of dist_to_edge inside comp.
            dvals = dist_to_edge[comp]
            if dvals.size <= 0:
                d_comp = 0.0
            else:
                d_comp = float(np.percentile(dvals.astype(np.float32), 10.0))

            # Map distance-to-edge to expansion radius [min_expand_i .. max_expand]
            r_f = float(
                np.interp(
                    np.array([d_comp], dtype=np.float32),
                    [0.0, near_edge_px_f, far_edge_px_f],
                    [float(min_expand_i), float(min_expand_i), float(max_expand)],
                )[0]
            )
            r = int(np.clip(int(round(r_f)), min_expand_i, max_expand))

            if return_debug:
                ys, xs = np.where(comp)
                if ys.size > 0:
                    cy = float(np.mean(ys))
                    cx = float(np.mean(xs))
                    px = int(ys.size)
                else:
                    cy, cx, px = 0.0, 0.0, 0
                comp_debug.append(
                    {
                        "id": int(i),
                        "px": int(px),
                        "d_comp": float(d_comp),
                        "r": int(r),
                        "cx": float(cx),
                        "cy": float(cy),
                    }
                )

            # Dilate the component by radius r using distance transform
            if r <= 0:
                expanded |= comp
            else:
                dist_to_comp = ndimage.distance_transform_edt(~comp)
                expanded |= (dist_to_comp <= float(r))

    adaptive_mask = (expanded.astype(np.uint8) * 255)
    
    # 跳过平滑处理，保持精确的mask边界
    # adaptive_mask = ndimage.binary_closing(adaptive_mask, structure=np.ones((1,1))).astype(np.uint8) * 255
    
    # 确保数据类型正确
    adaptive_mask = adaptive_mask.astype(np.uint8)

    out_mask = Image.fromarray(adaptive_mask, mode='L')
    if return_debug:
        return out_mask, comp_debug
    return out_mask


def _save_adaptive_component_vis(
    image_path: Path,
    comp_debug: list[dict],
    out_path: Path,
) -> None:
    from PIL import Image, ImageDraw

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for d in comp_debug:
        cid = int(d.get("id", 0))
        r = int(d.get("r", 0))
        cx = float(d.get("cx", 0.0))
        cy = float(d.get("cy", 0.0))
        text = f"{cid}:{r}"
        x = int(round(cx))
        y = int(round(cy))
        # simple high-contrast outline
        draw.text((x + 1, y + 1), text, fill=(0, 0, 0))
        draw.text((x, y), text, fill=(255, 255, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def _simple_lama_inpaint(
    image_path: Path,
    mask_path: Path,
    output_path: Path,
    expand_mask: int = 0,
    adaptive: bool = False,
    save_masked: bool = False,
    save_debug: bool = False,
) -> None:
    try:
        from simple_lama_inpainting import SimpleLama
    except Exception as e:
        raise RuntimeError(
            f"simple-lama-inpainting is not available ({e}). "
            f"Install: pip install simple-lama-inpainting"
        )

    from PIL import Image, ImageFilter
    import numpy as np

    simple_lama = SimpleLama()
    image = Image.open(image_path).convert("RGB")
    
    if adaptive:
        base = int(expand_mask)
        if getattr(_simple_lama_inpaint, "_auto_expand", False):
            from PIL import Image
            import numpy as np
            from scipy import ndimage

            image_dbg = Image.open(image_path).convert("RGB")
            mask_dbg = Image.open(mask_path).convert("L")

            img = np.array(image_dbg, dtype=np.float32)
            m = (np.array(mask_dbg, dtype=np.uint8) > 128)
            gray = img.mean(axis=2)
            local_mean = ndimage.uniform_filter(gray, size=5, mode="reflect")
            local_mean2 = ndimage.uniform_filter(gray * gray, size=5, mode="reflect")
            local_var = np.maximum(local_mean2 - local_mean * local_mean, 0.0)
            complexity = np.sqrt(local_var)
            distance_map = ndimage.distance_transform_edt(~m)

            prev = None
            chosen = int(getattr(_simple_lama_inpaint, "_auto_expand_min", 1))
            max_expand = int(getattr(_simple_lama_inpaint, "_auto_expand_max", 30))
            delta_abs = float(getattr(_simple_lama_inpaint, "_auto_expand_delta", 6.0))
            for d in range(1, max(1, max_expand) + 1):
                ring = (distance_map >= d - 1) & (distance_map < d)
                if not np.any(ring):
                    continue
                cur = float(np.mean(complexity[ring]))
                if prev is not None and abs(cur - prev) >= float(delta_abs):
                    chosen = d
                    break
                prev = cur
                chosen = d

            lo = int(getattr(_simple_lama_inpaint, "_auto_expand_min", 1))
            hi = int(max(1, max_expand))
            if hi < lo:
                hi = lo
            if chosen < lo:
                chosen = lo
            if chosen > hi:
                chosen = hi

            base = int(chosen)
            print(f"  auto expand-mask: {base}px")
            if getattr(_simple_lama_inpaint, "_debug_vis", False):
                dbg_dir = getattr(_simple_lama_inpaint, "_debug_vis_dir", "")
                dbg_base = Path(dbg_dir) if dbg_dir else output_path.parent
                dbg_path = dbg_base / f"{image_path.stem}_autoexpand_dbg.png"
                try:
                    _save_auto_expand_debug_vis(
                        image_path=image_path,
                        mask_path=mask_path,
                        out_path=dbg_path,
                        chosen_expand=base,
                        complexity=complexity,
                        distance_map=distance_map,
                    )
                    print(f"  debug-vis: {dbg_path}")
                except Exception as e:
                    print(f"  debug-vis failed: {e}")
        want_comp_vis = bool(getattr(_simple_lama_inpaint, "_adaptive_comp_vis", False))
        cm = _create_adaptive_mask(
            image_path,
            mask_path,
            base,
            edge_percentile=float(getattr(_simple_lama_inpaint, "_adaptive_edge_percentile", 97.0)),
            near_edge_px=float(getattr(_simple_lama_inpaint, "_adaptive_near_edge_px", 2.0)),
            far_edge_px=float(getattr(_simple_lama_inpaint, "_adaptive_far_edge_px", 10.0)),
            min_expand=int(getattr(_simple_lama_inpaint, "_adaptive_min_expand", 1)),
            return_debug=want_comp_vis,
        )
        if want_comp_vis:
            mask, comp_debug = cm
            comp_debug = list(comp_debug or [])
            for d in comp_debug:
                cid = int(d.get("id", 0))
                px = int(d.get("px", 0))
                dc = float(d.get("d_comp", 0.0))
                rr = int(d.get("r", 0))
                print(f"  adaptive[comp={cid:03d}] px={px} d_comp={dc:.2f} -> r={rr}")
            try:
                vis_path = output_path.parent / f"{output_path.stem}_adaptive_comp.png"
                _save_adaptive_component_vis(image_path=image_path, comp_debug=comp_debug, out_path=vis_path)
                print(f"  adaptive-comp-vis: {vis_path}")
            except Exception as e:
                print(f"  adaptive-comp-vis failed: {e}")
        else:
            mask = cm
        print(f"  使用像素级自适应mask，基础扩展: {base}px")
    else:
        # 使用传统均匀扩展
        mask = Image.open(mask_path).convert("L")
        if expand_mask > 0:
            mask_array = np.array(mask)
            from scipy import ndimage
            kernel = np.ones((expand_mask, expand_mask), np.uint8)
            mask_array = ndimage.binary_dilation(mask_array, kernel).astype(np.uint8) * 255
            mask = Image.fromarray(mask_array)
    
    # 确保mask是正确的数据类型
    mask = mask.convert("L")
    
    # 对mask进行轻微模糊以获得更自然的边缘
    mask = mask.filter(ImageFilter.GaussianBlur(radius=1))
    
    if save_masked:
        masked_img = image.copy()
        # Use a distinctive color (magenta) to show what is being removed
        # Paste magenta where mask > 0
        from PIL import Image, ImageDraw, ImageFont
        magenta = Image.new("RGB", image.size, (255, 0, 255))
        masked_img = Image.composite(magenta, image, mask)

        try:
            import numpy as np
            from scipy import ndimage

            ma = (np.array(mask, dtype=np.uint8) > 128)
            labeled, num = ndimage.label(ma)
            draw = ImageDraw.Draw(masked_img)
            font = ImageFont.load_default()
            for cid in range(1, int(num) + 1):
                comp = (labeled == cid)
                if not np.any(comp):
                    continue
                ys, xs = np.where(comp)
                if ys.size <= 0:
                    continue
                cx = int(round(float(xs.mean())))
                cy = int(round(float(ys.mean())))
                try:
                    draw.text(
                        (cx, cy),
                        str(int(cid)),
                        fill=(0, 0, 0),
                        font=font,
                        anchor="mm",
                        stroke_width=2,
                        stroke_fill=(255, 255, 255),
                    )
                except Exception:
                    draw.text((cx, cy), str(int(cid)), fill=(255, 255, 255), font=font)
        except Exception:
            pass
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        masked_path = output_path.parent / f"{output_path.stem}_masked{output_path.suffix}"
        masked_img.save(masked_path)
        print(f"  saved masked image (pre-inpaint): {masked_path}")

    result = simple_lama(image, mask)
    if getattr(_simple_lama_inpaint, "_postprocess", False) and bool(
        getattr(_simple_lama_inpaint, "_post_save_pre", True)
    ):
        pre_path = output_path.with_name(f"{output_path.stem}_pre{output_path.suffix}")
        try:
            pre_path.parent.mkdir(parents=True, exist_ok=True)
            result.save(pre_path)
        except Exception as e:
            print(f"  postprocess: save pre failed: {e}")
    if getattr(_simple_lama_inpaint, "_postprocess", False) and bool(
        getattr(_simple_lama_inpaint, "_post_noise_vis", False)
    ):
        try:
            dbg_dir = str(getattr(_simple_lama_inpaint, "_post_noise_vis_dir", ""))
            base = Path(dbg_dir) if dbg_dir else output_path.parent
            vis_path = base / f"{image_path.stem}_post_noise.png"
            _save_post_noise_vis(
                result,
                mask,
                vis_path,
                mode=str(getattr(_simple_lama_inpaint, "_post_noise_mode", "lab_outlier")),
                percentile=float(getattr(_simple_lama_inpaint, "_post_noise_percentile", 99.5)),
                sigma=float(getattr(_simple_lama_inpaint, "_post_noise_sigma", 6.0)),
                edge_exclude=int(getattr(_simple_lama_inpaint, "_post_noise_edge", 4)),
                flat=float(getattr(_simple_lama_inpaint, "_post_noise_flat", 8.0)),
                kmeans_sep=float(getattr(_simple_lama_inpaint, "_post_noise_kmeans_sep", 8.0)),
                kmeans_max_frac=float(getattr(_simple_lama_inpaint, "_post_noise_kmeans_max_frac", 0.25)),
                z_thresh=float(getattr(_simple_lama_inpaint, "_post_noise_z", 6.0)),
                min_area=int(getattr(_simple_lama_inpaint, "_post_noise_min_area", 6)),
            )
        except Exception as e:
            print(f"  postprocess: noise-vis failed: {e}")
    if getattr(_simple_lama_inpaint, "_postprocess", False):
        try:
            noise_mask = _detect_post_noise_mask(
                result,
                mask,
                mode=str(getattr(_simple_lama_inpaint, "_post_noise_mode", "lab_outlier")),
                percentile=float(getattr(_simple_lama_inpaint, "_post_noise_percentile", 99.5)),
                sigma=float(getattr(_simple_lama_inpaint, "_post_noise_sigma", 6.0)),
                edge_exclude=int(getattr(_simple_lama_inpaint, "_post_noise_edge", 4)),
                flat=float(getattr(_simple_lama_inpaint, "_post_noise_flat", 8.0)),
                kmeans_sep=float(getattr(_simple_lama_inpaint, "_post_noise_kmeans_sep", 8.0)),
                kmeans_max_frac=float(getattr(_simple_lama_inpaint, "_post_noise_kmeans_max_frac", 0.25)),
                z_thresh=float(getattr(_simple_lama_inpaint, "_post_noise_z", 6.0)),
                min_area=int(getattr(_simple_lama_inpaint, "_post_noise_min_area", 6)),
            )
            try:
                m_np = (np.array(mask, dtype=np.uint8) > 128)
                denom = int(np.sum(m_np))
                num = int(np.sum(noise_mask)) if noise_mask is not None else 0
                ratio = (float(num) / float(denom)) if denom > 0 else 0.0
                print(f"  post-noise: pixels={num} / {denom} ({ratio:.6f})")
            except Exception:
                pass
            if bool(getattr(_simple_lama_inpaint, "_post_repair_global", False)) and (noise_mask is not None) and np.any(noise_mask):
                try:
                    from scipy import ndimage
                    nm = noise_mask.astype(bool)
                    nm = ndimage.binary_dilation(nm, iterations=2)
                    m2 = Image.fromarray((nm.astype(np.uint8) * 255), mode="L")
                    m2 = m2.filter(ImageFilter.GaussianBlur(radius=1))
                    result = simple_lama(result.convert("RGB"), m2)
                    print(f"  postprocess: global repair applied (pixels={int(np.sum(nm))})")
                except Exception as e:
                    print(f"  postprocess: global repair failed: {e}")
            else:
                result = _postprocess_components(
                    result,
                    mask,
                    delta_abs=float(getattr(_simple_lama_inpaint, "_post_delta", 6.0)),
                    min_inner=int(getattr(_simple_lama_inpaint, "_post_min_inner", 2)),
                    max_inner=int(getattr(_simple_lama_inpaint, "_post_max_inner", 30)),
                    margin=int(getattr(_simple_lama_inpaint, "_post_margin", 64)),
                    dilate=int(getattr(_simple_lama_inpaint, "_post_dilate", 2)),
                    noise_mask=noise_mask,
                    dry_run=bool(getattr(_simple_lama_inpaint, "_post_dry_run", False)),
                    debug_dir=str(getattr(_simple_lama_inpaint, "_post_debug_dir", "")),
                )
        except Exception as e:
            print(f"  postprocess failed: {e}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    if save_debug:
        try:
            dbg = output_path.parent / "_debug" / output_path.stem
            vis_mask = mask.point(lambda p: 255 if p >= 127 else 0).convert("L")
            _save_debug_bundle(
                original_path=image_path,
                mask_l=vis_mask,
                result_path=output_path,
                debug_dir=dbg,
                prefilled=None,
            )
            print(f"  saved debug bundle: {dbg}")
        except Exception as e:
            print(f"  debug bundle failed: {e}")


def main() -> int:
    try:
        os.environ.setdefault("PYTHONIOENCODING", "utf-8:backslashreplace")
        os.environ.setdefault("LANG", "C.UTF-8")
        os.environ.setdefault("LC_ALL", "C.UTF-8")
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        else:
            import io

            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="backslashreplace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
        else:
            import io

            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Generative inpainting demo using LaMa via lama-cleaner (image + mask -> output)")
    parser.add_argument("--image", default="", help="Input image path")
    parser.add_argument("--mask", default="", help="Mask image path (white=remove)")
    parser.add_argument(
        "--noise-only-image",
        default="",
        help="Run only post-noise detection on this image and exit (no inpaint). Uses a full-image mask unless --noise-only-mask is provided.",
    )
    parser.add_argument(
        "--noise-only-mask",
        default="",
        help="Optional mask for --noise-only-image (white=region to analyze). If empty, use full-image mask.",
    )
    parser.add_argument(
        "--noise-only-out",
        default="",
        help="Output path for noise visualization. Default: next to input as '<stem>_post_noise.png'.",
    )
    parser.add_argument("--image-dir", default="", help="Batch mode: input image directory")
    parser.add_argument("--mask-dir", default="", help="Batch mode: mask directory containing '<stem>_mask.png'")
    parser.add_argument("--recursive", action="store_true", help="Batch mode: recurse into subdirectories")
    parser.add_argument("--limit", type=int, default=0, help="Batch mode: only process the first N images (0 = no limit)")
    parser.add_argument("--out-dir", default="", help="Base output directory")
    parser.add_argument("--output", default="", help="Output image path (single mode) or output filename prefix (batch mode)")
    parser.add_argument(
        "--backend",
        choices=["simple_lama", "iopaint", "mat", "lama_cleaner_cli", "flux_fill_api", "flux_fill_local", "sdxl_inpaint_local"],
        default="simple_lama",
        help="Inpainting backend (default: simple_lama). lama_cleaner_cli requires a lama-cleaner CLI that supports batch input and may not support per-image masks.",
    )
    parser.add_argument("--model", default="lama", help="Model name (for output folder naming)")
    parser.add_argument("--device", default="cpu", help="Device name (for output folder naming)")
    parser.add_argument("--hd", action="store_true", help="Enable HD strategy if supported by lama-cleaner")
    parser.add_argument(
        "--method-subdir",
        action="store_true",
        help="Write outputs into a per-method subdirectory under --out-dir (default: false)",
    )
    parser.add_argument(
        "--skip-missing-mask",
        action="store_true",
        help="Batch mode: skip images when mask is missing (default: false -> fail)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if output file already exists",
    )
    parser.add_argument(
        "--expand-mask",
        type=int,
        default=0,
        help="Expand mask by N pixels to ensure complete coverage (default: 0)",
    )
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save debug assets into '<out_dir>/_debug/<stem>/' (original/mask/mask_vis/prefilled/result)",
    )
    parser.add_argument(
        "--flux-prompt",
        default="",
        help="Prompt used for flux_fill_api backend. If empty, uses a default text-removal prompt.",
    )
    parser.add_argument(
        "--flux-steps",
        type=int,
        default=50,
        help="Steps for flux_fill_api backend (default: 50)",
    )
    parser.add_argument(
        "--flux-guidance",
        type=float,
        default=1.5,
        help="Guidance for flux_fill_api backend (default: 30)",
    )
    parser.add_argument(
        "--flux-safety-tolerance",
        type=int,
        default=2,
        help="Safety tolerance for flux_fill_api backend (default: 2)",
    )
    parser.add_argument(
        "--flux-output-format",
        choices=["jpeg", "jpg", "png", "webp"],
        default="png",
        help="Output format for flux_fill_api backend (default: png)",
    )
    parser.add_argument(
        "--flux-endpoint",
        default="https://api.bfl.ai/v1/flux-pro-1.0-fill",
        help="HTTP endpoint for flux_fill_api backend",
    )
    parser.add_argument(
        "--flux-model-dir",
        default="/home/wangyankun/models/flux_fill",
        help="Local FLUX.1-Fill-dev model directory for flux_fill_local backend",
    )
    parser.add_argument(
        "--flux-lora-path",
        default="/home/wangyankun/models/loras/removalV2.safetensors",
        help="Local LoRA .safetensors path for flux_fill_local backend (ObjectRemovalFluxFill)",
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
        "--flux-poll-interval",
        type=float,
        default=0.5,
        help="Polling interval in seconds for flux_fill_api backend (default: 0.5)",
    )
    parser.add_argument(
        "--flux-poll-timeout",
        type=float,
        default=300.0,
        help="Polling timeout in seconds for flux_fill_api backend (default: 300)",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Enable adaptive mask expansion based on background complexity",
    )
    parser.add_argument(
        "--auto-expand",
        action="store_true",
        help="Auto estimate expand-mask by scanning distance rings and stopping at a complexity change point",
    )
    parser.add_argument(
        "--auto-expand-max",
        type=int,
        default=30,
        help="Auto-expand maximum search radius (default: 30)",
    )
    parser.add_argument(
        "--auto-expand-delta",
        type=float,
        default=6.0,
        help="Auto-expand stop threshold on ring mean complexity delta (default: 6.0)",
    )
    parser.add_argument(
        "--auto-expand-min",
        type=int,
        default=1,
        help="Auto-expand minimum returned radius (default: 1)",
    )
    parser.add_argument(
        "--adaptive-edge-percentile",
        type=float,
        default=97.0,
        help="When using --adaptive, edge percentile (Sobel grad) used to define strong edges/lines (default: 97.0)",
    )
    parser.add_argument(
        "--adaptive-near-edge-px",
        type=float,
        default=2.0,
        help="When using --adaptive, distance (px) to edges under which expansion is clamped to adaptive-min-expand (default: 2.0)",
    )
    parser.add_argument(
        "--adaptive-far-edge-px",
        type=float,
        default=10.0,
        help="When using --adaptive, distance (px) to edges beyond which full base expand is allowed (default: 10.0)",
    )
    parser.add_argument(
        "--adaptive-min-expand",
        type=int,
        default=1,
        help="When using --adaptive, minimum expansion near strong edges/lines (default: 1). Set to 2 to enforce fixed 2px near edges.",
    )
    parser.add_argument(
        "--adaptive-comp-vis",
        action="store_true",
        help="When using --adaptive, label mask connected components and save a visualization '<stem>_adaptive_comp.png' with component ids and chosen expansion radii; also print per-component stats.",
    )
    parser.add_argument(
        "--debug-vis",
        action="store_true",
        help="Save debug visualization for auto-expand/adaptive (complexity heatmap + chosen ring)",
    )
    parser.add_argument(
        "--debug-vis-dir",
        default="",
        help="Directory to write debug visualizations (default: alongside outputs)",
    )
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Postprocess after first inpaint: scan mask rings outer->inner, optionally re-inpaint inner dirty cores",
    )
    parser.add_argument(
        "--post-repair-global",
        action="store_true",
        help="When using --postprocess, do a single global second-pass inpaint using the detected noise mask (instead of many per-component patch reinpaints)",
    )
    parser.add_argument(
        "--post-delta",
        type=float,
        default=6.0,
        help="Postprocess trigger threshold on ring mean complexity delta (default: 6.0)",
    )
    parser.add_argument(
        "--post-min-inner",
        type=int,
        default=2,
        help="Postprocess minimum inner ring index to allow triggering (default: 2)",
    )
    parser.add_argument(
        "--post-max-inner",
        type=int,
        default=30,
        help="Postprocess maximum inner ring index to scan (default: 30)",
    )
    parser.add_argument(
        "--post-margin",
        type=int,
        default=64,
        help="Postprocess crop margin around a component for selective re-inpaint (default: 64)",
    )
    parser.add_argument(
        "--post-dilate",
        type=int,
        default=2,
        help="Postprocess dilate each connected component by N pixels before scoring/repair (default: 2)",
    )
    parser.add_argument(
        "--post-dry-run",
        action="store_true",
        help="Postprocess only scoring/debug outputs; do not run second inpaint",
    )
    parser.add_argument(
        "--post-debug-dir",
        default="",
        help="Directory to write postprocess debug files (per-component mask/inner mask)",
    )
    parser.add_argument(
        "--post-save-pre",
        action="store_true",
        default=True,
        help="When using --postprocess, also save the first-pass result as '<name>_pre.png' for comparison (default: true)",
    )
    parser.add_argument(
        "--post-no-save-pre",
        action="store_false",
        dest="post_save_pre",
        help="Disable saving the first-pass '<name>_pre.png' image",
    )
    parser.add_argument(
        "--post-noise-vis",
        action="store_true",
        default=True,
        help="Save a visualization '<stem>_post_noise.png' highlighting detected noise pixels within the mask (default: true)",
    )
    parser.add_argument(
        "--post-no-noise-vis",
        action="store_false",
        dest="post_noise_vis",
        help="Disable saving the post noise visualization",
    )
    parser.add_argument(
        "--post-noise-vis-dir",
        default="",
        help="Directory to write post noise visualizations (default: alongside outputs)",
    )
    parser.add_argument(
        "--post-noise-mode",
        choices=["auto", "resid_gray", "lab_outlier", "lab_resid", "lab_kmeans"],
        default="lab_outlier",
        help="Noise detection mode used for visualization and repair (default: lab_outlier)",
    )
    parser.add_argument(
        "--post-noise-percentile",
        type=float,
        default=99.5,
        help="Percentile threshold of high-frequency residual inside mask used for noise detection (default: 99.5)",
    )
    parser.add_argument(
        "--post-noise-sigma",
        type=float,
        default=6.0,
        help="Gaussian blur sigma used to form low-frequency baseline for blotch detection (default: 6.0)",
    )
    parser.add_argument(
        "--post-noise-edge",
        type=int,
        default=4,
        help="Exclude this many pixels near mask boundary when detecting inner blotches (mask erosion iterations, default: 4)",
    )
    parser.add_argument(
        "--post-noise-flat",
        type=float,
        default=8.0,
        help="Only detect noise in relatively flat regions; threshold on grayscale Sobel gradient magnitude (<=flat). Set 0 to disable (default: 8.0)",
    )
    parser.add_argument(
        "--post-noise-kmeans-sep",
        type=float,
        default=8.0,
        help="When using lab_kmeans, minimum required separation between the 2 cluster centers in Lab(a,b) space (default: 8.0)",
    )
    parser.add_argument(
        "--post-noise-kmeans-max-frac",
        type=float,
        default=0.25,
        help="When using lab_kmeans, discard the result if the smaller cluster occupies more than this fraction of mask pixels (default: 0.25)",
    )
    parser.add_argument(
        "--post-noise-z",
        type=float,
        default=6.0,
        help="Robust z-score threshold for Lab outlier detection; final threshold is max(z, percentile) (default: 6.0)",
    )
    parser.add_argument(
        "--post-noise-min-area",
        type=int,
        default=6,
        help="Minimum blob area to keep for noise visualization (default: 6)",
    )
    parser.add_argument(
        "--save-masked",
        action="store_true",
        help="Save the masked image (filled with magenta) before inpainting for debugging",
    )
    parser.add_argument(
        "--iopaint-config",
        default="",
        help="Pass a config json file to iopaint backend (e.g. to control hd_strategy/crop params)",
    )
    args = parser.parse_args()

    if args.noise_only_image:
        from PIL import Image
        import numpy as np

        img_path = Path(args.noise_only_image)
        if not img_path.exists():
            print(f"noise-only-image not found: {img_path}", file=sys.stderr)
            return 2
        img = Image.open(img_path).convert("RGB")
        if args.noise_only_mask:
            mpath = Path(args.noise_only_mask)
            if not mpath.exists():
                print(f"noise-only-mask not found: {mpath}", file=sys.stderr)
                return 3
            m = Image.open(mpath).convert("L")
        else:
            m = Image.fromarray(np.full((img.size[1], img.size[0]), 255, dtype=np.uint8), mode="L")

        out = Path(args.noise_only_out) if args.noise_only_out else img_path.with_name(f"{img_path.stem}_post_noise.png")
        nm = _detect_post_noise_mask(
            img,
            m,
            mode=str(args.post_noise_mode),
            percentile=float(args.post_noise_percentile),
            sigma=float(args.post_noise_sigma),
            edge_exclude=int(args.post_noise_edge),
            flat=float(args.post_noise_flat),
            kmeans_sep=float(args.post_noise_kmeans_sep),
            kmeans_max_frac=float(args.post_noise_kmeans_max_frac),
            z_thresh=float(args.post_noise_z),
            min_area=int(args.post_noise_min_area),
        )

        # Always write output (avoid stale previous overlays when no noise is detected)
        img_np = np.array(img, dtype=np.uint8)
        overlay = img_np.copy()
        if nm is not None and np.any(nm):
            overlay[nm] = (255, 0, 0)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(out)

        m_np = (np.array(m.resize(img.size, resample=Image.NEAREST), dtype=np.uint8) > 128)
        denom = int(np.sum(m_np))
        num = int(np.sum(nm)) if nm is not None else 0
        ratio = (float(num) / float(denom)) if denom > 0 else 0.0
        print(f"noise-only: out={out}")
        print(f"noise-only: noise_pixels={num} / {denom} ({ratio:.6f})")
        return 0

    setattr(_simple_lama_inpaint, "_auto_expand", bool(args.auto_expand))
    setattr(_simple_lama_inpaint, "_auto_expand_max", int(args.auto_expand_max))
    setattr(_simple_lama_inpaint, "_auto_expand_delta", float(args.auto_expand_delta))
    setattr(_simple_lama_inpaint, "_auto_expand_min", int(args.auto_expand_min))
    setattr(_simple_lama_inpaint, "_debug_vis", bool(args.debug_vis))
    setattr(_simple_lama_inpaint, "_debug_vis_dir", str(args.debug_vis_dir or ""))
    setattr(_simple_lama_inpaint, "_postprocess", bool(args.postprocess))
    setattr(_simple_lama_inpaint, "_post_repair_global", bool(getattr(args, "post_repair_global", False)))
    setattr(_simple_lama_inpaint, "_post_delta", float(args.post_delta))
    setattr(_simple_lama_inpaint, "_post_min_inner", int(args.post_min_inner))
    setattr(_simple_lama_inpaint, "_post_max_inner", int(args.post_max_inner))
    setattr(_simple_lama_inpaint, "_post_margin", int(args.post_margin))
    setattr(_simple_lama_inpaint, "_post_dilate", int(args.post_dilate))
    setattr(_simple_lama_inpaint, "_post_dry_run", bool(args.post_dry_run))
    setattr(_simple_lama_inpaint, "_post_debug_dir", str(args.post_debug_dir or ""))
    setattr(_simple_lama_inpaint, "_post_save_pre", bool(args.post_save_pre))
    setattr(_simple_lama_inpaint, "_post_noise_vis", bool(args.post_noise_vis))
    setattr(_simple_lama_inpaint, "_post_noise_vis_dir", str(args.post_noise_vis_dir or ""))
    setattr(_simple_lama_inpaint, "_post_noise_mode", str(args.post_noise_mode))
    setattr(_simple_lama_inpaint, "_post_noise_percentile", float(args.post_noise_percentile))
    setattr(_simple_lama_inpaint, "_post_noise_sigma", float(args.post_noise_sigma))
    setattr(_simple_lama_inpaint, "_post_noise_edge", int(args.post_noise_edge))
    setattr(_simple_lama_inpaint, "_post_noise_flat", float(args.post_noise_flat))
    setattr(_simple_lama_inpaint, "_post_noise_kmeans_sep", float(args.post_noise_kmeans_sep))
    setattr(_simple_lama_inpaint, "_post_noise_kmeans_max_frac", float(args.post_noise_kmeans_max_frac))
    setattr(_simple_lama_inpaint, "_post_noise_z", float(args.post_noise_z))
    setattr(_simple_lama_inpaint, "_post_noise_min_area", int(args.post_noise_min_area))
    setattr(_simple_lama_inpaint, "_save_masked", bool(args.save_masked))
    setattr(_simple_lama_inpaint, "_adaptive_edge_percentile", float(args.adaptive_edge_percentile))
    setattr(_simple_lama_inpaint, "_adaptive_near_edge_px", float(args.adaptive_near_edge_px))
    setattr(_simple_lama_inpaint, "_adaptive_far_edge_px", float(args.adaptive_far_edge_px))
    setattr(_simple_lama_inpaint, "_adaptive_min_expand", int(args.adaptive_min_expand))
    setattr(_simple_lama_inpaint, "_adaptive_comp_vis", bool(getattr(args, "adaptive_comp_vis", False)))

    is_batch = bool(args.image_dir)
    if is_batch:
        image_dir = Path(args.image_dir)
        mask_dir = Path(args.mask_dir) if args.mask_dir else None
        if not image_dir.exists() or not image_dir.is_dir():
            print(f"image-dir not found: {image_dir}", file=sys.stderr)
            return 2
        if mask_dir is None or not mask_dir.exists() or not mask_dir.is_dir():
            print("batch mode requires --mask-dir (directory)", file=sys.stderr)
            return 3
        if not args.out_dir:
            print("batch mode requires --out-dir", file=sys.stderr)
            return 4

        out_dir = _method_dir(Path(args.out_dir), args.model, args.device, args.hd, args.method_subdir)
        prefix = args.output.strip() if args.output else ""

        images = _iter_images(image_dir, recursive=args.recursive)
        if not images:
            print(f"no images found in: {image_dir}", file=sys.stderr)
            return 5

        lim = int(getattr(args, "limit", 0) or 0)
        if lim > 0:
            images = images[:lim]

        print(f"Found {len(images)} images")
        failed = 0

        if args.backend in {"iopaint", "mat"}:
            try:
                model = "mat" if args.backend == "mat" else (args.model or "lama")
                _iopaint_run_batch(
                    image_dir,
                    mask_dir,
                    out_dir,
                    model=model,
                    device=args.device,
                    images=images,
                    expand_mask=args.expand_mask,
                    adaptive=args.adaptive,
                    config=args.iopaint_config,
                )
            except Exception as e:
                print(f"[FAIL] iopaint batch: {e}", file=sys.stderr)
                return 10
            print("Done")
            return 0
        for img in images:
            # Support both old structure (<uuid>_mask.png) and new structure (<seq>_<uuid>/mask.png)
            mask = mask_dir / f"{img.stem}_mask.png"
            if not mask.exists():
                # Try new structure: find folder containing this UUID
                for sub in mask_dir.iterdir():
                    if sub.is_dir() and sub.name.endswith(f"_{img.stem}"):
                        candidate = sub / "mask.png"
                        if candidate.exists():
                            mask = candidate
                            break
            if not mask.exists():
                msg = f"mask not found: {mask} (for image {img.name})"
                if args.skip_missing_mask:
                    print(f"[SKIP] {msg}")
                    continue
                print(msg, file=sys.stderr)
                failed += 1
                continue

            out_name = f"{prefix}{img.stem}.png" if prefix else f"{img.stem}.png"
            out = out_dir / out_name
            if args.skip_existing and out.exists():
                print(f"[SKIP] exists: {out}")
                continue

            print(f"\n[{img.name}] -> {out}")
            try:
                if args.backend == "simple_lama":
                    _simple_lama_inpaint(
                        image_path=img,
                        mask_path=mask,
                        output_path=out,
                        expand_mask=int(args.expand_mask),
                        adaptive=bool(args.adaptive),
                        save_masked=bool(args.save_masked),
                        save_debug=bool(args.save_debug),
                    )
                elif args.backend == "flux_fill_api":
                    _flux_fill_api_inpaint(
                        image_path=img,
                        mask_path=mask,
                        output_path=out,
                        expand_mask=int(args.expand_mask),
                        adaptive=bool(args.adaptive),
                        save_masked=bool(args.save_masked),
                        save_debug=bool(args.save_debug),
                        prompt=str(args.flux_prompt or ""),
                        steps=int(args.flux_steps),
                        guidance=float(args.flux_guidance),
                        safety_tolerance=int(args.flux_safety_tolerance),
                        output_format=str(args.flux_output_format),
                        poll_interval=float(args.flux_poll_interval),
                        poll_timeout=float(args.flux_poll_timeout),
                        endpoint=str(args.flux_endpoint),
                    )
                elif args.backend == "flux_fill_local":
                    _flux_fill_local_inpaint(
                        image_path=img,
                        mask_path=mask,
                        output_path=out,
                        expand_mask=int(args.expand_mask),
                        adaptive=bool(args.adaptive),
                        save_masked=bool(args.save_masked),
                        save_debug=bool(args.save_debug),
                        prompt=str(args.flux_prompt or ""),
                        steps=int(args.flux_steps),
                        guidance=float(args.flux_guidance),
                        model_dir=str(args.flux_model_dir),
                        lora_path=str(args.flux_lora_path),
                        lora_scale=float(args.flux_lora_scale),
                        device=str(args.device),
                    )
                elif args.backend == "sdxl_inpaint_local":
                    _sdxl_inpaint_local_inpaint(
                        image_path=img,
                        mask_path=mask,
                        output_path=out,
                        expand_mask=int(args.expand_mask),
                        adaptive=bool(args.adaptive),
                        save_masked=bool(args.save_masked),
                        save_debug=bool(args.save_debug),
                        prompt=str(args.flux_prompt or ""),
                        negative_prompt=str(args.sdxl_negative_prompt or ""),
                        steps=int(args.sdxl_steps),
                        guidance=float(args.sdxl_guidance),
                        strength=float(args.sdxl_strength),
                        model=str(args.sdxl_model),
                        device=str(args.device),
                    )
                elif args.backend in {"iopaint", "mat"}:
                    model = "mat" if args.backend == "mat" else (args.model or "lama")
                    _iopaint_run(
                        img,
                        mask,
                        out,
                        model=model,
                        device=args.device,
                        expand_mask=args.expand_mask,
                        adaptive=args.adaptive,
                        config=args.iopaint_config,
                    )
                else:
                    print(
                        "lama_cleaner_cli backend does not support per-image masks with this lama-cleaner version. "
                        "Use --backend simple_lama instead.",
                        file=sys.stderr,
                    )
                    failed += 1
                    continue
            except Exception as e:
                print(f"[FAIL] {img.name}: {e}", file=sys.stderr)
                failed += 1

        if failed:
            print(f"Done with failures: {failed}", file=sys.stderr)
            return 10
        print("Done")
        return 0

    # Single-image mode
    if not args.image or not args.mask:
        print("single mode requires --image, --mask, --output (or use batch mode with --image-dir)", file=sys.stderr)
        return 2

    img = Path(args.image)
    mask = Path(args.mask)
    if not img.exists():
        print(f"image not found: {img}", file=sys.stderr)
        return 2
    if not mask.exists():
        print(f"mask not found: {mask}", file=sys.stderr)
        return 3

    if not args.output:
        print("single mode requires --output", file=sys.stderr)
        return 4

    out = Path(args.output)
    if args.out_dir:
        out_dir = _method_dir(Path(args.out_dir), args.model, args.device, args.hd, args.method_subdir)
        if out.suffix:
            out = out_dir / out.name
        else:
            out = out_dir / f"{out.name}.png"

    try:
        if args.backend == "simple_lama":
            _simple_lama_inpaint(
                image_path=img,
                mask_path=mask,
                output_path=out,
                expand_mask=int(args.expand_mask),
                adaptive=bool(args.adaptive),
                save_masked=bool(args.save_masked),
                save_debug=bool(args.save_debug),
            )
        elif args.backend == "flux_fill_api":
            _flux_fill_api_inpaint(
                image_path=img,
                mask_path=mask,
                output_path=out,
                expand_mask=int(args.expand_mask),
                adaptive=bool(args.adaptive),
                save_masked=bool(args.save_masked),
                save_debug=bool(args.save_debug),
                prompt=str(args.flux_prompt or ""),
                steps=int(args.flux_steps),
                guidance=float(args.flux_guidance),
                safety_tolerance=int(args.flux_safety_tolerance),
                output_format=str(args.flux_output_format),
                poll_interval=float(args.flux_poll_interval),
                poll_timeout=float(args.flux_poll_timeout),
                endpoint=str(args.flux_endpoint),
            )
        elif args.backend == "flux_fill_local":
            _flux_fill_local_inpaint(
                image_path=img,
                mask_path=mask,
                output_path=out,
                expand_mask=int(args.expand_mask),
                adaptive=bool(args.adaptive),
                save_masked=bool(args.save_masked),
                save_debug=bool(args.save_debug),
                prompt=str(args.flux_prompt or ""),
                steps=int(args.flux_steps),
                guidance=float(args.flux_guidance),
                model_dir=str(args.flux_model_dir),
                lora_path=str(args.flux_lora_path),
                lora_scale=float(args.flux_lora_scale),
                device=str(args.device),
            )
        elif args.backend == "sdxl_inpaint_local":
            _sdxl_inpaint_local_inpaint(
                image_path=img,
                mask_path=mask,
                output_path=out,
                expand_mask=int(args.expand_mask),
                adaptive=bool(args.adaptive),
                save_masked=bool(args.save_masked),
                save_debug=bool(args.save_debug),
                prompt=str(args.flux_prompt or ""),
                negative_prompt=str(args.sdxl_negative_prompt or ""),
                steps=int(args.sdxl_steps),
                guidance=float(args.sdxl_guidance),
                strength=float(args.sdxl_strength),
                model=str(args.sdxl_model),
                device=str(args.device),
            )
        elif args.backend in {"iopaint", "mat"}:
            model = "mat" if args.backend == "mat" else (args.model or "lama")
            _iopaint_run(
                img,
                mask,
                out,
                model=model,
                device=args.device,
                expand_mask=args.expand_mask,
                adaptive=args.adaptive,
                config=args.iopaint_config,
            )
        else:
            print(
                "lama_cleaner_cli backend does not support per-image masks with this lama-cleaner version. "
                "Use --backend simple_lama instead.",
                file=sys.stderr,
            )
            return 6
    except Exception as e:
        print(f"failed: {e}", file=sys.stderr)
        return 7

    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
