# IMG2XML

将技术架构图图片转换为结构化 XML 的工具集，支持 OCR 文字识别、SAM3 箭头/图标检测、以及基于 Flux 的智能 inpainting 去除。

## 功能特性

- **OCR 文字识别**：基于 Azure Document Intelligence，支持多种 mask 生成模式
- **SAM3 目标检测**：检测箭头、图标、连接线等元素
- **智能 Inpainting**：使用 Flux/SDXL/LaMA 等模型去除文字和图形元素
- **结构化输出**：生成按图片分组的 mask 文件，便于检查和后续处理

## 目录结构

```
IMG2XML/
├── run_azure_ocr_demo.py      # OCR + mask 生成主脚本
├── run_gen_inpaint_demo.py    # Inpainting 主脚本
├── run_mask_inpaint_pipeline.py  # 完整 pipeline 调度脚本
├── azure.py                   # Azure OCR 客户端
├── litserve_model/            # SAM3 等模型服务
│   ├── main_server.py         # 模型服务主入口
│   ├── apis/                  # API 定义
│   └── sam3/                  # SAM3 模型实现
├── modules/                   # 图像处理模块
│   ├── arrow_processor.py     # 箭头处理
│   ├── text_processor.py      # 文字处理
│   ├── slide/                 # PPT 生成相关
│   └── utils/                 # 工具函数
├── prompts/                   # Prompt 模板
├── config/                    # 配置文件
└── figure/                    # 输入图片目录（需自行准备）
```

## 环境要求

- Python 3.10+
- CUDA 11.8+ (GPU 推理)
- Azure Document Intelligence 容器服务
- 可选：Flux Fill 模型、SDXL Inpaint 模型、LaMA 模型

## 安装

```bash
# 克隆仓库
git clone git@github.com:hhq2004/banana.git
cd banana

# 安装依赖
pip install -r requirements.txt  # 如果有
```

## 使用方法

### 1. 启动 SAM3 服务（可选，用于箭头/图标检测）

```bash
python litserve_model/main_server.py --services sam3 &
```

### 2. 运行完整 Pipeline

```bash
python run_mask_inpaint_pipeline.py \
  --endpoint http://localhost:5001 \
  --input figure \
  --out-root result \
  --mask-mode line \
  --backend flux_fill_local \
  --erase-sam3 \
  --sam3-prompts "arrow,line,connector,icon,logo,chart" \
  --sam3-score-threshold 0.25 \
  --sam3-min-area 10 \
  --expand-mask-grid 2 \
  --flux-model-dir /path/to/flux_fill \
  --flux-lora-path /path/to/removalV2.safetensors \
  --flux-lora-scale 0.9 \
  --flux-steps 50 \
  --flux-guidance 1.5 \
  --flux-prompt "a pristine, empty technical architecture diagram template..." \
  --save-debug \
  --limit 20
```

### 3. 单独运行 OCR

```bash
python run_azure_ocr_demo.py \
  --endpoint http://localhost:5001 \
  --input figure \
  --output result/ocr/masks \
  --mask-mode line \
  --save-mask \
  --limit 20
```

### 4. 单独运行 Inpainting

```bash
python run_gen_inpaint_demo.py \
  --image-dir figure \
  --mask-dir result/ocr/masks \
  --out-dir result/inpaint \
  --backend flux_fill_local \
  --expand-mask 2
```

## 参数说明

### OCR 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--endpoint` | `http://localhost:5000` | Azure OCR 容器端点 |
| `--input` | `./图例` | 输入图片目录或文件 |
| `--output` | `./ocr_out` | 输出目录 |
| `--mask-mode` | `line` | Mask 模式: line/word/pixel |
| `--mask-dilate` | `2` | Mask 扩展像素 |
| `--save-mask` | False | 保存 mask 文件 |
| `--limit` | `0` | 限制处理图片数量 (0=不限制) |

### Inpainting 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--backend` | `simple_lama` | 后端: simple_lama/flux_fill_api/flux_fill_local/sdxl_inpaint_local |
| `--expand-mask` | `10` | Mask 扩展像素 |
| `--adaptive` | False | 自适应 mask 扩展 |
| `--save-debug` | False | 保存调试信息 |

### SAM3 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--erase-sam3` | False | 启用 SAM3 检测 |
| `--sam3-prompts` | `""` | 检测目标提示词，逗号分隔 |
| `--sam3-score-threshold` | `0.25` | 置信度阈值 |
| `--sam3-min-area` | `10` | 最小检测区域 |

## 输出结构

### OCR 输出

```
result/ocr/mask-line_d2_b2_pd18/
├── 001_<uuid>/
│   ├── vis.png          # 可视化结果
│   ├── mask.png         # 文字 mask
│   └── sam3_mask.png    # SAM3 检测 mask (如果启用)
├── 002_<uuid>/
│   └── ...
└── 020_<uuid>/
    └── ...
```

### Inpainting 输出

```
result/inpaint/md2_em2/model-flux_dev-gpu/
├── <uuid>.png           # 处理后的图片
└── _debug/              # 调试信息 (如果启用)
    └── <uuid>/
        ├── original.png
        ├── mask.png
        ├── prefilled.png
        └── result.png
```

## 支持的后端

1. **simple_lama**: LaMA 模型，快速但效果一般
2. **flux_fill_api**: Black Forest Labs Flux Fill API（需要 API Key）
3. **flux_fill_local**: 本地 Flux Fill 模型，效果最好
4. **sdxl_inpaint_local**: 本地 SDXL Inpaint 模型

## 环境变量

```bash
# Flux API Key (使用 flux_fill_api 后端时需要)
export BFL_API_KEY="your-api-key"

# Azure OCR 配置 (如果使用云服务)
export AZURE_OCR_ENDPOINT="your-endpoint"
export AZURE_OCR_KEY="your-key"
```

## 注意事项

1. **Azure OCR 配额**：确保 Azure Document Intelligence 容器服务正常运行且有足够配额
2. **GPU 内存**：Flux 和 SDXL 模型需要较大 GPU 内存（建议 24GB+）
3. **模型文件**：大模型文件（`sdxl_inpaint_0.1/`、`models/`）已通过 `.gitignore` 排除，需自行下载
4. **输入图片**：`figure/` 目录已排除，需自行准备输入数据

## License

MIT

## Author

huhanqing (3041034812@qq.com)
