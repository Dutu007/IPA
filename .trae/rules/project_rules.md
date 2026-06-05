# 智析病理 — 项目规则

## 应用形态

**本地 Web 服务 + 浏览器前端**：用户输入 WSI + 文字描述，经三步流水线处理后展示综合分析报告。

- 开发阶段：`python main.py` → 浏览器打开 `localhost:8000`
- 交付阶段：`start.bat` 一键启动（可选 PyInstaller 打包 .exe）
- **核心优势**：WSI 数据不出本机、无需网络、无需上传、部署轻量

## 用户输入与输出

| 输入 | 输出 |
|------|------|
| WSI 病理切片（.svs / .tiff） | 综合分析报告（网页展示） |
| 文字描述（病理科医生描述） | 含 QC 统计、ROI 热力图、精细分析结论 |

## 三步流水线

```
用户输入 → Step1(QC) → Step2(ROI) → Step3(精细分析) → 综合报告
```

## 编码规范

### 1. 函数化设计（关键）

所有脚本中的核心逻辑必须封装为可复用的函数，而不是裸写在 `__main__` 中，以便后续 Web 层直接 `import` 调用。

```python
# ✅ 正确
def process_step1(wsi_path: str, output_dir: str, tile_size: int = 256) -> dict:
    """WSI 切割 + QC 检测，返回统计结果"""
    ...

# ❌ 错误：裸脚本，Web 层无法复用
wsi_path = "../slide.svs"
...
```

### 2. 配置外置

路径、阈值、模型名、超参等通过函数参数或配置文件传入，不硬编码。

```python
def process_wsi(
    wsi_path: str,
    tile_size: int = 256,
    stride: int = 200,
    density_threshold: int = 3,
    model_path: str = "models/yolo_best.pt",
) -> dict:
    ...
```

### 3. 返回值约定

函数返回结构化 dict，包含状态码、统计信息和输出路径，便于 Web API 序列化为 JSON。

```python
def run_step1(wsi_path: str, output_dir: str) -> dict:
    return {
        "status": "ok",              # "ok" | "error"
        "message": "QC 完成",
        "total_tiles": 1200,
        "passed_tiles": 1100,
        "rejected_tiles": 100,
        "pass_rate": 0.917,
        "output_dir": "output/session_xxx/step1/",
    }
```

### 4. 进度反馈

长耗时任务使用回调函数输出进度，后续 Web 层用 WebSocket/SSE 推送前端。

```python
def run_pipeline(wsi_path: str, description: str, progress_callback=None) -> dict:
    """
    执行完整三步流水线

    参数:
        wsi_path: WSI 文件路径
        description: 用户文字描述
        progress_callback: 进度回调, 签名为 callback(step: str, percent: float, message: str)

    返回:
        dict: 综合分析报告
    """
    steps = [
        ("step1", step1_qc),
        ("step2", step2_roi),
        ("step3", step3_analysis),
    ]
    results = {}
    for i, (name, func) in enumerate(steps):
        if progress_callback:
            progress_callback(name, i / len(steps), f"{name} 开始...")
        results[name] = func(...)
        if progress_callback:
            progress_callback(name, (i + 1) / len(steps), f"{name} 完成")
    return combine_report(results, description)
```

### 5. 输出目录结构

所有分析结果按 session 隔离，放在 `output/` 下：

```
output/
└── <session_id>/
    ├── input/
    │   ├── original.svs     ← 原始 WSI（可选复制）
    │   └── description.txt  ← 用户文字描述
    ├── step1/
    │   ├── tiles/           ← 通过质检的瓦片
    │   ├── rejected/        ← 被拒绝的瓦片
    │   └── qc_report.json   ← QC 统计
    ├── step2/
    │   ├── heatmap.png      ← 病变密度热力图
    │   ├── regions/         ← 候选 ROI
    │   └── metadata.json    ← ROI 元数据
    ├── step3/
    │   └── analysis.json    ← 精细分析结果
    └── report.json           ← 综合报告（前端渲染用）
```

### 6. 注释规范

- 文件级：说明模块功能和整体流程
- 函数级：说明功能、参数、返回值、异常
- 注释语言：中文

### 7. 项目目录约定

```
IPA/
├── scripts/          ← 所有 Python 脚本（核心逻辑）
├── app/              ← FastAPI Web 应用（后续创建）
├── frontend/         ← Vue 3 前端（后续创建）
├── pannuke_yolo/     ← YOLOv8 标准数据集
├── pannuke_data/     ← 原始 PanNuke 数据（gitignore）
├── output/           ← 分析结果输出（gitignore）
├── models/           ← 模型权重（gitignore）
├── runs/             ← YOLO 训练产物（gitignore）
├── PLAN.md           ← 项目计划文档
└── README.md         ← 项目说明
```

## 技术栈

| 层级 | 技术选型 |
|------|----------|
| 应用架构 | FastAPI 本地服务 + 浏览器前端 |
| 前端 | Vue 3 + Element Plus |
| WSI 处理 | lazyslide |
| 质量检测 | pathprofilerQC2 |
| 目标检测 | Ultralytics YOLOv8 |
| 任务队列 | Celery + Redis（长任务异步执行） |
| 交付方式 | start.bat 启动 / PyInstaller 打包 |

## 运行命令

```powershell
# 代码检查（如配置了 lint 工具后补充）
# TODO: 添加 lint 和类型检查命令
```

## 注意事项

- 不提交 `pannuke_data/`、`output/`、`runs/`、`models/`、`*.pt` 等大文件
- WSI 文件由用户本地指定，不复制到项目目录（仅读访问）
- 模型权重通过配置文件指定路径，不硬编码
- 所有输出以 JSON 为主，便于 Web 前端渲染
- Windows 环境下注意路径分隔符兼容性
