"""
PanNuke → YOLOv8 目标检测格式转换脚本
======================================
功能:
  1. 读取 pathml 下载的 PanNuke 数据集（单张 PNG + 6通道 .npy 掩膜）
  2. 从 6通道实例掩膜中提取每个细胞核的 bounding box
  3. 将原始 5类细胞核标签映射为病变/非病变二分类
  4. 按 80/10/10 随机划分 train/val/test，固定随机种子
  5. 输出 YOLOv8 标准目录结构及 data.yaml 配置文件

掩膜格式:
  每张 .npy 文件形状为 (6, 256, 256), float64
  - Ch 0: Neoplastic (肿瘤)          → lesion (class 0)
  - Ch 1: Inflammatory (炎症)        → lesion (class 0)
  - Ch 2: Connective (结缔组织)      → non_lesion (class 1)
  - Ch 3: Dead (坏死)                → lesion (class 0)
  - Ch 4: Epithelial (上皮)          → non_lesion (class 1)
  - Ch 5: Background (背景)          → 忽略
  每个通道中像素值为实例ID (1, 2, 3, ...)，0 为背景

输出目录结构:
  pannuke_yolo/
  ├── train/
  │   ├── images/   ← PNG 图像
  │   └── labels/   ← YOLO 格式 .txt 标签
  ├── val/
  │   ├── images/
  │   └── labels/
  ├── test/
  │   ├── images/
  │   └── labels/
  └── data.yaml     ← YOLOv8 训练配置

最终目录结构:
pannuke_yolo/
├── data.yaml              ← YOLOv8 训练配置文件 ✅
├── train/
│   ├── images/  6320 张   ← 6048 有核 + 272 无核
│   └── labels/  6320 个   ← 对应 YOLO txt
├── val/
│   ├── images/  790 张    ← 755 有核 + 35 无核
│   └── labels/  790 个
└── test/
    ├── images/  791 张
    └── labels/  791 个
"""

import os
import shutil
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ==================== 用户配置 ====================

# 输入: pathml 下载后的 PanNuke 目录 (含 images/ 和 masks/ 子目录)
PANNUKE_DIR = "../pannuke_data"

# 输出: YOLOv8 格式数据集根目录
OUTPUT_DIR = "../pannuke_yolo"

# 训练 / 验证 / 测试比例 (之和必须为 1.0)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# 随机种子，确保每次运行划分结果一致
RANDOM_SEED = 42

# ==================== 标签映射 ====================

# 掩膜通道 → YOLO类别ID
# 0:Neoplastic→0(lesion), 1:Inflammatory→0(lesion), 2:Connective→1(non_lesion),
# 3:Dead→0(lesion), 4:Epithelial→1(non_lesion), 5:Background(忽略)
CHANNEL_TO_CLASS = {
    0: 0,
    1: 0,
    2: 1,
    3: 0,
    4: 1,
}

# YOLO 类别名称 (按 class_id 顺序)
CLASS_NAMES = ["lesion", "non_lesion"]

# 各通道名称 (用于统计输出)
CHANNEL_NAMES = {
    0: "Neoplastic",
    1: "Inflammatory",
    2: "Connective",
    3: "Dead",
    4: "Epithelial",
}


# ==================== 核心函数 ====================

def extract_bboxes_from_mask(mask):
    """
    从 6通道实例掩膜中提取所有细胞核的 YOLO bbox 和类别

    算法:
      对每个非背景通道，找出所有非零像素值的唯一实例ID，
      为每个实例计算最小外接矩形并归一化为 YOLO 格式。

    参数:
        mask: numpy 数组, shape (6, H, W), float64

    返回:
        list of (class_id, cx, cy, w, h):
          class_id: YOLO 类别 ID
          cx, cy: 归一化中心坐标 [0, 1]
          w, h: 归一化宽高 [0, 1]
    """
    H, W = mask.shape[1], mask.shape[2]
    bboxes = []

    for channel_id, class_id in CHANNEL_TO_CLASS.items():
        ch_mask = mask[channel_id]
        non_zero = ch_mask[ch_mask > 0]

        if len(non_zero) == 0:
            continue

        instance_ids = np.unique(non_zero)
        for inst_id in instance_ids:
            ys, xs = np.where(ch_mask == inst_id)
            if len(xs) == 0:
                continue

            x_min = xs.min()
            x_max = xs.max()
            y_min = ys.min()
            y_max = ys.max()

            # YOLO 格式: (cx, cy, w, h) 归一化
            cx = (x_min + x_max) / 2.0 / W
            cy = (y_min + y_max) / 2.0 / H
            nw = (x_max - x_min) / W
            nh = (y_max - y_min) / H

            # 裁剪到 [0, 1] 范围内
            cx = np.clip(cx, 0.0, 1.0)
            cy = np.clip(cy, 0.0, 1.0)
            nw = np.clip(nw, 0.0, 1.0)
            nh = np.clip(nh, 0.0, 1.0)

            bboxes.append((class_id, cx, cy, nw, nh))

    return bboxes


def process_dataset(input_dir, output_dir):
    """
    批量处理: 读取所有图像和掩膜，提取 bbox，写入 YOLO 标签文件

    参数:
        input_dir: PanNuke 数据目录 (含 images/ 和 masks/)
        output_dir: YOLO 输出根目录

    返回:
        list of (stem, has_label): 每个样本的文件名主干和是否有标签的元组
    """
    input_dir = Path(input_dir)
    im_dir = input_dir / "images"
    mask_dir = input_dir / "masks"

    out_im_dir = Path(output_dir) / "all" / "images"
    out_lb_dir = Path(output_dir) / "all" / "labels"
    out_im_dir.mkdir(parents=True, exist_ok=True)
    out_lb_dir.mkdir(parents=True, exist_ok=True)

    all_images = sorted(im_dir.glob("*.png"))

    print(f"读取图像: {len(all_images)} 张")

    stats = {
        "total_nuclei": 0,
        "lesion": 0,
        "non_lesion": 0,
        "empty_images": 0,
    }
    per_channel_stats = {ch: 0 for ch in CHANNEL_TO_CLASS}

    records = []
    for img_path in tqdm(all_images, desc="提取 bbox"):
        stem = img_path.stem
        mask_path = mask_dir / f"{stem}.npy"

        if not mask_path.is_file():
            tqdm.write(f"[警告] 无对应掩膜: {stem}")
            continue

        img = cv2.imread(str(img_path))
        mask = np.load(str(mask_path))

        bboxes = extract_bboxes_from_mask(mask)
        has_label = len(bboxes) > 0

        if has_label:
            for cls_id, _, _, _, _ in bboxes:
                stats["total_nuclei"] += 1
                if cls_id == 0:
                    stats["lesion"] += 1
                else:
                    stats["non_lesion"] += 1
        else:
            stats["empty_images"] += 1

        # 按通道统计
        for ch in CHANNEL_TO_CLASS:
            per_channel_stats[ch] += int((mask[ch] > 0).any())

        # 拷贝图像 (符号链接在 Windows 上不够稳定，直接复制)
        dst_im = out_im_dir / f"{stem}.png"
        if not dst_im.exists():
            shutil.copy2(str(img_path), str(dst_im))

        # 写入 YOLO 标签
        dst_lb = out_lb_dir / f"{stem}.txt"
        if bboxes:
            lines = [
                f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
                for cid, cx, cy, nw, nh in bboxes
            ]
            dst_lb.write_text("\n".join(lines), encoding="utf-8")
        else:
            dst_lb.write_text("", encoding="utf-8")

        records.append((stem, has_label))

    return records, stats, per_channel_stats


def split_dataset(records, output_dir):
    """
    随机划分 train/val/test 集并建立 YOLOv8 目录结构
    若目标目录已存在则跳过。

    参数:
        records: list of (stem, has_label)
        output_dir: YOLO 输出根目录
    """
    output_dir = Path(output_dir)
    all_im_dir = output_dir / "all" / "images"
    all_lb_dir = output_dir / "all" / "labels"

    all_exist = all((output_dir / s).is_dir() for s in ["train", "val", "test"])
    if all_exist:
        counts = {}
        for s in ["train", "val", "test"]:
            n = len(list((output_dir / s / "images").glob("*.png")))
            counts[s] = n
            print(f"  [{s}] 已存在 ({n} 张)")
        print("  跳过划分步骤")
        shutil.rmtree(str(output_dir / "all"), ignore_errors=True)
        return counts

    random.seed(RANDOM_SEED)
    stems = [r[0] for r in records]
    random.shuffle(stems)

    n_total = len(stems)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    n_test = n_total - n_train - n_val

    splits = {
        "train": stems[:n_train],
        "val": stems[n_train:n_train + n_val],
        "test": stems[n_train + n_val:],
    }

    for split_name, split_stems in splits.items():
        im_out = output_dir / split_name / "images"
        lb_out = output_dir / split_name / "labels"

        if im_out.is_dir():
            print(f"  [{split_name}] 已存在，跳过")
            continue

        im_out.mkdir(parents=True, exist_ok=True)
        lb_out.mkdir(parents=True, exist_ok=True)

        empty_count = 0
        for stem in tqdm(split_stems, desc=f"构建 {split_name} 集"):
            src_im = all_im_dir / f"{stem}.png"
            src_lb = all_lb_dir / f"{stem}.txt"
            dst_im = im_out / f"{stem}.png"
            dst_lb = lb_out / f"{stem}.txt"

            if src_im.exists() and not dst_im.exists():
                shutil.copy2(str(src_im), str(dst_im))
            if src_lb.exists():
                shutil.copy2(str(src_lb), str(dst_lb))
                if src_lb.stat().st_size == 0:
                    empty_count += 1

        has_label = len(split_stems) - empty_count
        print(f"  {split_name}: {len(split_stems)} 张 ({has_label} 有核, {empty_count} 无核)")

    # 清理临时 all 目录
    shutil.rmtree(str(output_dir / "all"), ignore_errors=True)
    return {"train": len(stems[:n_train]), "val": len(stems[n_train:n_train + n_val]),
            "test": len(stems[n_train + n_val:])}


def create_data_yaml(output_dir, class_names):
    """
    生成 YOLOv8 训练配置文件 data.yaml

    参数:
        output_dir: YOLO 输出根目录
        class_names: 类别名称列表
    """
    yaml_path = Path(output_dir) / "data.yaml"
    abs_path = Path(output_dir).resolve()

    content = f"""# PanNuke 病变/非病变 目标检测数据集 (YOLOv8 格式)
# 类别数
nc: {len(class_names)}
# 类别名称
names: {class_names}

# 数据集路径 (相对于此 data.yaml 所在目录)
path: .
train: train/images
val: val/images
test: test/images
"""
    yaml_path.write_text(content, encoding="utf-8")
    print(f"   配置文件: {yaml_path}")


# ==================== 主流程 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("PanNuke -> YOLOv8 格式转换")
    print(f"输入目录: {PANNUKE_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"划分比例: train={TRAIN_RATIO} / val={VAL_RATIO} / test={TEST_RATIO}")
    print(f"随机种子: {RANDOM_SEED}")
    print("=" * 60)

    # 1. 批量提取 bbox，保存到临时 all/ 目录
    all_im_dir = Path(OUTPUT_DIR) / "all" / "images"
    input_im_dir = Path(PANNUKE_DIR) / "images"
    expected_count = len(list(input_im_dir.glob("*.png")))
    existing_count = len(list(all_im_dir.glob("*.png"))) if all_im_dir.is_dir() else 0

    if existing_count == expected_count:
        print("\n[1/3] 检测到已有提取结果，跳过 bbox 提取 ...")
        records = [(p.stem, True) for p in sorted(all_im_dir.glob("*.png"))]
        stats = {"total_nuclei": "N/A", "lesion": "N/A",
                 "non_lesion": "N/A", "empty_images": "N/A"}
        per_ch_stats = {}
    else:
        print("\n[1/3] 提取所有 bounding box ...")
        records, stats, per_ch_stats = process_dataset(PANNUKE_DIR, OUTPUT_DIR)

        print(f"\n== 统计数据 ==")
        print(f"   总细胞核: {stats['total_nuclei']}")
        print(f"   病变(lesion): {stats['lesion']}")
        print(f"   非病变(non_lesion): {stats['non_lesion']}")
        print(f"   无细胞核图像: {stats['empty_images']}")
        print(f"   各通道覆盖图像数:")
        for ch, name in CHANNEL_NAMES.items():
            print(f"     Ch{ch} {name}: {per_ch_stats[ch]}")

    # 2. 随机划分数据集
    print("\n[2/3] 划分 train / val / test ...")
    split_counts = split_dataset(records, OUTPUT_DIR)

    # 3. 生成 YOLO 配置
    print("\n[3/3] 生成 YOLOv8 配置文件 ...")
    create_data_yaml(OUTPUT_DIR, CLASS_NAMES)

    print("\n" + "=" * 60)
    print(f"  转换完成!")
    print(f"  训练集: {split_counts['train']} 张")
    print(f"  验证集: {split_counts['val']} 张")
    print(f"  测试集: {split_counts['test']} 张")
    print(f"\n  训练命令:")
    print(f"  yolo detect train data={Path(OUTPUT_DIR).resolve() / 'data.yaml'} \\")
    print(f"      model=yolov8n.pt epochs=50 imgsz=256")
    print("=" * 60)
