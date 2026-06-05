"""
YOLOv8 模型深度分析与可视化脚本
===============================
功能:
  1. 假阴性分析: 定位漏检样本，保存带标注的可视化图
  2. 按组织类型分组评估: 19种组织的分组指标 + 柱状图
  3. 置信度分布: 预测置信度直方图，TP/FP 对比
  4. BBox 尺寸分层评估: 按大小分组计算 recall
  5. 综合 HTML 报告生成

使用方式:
  python -m scripts.yolo_training.analysis --model runs/detect/train/weights/best.pt

注意:
  本脚本为开发期工具，用于深度评估模型质量，不参与 Web 运行时。
"""

import os
import sys
import json
import re
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


def get_project_root():
    """
    获取项目根目录（IPA/）

    返回:
        Path: 项目根目录路径
    """
    return Path(__file__).resolve().parent.parent.parent


TISSUE_MAPPING = {
    "Breast": "Breast",
    "Lung": "Lung",
    "Skin": "Skin",
    "Colon": "Colon",
    "Cervix": "Cervix",
    "Kidney": "Kidney",
    "Stomach": "Stomach",
    "Prostate": "Prostate",
    "Testis": "Testis",
    "Liver": "Liver",
    "Thyroid": "Thyroid",
    "Pancreatic": "Pancreas",
    "Ovarian": "Ovary",
    "Bladder": "Bladder",
    "Bile-duct": "Bile-duct",
    "Uterus": "Uterus",
    "Adrenal-gland": "Adrenal-gland",
    "HeadNeck": "Head&Neck",
    "Esophagus": "Esophagus",
    "Head&Neck": "Head&Neck",
}

CLASS_NAMES = {0: "lesion", 1: "non_lesion"}

BBOX_SIZE_THRESHOLDS = {
    "small": 0.01,
    "medium": 0.05,
}


def _parse_tissue_from_filename(stem):
    """
    从文件名解析组织类型

    参数:
        stem: 文件名主干，如 "fold1_532_Breast"

    返回:
        str: 标准化后的组织类型名
    """
    for raw, std in TISSUE_MAPPING.items():
        if stem.endswith(f"_{raw}") or stem.endswith(f"-{raw}"):
            return std
    return "Unknown"


def _load_labels(label_dir, stem):
    """
    读取 YOLO 格式标签文件

    参数:
        label_dir: 标签目录
        stem: 文件名主干

    返回:
        list of [class_id, cx, cy, w, h]: 归一化坐标
    """
    label_path = Path(label_dir) / f"{stem}.txt"
    if not label_path.is_file():
        return []
    labels = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    labels.append([float(x) for x in parts[:5]])
                except ValueError:
                    continue
    return labels


def _iou(box1, box2):
    """
    计算两个 bbox 的 IoU (YOLO 归一化格式)

    参数:
        box1: [cx, cy, w, h] 归一化
        box2: [cx, cy, w, h] 归一化

    返回:
        float: IoU 值 [0, 1]
    """
    def to_xyxy(box):
        cx, cy, w, h = box
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    a = to_xyxy(box1)
    b = to_xyxy(box2)
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    iou_val = inter / (area_a + area_b - inter) if (area_a + area_b - inter) > 0 else 0
    return iou_val


def _bbox_area(w, h):
    """计算归一化 bbox 面积"""
    return w * h


def _draw_detections(img, gt_boxes, pred_boxes, img_size=256):
    """
    在原图上绘制 GT 和预测框

    参数:
        img: 原始图像 (H, W, 3)
        gt_boxes: 真实框列表
        pred_boxes: 预测框列表
        img_size: 图像尺寸

    返回:
        np.ndarray: 带标注的图像
    """
    vis = img.copy()
    for box in gt_boxes:
        cls_id, cx, cy, w, h = int(box[0]), box[1], box[2], box[3], box[4]
        x1 = int((cx - w / 2) * img_size)
        y1 = int((cy - h / 2) * img_size)
        x2 = int((cx + w / 2) * img_size)
        y2 = int((cy + h / 2) * img_size)
        color = (0, 255, 0) if cls_id == 0 else (0, 200, 200)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = "GT:lesion" if cls_id == 0 else "GT:non_lesion"
        cv2.putText(vis, label, (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    for box in pred_boxes:
        cx, cy, w, h, conf = box[0], box[1], box[2], box[3], box[4]
        x1 = int((cx - w / 2) * img_size)
        y1 = int((cy - h / 2) * img_size)
        x2 = int((cx + w / 2) * img_size)
        y2 = int((cy + h / 2) * img_size)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, f"pred:{conf:.2f}", (x1, y2 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    return vis


def analyze_false_negatives(
    model_path: str,
    data_yaml: str = None,
    split: str = "test",
    iou_threshold: float = 0.3,
    conf_threshold: float = 0.25,
    max_samples: int = 30,
    output_dir: str = None,
) -> dict:
    """
    假阴性分析：找出 test 集中漏检最严重的样本并可视化

    对每张测试图像运行推理，将预测框与 GT 框做 IoU 匹配，
    统计每张图 lesion 和 non_lesion 的漏检数，输出 top-k 漏检图。

    参数:
        model_path: 模型权重路径 (.pt)
        data_yaml: 数据集配置 yaml
        split: 评估划分 ("test" | "val")
        iou_threshold: 匹配 IoU 阈值
        conf_threshold: 预测置信度阈值
        max_samples: 最多可视化样本数
        output_dir: 输出目录，默认 runs/analysis/false_negatives

    返回:
        dict: {"total_gt_lesion": int, "total_fn_lesion": int,
               "fn_rate_lesion": float, "top_fn_samples": list, ...}
    """
    root = get_project_root()
    if data_yaml is None:
        data_yaml = str(root / "pannuke_yolo" / "data.yaml")
    if output_dir is None:
        output_dir = str(root / "runs" / "analysis" / "false_negatives")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(data_yaml).parent
    img_dir = data_dir / split / "images"
    label_dir = data_dir / split / "labels"

    image_paths = sorted(img_dir.glob("*.png"))
    print(f"[假阴性分析] 评估图像: {len(image_paths)} 张")

    model = YOLO(model_path)

    total_gt_lesion = 0
    total_gt_non_lesion = 0
    total_fn_lesion = 0
    total_fn_non_lesion = 0
    per_image_stats = []

    for img_path in image_paths:
        stem = img_path.stem
        gt_labels = _load_labels(label_dir, stem)
        if not gt_labels:
            continue

        results = model(img_path, verbose=False, conf=conf_threshold,
                        imgsz=256, device=model.device)

        pred_boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                xywh = boxes.xywhn[i].cpu().numpy()
                conf = float(boxes.conf[i])
                pred_boxes.append([cls_id, xywh[0], xywh[1], xywh[2],
                                   xywh[3], conf])

        gt_by_class = {0: [], 1: []}
        for gt in gt_labels:
            cls_id = int(gt[0])
            gt_by_class[cls_id].append(gt[1:])

        pred_by_class = {0: [], 1: []}
        for pb in pred_boxes:
            cls_id = int(pb[0])
            pred_by_class[cls_id].append(pb[1:])

        fn_lesion = 0
        fn_non_lesion = 0
        fn_boxes_lesion = []
        fn_boxes_non_lesion = []

        for cls_id in [0, 1]:
            matched = set()
            for pb in pred_by_class[cls_id]:
                best_iou = 0
                best_idx = -1
                for j, gt in enumerate(gt_by_class[cls_id]):
                    if j in matched:
                        continue
                    iou_val = _iou(gt, pb[:4])
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_idx = j
                if best_iou >= iou_threshold and best_idx >= 0:
                    matched.add(best_idx)
            for j, gt in enumerate(gt_by_class[cls_id]):
                if j not in matched:
                    if cls_id == 0:
                        fn_lesion += 1
                        fn_boxes_lesion.append([cls_id] + list(gt))
                    else:
                        fn_non_lesion += 1
                        fn_boxes_non_lesion.append([cls_id] + list(gt))

        total_gt_lesion += len(gt_by_class[0])
        total_gt_non_lesion += len(gt_by_class[1])
        total_fn_lesion += fn_lesion
        total_fn_non_lesion += fn_non_lesion

        tissue = _parse_tissue_from_filename(stem)
        per_image_stats.append({
            "stem": stem,
            "tissue": tissue,
            "gt_lesion": len(gt_by_class[0]),
            "gt_non_lesion": len(gt_by_class[1]),
            "fn_lesion": fn_lesion,
            "fn_non_lesion": fn_non_lesion,
            "fn_total": fn_lesion + fn_non_lesion,
        })

    df = pd.DataFrame(per_image_stats)
    df["fn_rate_lesion"] = df.apply(
        lambda r: r["fn_lesion"] / r["gt_lesion"] if r["gt_lesion"] > 0 else 0,
        axis=1)
    df.sort_values("fn_lesion", ascending=False, inplace=True)

    top_k = df.head(max_samples)

    for _, row in top_k.iterrows():
        stem = row["stem"]
        img = cv2.imread(str(img_dir / f"{stem}.png"))
        if img is None:
            continue
        gt_labels = _load_labels(label_dir, stem)
        results = model(str(img_dir / f"{stem}.png"), verbose=False,
                        conf=conf_threshold, imgsz=256)
        pred_boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xywh = boxes.xywhn[i].cpu().numpy()
                conf = float(boxes.conf[i])
                pred_boxes.append([xywh[0], xywh[1], xywh[2], xywh[3], conf])

        vis = _draw_detections(img, gt_labels, pred_boxes)
        out_path = out_dir / f"fn_{row['fn_lesion']}_{stem}.png"
        cv2.imwrite(str(out_path), vis)

    fn_rate = total_fn_lesion / total_gt_lesion if total_gt_lesion > 0 else 0
    fn_rate_nl = (total_fn_non_lesion / total_gt_non_lesion
                  if total_gt_non_lesion > 0 else 0)

    print(f"\n[假阴性分析] 完成")
    print(f"  病变 GT 总数:   {total_gt_lesion}")
    print(f"  病变漏检数:     {total_fn_lesion}  ({fn_rate:.2%})")
    print(f"  非病变 GT 总数: {total_gt_non_lesion}")
    print(f"  非病变漏检数:   {total_fn_non_lesion}  ({fn_rate_nl:.2%})")
    print(f"  可视化保存至:   {out_dir}")

    return {
        "status": "ok",
        "total_gt_lesion": total_gt_lesion,
        "total_gt_non_lesion": total_gt_non_lesion,
        "total_fn_lesion": total_fn_lesion,
        "total_fn_non_lesion": total_fn_non_lesion,
        "fn_rate_lesion": round(fn_rate, 4),
        "fn_rate_non_lesion": round(fn_rate_nl, 4),
        "top_fn_samples": top_k[["stem", "tissue", "fn_lesion",
                                 "gt_lesion", "fn_rate_lesion"]].head(20).to_dict(
                                     orient="records"),
        "output_dir": str(out_dir),
    }


def evaluate_by_tissue(
    model_path: str,
    data_yaml: str = None,
    split: str = "test",
    output_dir: str = None,
) -> dict:
    """
    按组织类型分组评估模型性能

    对 test 集中 19 种组织类型分别计算 mAP50、recall、precision，
    生成分组柱状图。

    参数:
        model_path: 模型权重路径 (.pt)
        data_yaml: 数据集配置 yaml
        split: 评估划分
        output_dir: 输出目录，默认 runs/analysis/by_tissue

    返回:
        dict: 各组织类型的指标字典 + 图表路径
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = get_project_root()
    if data_yaml is None:
        data_yaml = str(root / "pannuke_yolo" / "data.yaml")
    if output_dir is None:
        output_dir = str(root / "runs" / "analysis" / "by_tissue")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(data_yaml).parent
    img_dir = data_dir / split / "images"
    label_dir = data_dir / split / "labels"

    image_paths = sorted(img_dir.glob("*.png"))
    tissue_groups = defaultdict(list)
    for p in image_paths:
        tissue = _parse_tissue_from_filename(p.stem)
        tissue_groups[tissue].append(p.stem)

    print(f"[组织分组评估] {len(tissue_groups)} 种组织类型")

    model = YOLO(model_path)
    tissue_metrics = {}

    for tissue, stems in sorted(tissue_groups.items()):
        tmp_dir = out_dir / "tmp" / tissue
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_img_dir = tmp_dir / "images"
        tmp_lbl_dir = tmp_dir / "labels"
        tmp_img_dir.mkdir(exist_ok=True)
        tmp_lbl_dir.mkdir(exist_ok=True)

        for stem in stems:
            src_img = img_dir / f"{stem}.png"
            src_lbl = label_dir / f"{stem}.txt"
            if src_img.is_file() and src_lbl.is_file():
                import shutil
                shutil.copy(str(src_img), str(tmp_img_dir / f"{stem}.png"))
                shutil.copy(str(src_lbl), str(tmp_lbl_dir / f"{stem}.txt"))

        yaml_path = tmp_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            f.write(f"path: {tmp_dir}\n")
            f.write("train: images\nval: images\ntest: images\n")
            f.write("nc: 2\nnames: [lesion, non_lesion]\n")

        try:
            val_results = model.val(data=str(yaml_path), imgsz=256,
                                    device=model.device, plots=False,
                                    verbose=False)
            mAP50 = float(val_results.results_dict.get(
                "metrics/mAP50(B)", 0))
            mAP50_95 = float(val_results.results_dict.get(
                "metrics/mAP50-95(B)", 0))
            precision = float(val_results.results_dict.get(
                "metrics/precision(B)", 0))
            recall = float(val_results.results_dict.get(
                "metrics/recall(B)", 0))
        except Exception:
            mAP50, mAP50_95, precision, recall = 0, 0, 0, 0

        gt_counts = {"lesion": 0, "non_lesion": 0}
        for stem in stems:
            labels = _load_labels(label_dir, stem)
            for lbl in labels:
                cls_id = int(lbl[0])
                if cls_id == 0:
                    gt_counts["lesion"] += 1
                else:
                    gt_counts["non_lesion"] += 1

        tissue_metrics[tissue] = {
            "mAP50": round(mAP50, 4),
            "mAP50_95": round(mAP50_95, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "image_count": len(stems),
            "gt_lesion": gt_counts["lesion"],
            "gt_non_lesion": gt_counts["non_lesion"],
        }

        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    tissues = list(tissue_metrics.keys())
    mAP50_values = [tissue_metrics[t]["mAP50"] for t in tissues]
    mAP50_95_values = [tissue_metrics[t]["mAP50_95"] for t in tissues]
    recall_values = [tissue_metrics[t]["recall"] for t in tissues]
    precision_values = [tissue_metrics[t]["precision"] for t in tissues]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    bar_width = 0.6
    colors = plt.cm.tab20(np.linspace(0, 1, len(tissues)))

    for ax, values, title in [
        (axes[0][0], mAP50_values, "mAP@50 by Tissue Type"),
        (axes[0][1], mAP50_95_values, "mAP@50-95 by Tissue Type"),
        (axes[1][0], recall_values, "Recall by Tissue Type"),
        (axes[1][1], precision_values, "Precision by Tissue Type"),
    ]:
        bars = ax.bar(range(len(tissues)), values, bar_width, color=colors)
        ax.set_xticks(range(len(tissues)))
        ax.set_xticklabels(tissues, rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, max(max(values) * 1.15, 0.1))
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    chart_path = out_dir / "tissue_metrics.png"
    fig.savefig(str(chart_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[组织分组评估] 完成，图表: {chart_path}")

    return {
        "status": "ok",
        "tissue_metrics": tissue_metrics,
        "chart_path": str(chart_path),
    }


def analyze_confidence_distribution(
    model_path: str,
    data_yaml: str = None,
    split: str = "test",
    iou_threshold: float = 0.3,
    output_dir: str = None,
) -> dict:
    """
    置信度分布分析：统计预测框的置信度分布，区分 TP/FP

    参数:
        model_path: 模型权重路径
        data_yaml: 数据集配置 yaml
        split: 评估划分
        iou_threshold: TP 判定 IoU 阈值
        output_dir: 输出目录，默认 runs/analysis/confidence

    返回:
        dict: 置信度统计数据 + 图表路径
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = get_project_root()
    if data_yaml is None:
        data_yaml = str(root / "pannuke_yolo" / "data.yaml")
    if output_dir is None:
        output_dir = str(root / "runs" / "analysis" / "confidence")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(data_yaml).parent
    img_dir = data_dir / split / "images"
    label_dir = data_dir / split / "labels"
    image_paths = sorted(img_dir.glob("*.png"))

    model = YOLO(model_path)

    tp_confs = {0: [], 1: []}
    fp_confs = {0: [], 1: []}

    for img_path in image_paths:
        stem = img_path.stem
        gt_labels = _load_labels(label_dir, stem)
        if not gt_labels:
            continue

        results = model(img_path, verbose=False, conf=0.01,
                        imgsz=256, device=model.device)

        pred_boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                xywh = boxes.xywhn[i].cpu().numpy()
                conf = float(boxes.conf[i])
                pred_boxes.append([cls_id, xywh[0], xywh[1], xywh[2],
                                   xywh[3], conf])

        gt_by_class = {0: [], 1: []}
        for gt in gt_labels:
            gt_by_class[int(gt[0])].append(gt[1:])

        for cls_id in [0, 1]:
            matched = set()
            for pb in pred_boxes:
                if int(pb[0]) != cls_id:
                    continue
                best_iou = 0
                best_idx = -1
                for j, gt in enumerate(gt_by_class[cls_id]):
                    if j in matched:
                        continue
                    iou_val = _iou(gt, pb[1:5])
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_idx = j
                if best_iou >= iou_threshold and best_idx >= 0:
                    matched.add(best_idx)
                    tp_confs[cls_id].append(pb[5])
                else:
                    fp_confs[cls_id].append(pb[5])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, cls_id in enumerate([0, 1]):
        ax = axes[idx]
        tp_data = tp_confs[cls_id]
        fp_data = fp_confs[cls_id]
        bins = np.linspace(0, 1, 21)
        ax.hist(tp_data, bins=bins, alpha=0.6, label=f"TP (n={len(tp_data)})",
                color="green", edgecolor="black")
        ax.hist(fp_data, bins=bins, alpha=0.6, label=f"FP (n={len(fp_data)})",
                color="red", edgecolor="black")
        ax.set_title(f"{CLASS_NAMES[cls_id]} Confidence Distribution",
                     fontsize=12)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")
        ax.legend()
    plt.tight_layout()
    chart_path = out_dir / "confidence_distribution.png"
    fig.savefig(str(chart_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    stats = {}
    for cls_id in [0, 1]:
        tp = tp_confs[cls_id]
        fp = fp_confs[cls_id]
        all_confs = tp + fp
        stats[CLASS_NAMES[cls_id]] = {
            "tp_count": len(tp),
            "fp_count": len(fp),
            "tp_mean_conf": round(np.mean(tp), 4) if tp else 0,
            "fp_mean_conf": round(np.mean(fp), 4) if fp else 0,
            "overall_mean_conf": round(np.mean(all_confs), 4) if all_confs
            else 0,
        }

    return {
        "status": "ok",
        "confidence_stats": stats,
        "chart_path": str(chart_path),
    }


def analyze_bbox_size(
    model_path: str,
    data_yaml: str = None,
    split: str = "test",
    iou_threshold: float = 0.3,
    output_dir: str = None,
) -> dict:
    """
    BBox 尺寸分层评估：按面积分 small/medium/large 三组，计算各组 recall

    参数:
        model_path: 模型权重路径
        data_yaml: 数据集配置 yaml
        split: 评估划分
        iou_threshold: TP 判定 IoU 阈值
        output_dir: 输出目录，默认 runs/analysis/bbox_size

    返回:
        dict: 各尺寸组的 recall + 图表路径
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = get_project_root()
    if data_yaml is None:
        data_yaml = str(root / "pannuke_yolo" / "data.yaml")
    if output_dir is None:
        output_dir = str(root / "runs" / "analysis" / "bbox_size")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(data_yaml).parent
    img_dir = data_dir / split / "images"
    label_dir = data_dir / split / "labels"
    image_paths = sorted(img_dir.glob("*.png"))

    model = YOLO(model_path)

    size_groups = {"small": {"gt": 0, "tp": 0},
                   "medium": {"gt": 0, "tp": 0},
                   "large": {"gt": 0, "tp": 0}}

    for img_path in image_paths:
        stem = img_path.stem
        gt_labels = _load_labels(label_dir, stem)
        if not gt_labels:
            continue

        results = model(img_path, verbose=False, conf=0.25,
                        imgsz=256, device=model.device)
        pred_boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                xywh = boxes.xywhn[i].cpu().numpy()
                conf = float(boxes.conf[i])
                pred_boxes.append([cls_id, xywh[0], xywh[1], xywh[2],
                                   xywh[3], conf])

        for gt in gt_labels:
            cls_id = int(gt[0])
            area = _bbox_area(gt[3], gt[4])
            if area < BBOX_SIZE_THRESHOLDS["small"]:
                size_bin = "small"
            elif area < BBOX_SIZE_THRESHOLDS["medium"]:
                size_bin = "medium"
            else:
                size_bin = "large"
            size_groups[size_bin]["gt"] += 1

            matched = False
            for pb in pred_boxes:
                if int(pb[0]) != cls_id:
                    continue
                if _iou(gt[1:], pb[1:5]) >= iou_threshold:
                    matched = True
                    break
            if matched:
                size_groups[size_bin]["tp"] += 1

    sizes = ["small", "medium", "large"]
    recalls = []
    for s in sizes:
        r = (size_groups[s]["tp"] / size_groups[s]["gt"]
             if size_groups[s]["gt"] > 0 else 0)
        recalls.append(r)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(sizes, recalls, color=["#E74C3C", "#F39C12", "#27AE60"],
                  edgecolor="black")
    ax.set_title("Recall by BBox Size", fontsize=13)
    ax.set_ylabel("Recall")
    ax.set_xlabel("BBox Size (normalized area)")
    for b, r in zip(bars, recalls):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{r:.3f}", ha="center", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=recalls[0], color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    chart_path = out_dir / "bbox_size_recall.png"
    fig.savefig(str(chart_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[BBox分层评估] small: {recalls[0]:.3f}, "
          f"medium: {recalls[1]:.3f}, large: {recalls[2]:.3f}")

    return {
        "status": "ok",
        "size_groups": {
            s: {"gt": size_groups[s]["gt"], "tp": size_groups[s]["tp"],
                "recall": round(recalls[i], 4)}
            for i, s in enumerate(sizes)
        },
        "chart_path": str(chart_path),
    }


def generate_report(
    metrics: dict,
    fn_result: dict,
    tissue_result: dict,
    conf_result: dict,
    bbox_result: dict,
    output_dir: str = None,
) -> str:
    """
    生成综合 HTML 报告

    参数:
        metrics: 来自 evaluate.py 的评估指标
        fn_result: 假阴性分析结果
        tissue_result: 组织分组评估结果
        conf_result: 置信度分布结果
        bbox_result: BBox 分层评估结果
        output_dir: 输出目录

    返回:
        str: HTML 报告路径
    """
    root = get_project_root()
    if output_dir is None:
        output_dir = str(root / "runs" / "analysis" / "report")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _rel(p):
        try:
            return str(Path(p).relative_to(out_dir))
        except ValueError:
            return str(p)

    def _td(v, fmt=".4f"):
        if v is None:
            return "<td>N/A</td>"
        return f"<td>{float(v):{fmt}}</td>"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>YOLOv8 病变检测 — 模型评估报告</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; max-width: 1100px; margin: 0 auto;
         padding: 20px; background: #f5f7fa; color: #333; }}
  h1 {{ color: #1a5276; border-bottom: 3px solid #2980b9; padding-bottom: 8px; }}
  h2 {{ color: #2471a3; margin-top: 30px; }}
  .card {{ background: #fff; border-radius: 8px; padding: 16px; margin: 12px 0;
           box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ padding: 8px 12px; border: 1px solid #ddd; text-align: center; }}
  th {{ background: #2980b9; color: #fff; }}
  .badge-ok {{ color: #27ae60; font-weight: bold; }}
  .badge-warn {{ color: #e67e22; font-weight: bold; }}
  .badge-danger {{ color: #e74c3c; font-weight: bold; }}
  img {{ max-width: 100%; border-radius: 4px; margin: 8px 0; box-shadow: 0 1px 3px
         rgba(0,0,0,0.12); }}
</style>
</head>
<body>
<h1>🔬 YOLOv8 病变检测 — 模型评估报告</h1>

<div class="card">
<h2>📊 总体指标 ({metrics.get('split', 'test')} 集)</h2>
<table>
<tr><th>指标</th><th>值</th></tr>
<tr><td>mAP@50</td>{_td(metrics.get('mAP50'))}</tr>
<tr><td>mAP@50-95</td>{_td(metrics.get('mAP50_95'))}</tr>
<tr><td>Precision</td>{_td(metrics.get('precision'))}</tr>
<tr><td>Recall</td>{_td(metrics.get('recall'))}</tr>
<tr><td>病变 mAP@50</td>{_td(metrics.get('lesion_mAP50'))}</tr>
<tr><td>病变 Recall</td>{_td(metrics.get('lesion_recall'), '.3f')}
    <span class="{'badge-ok' if float(metrics.get('lesion_recall',0) or 0) > 0.85 else 'badge-warn'}">
    {'✅' if float(metrics.get('lesion_recall',0) or 0) > 0.85 else '⚠️'}</span></td></tr>
<tr><td>病变 Precision</td>{_td(metrics.get('lesion_precision'))}</tr>
<tr><td>非病变 mAP@50</td>{_td(metrics.get('non_lesion_mAP50'))}</tr>
<tr><td>非病变 Recall</td>{_td(metrics.get('non_lesion_recall'))}</tr>
<tr><td>非病变 Precision</td>{_td(metrics.get('non_lesion_precision'))}</tr>
</table>
</div>

<div class="card">
<h2>🚨 假阴性分析</h2>
<p>病变漏检率: <b>{fn_result.get('fn_rate_lesion', 0):.2%}</b>
   ({fn_result.get('total_fn_lesion', 0)}/{fn_result.get('total_gt_lesion', 0)})</p>
<p>非病变漏检率: <b>{fn_result.get('fn_rate_non_lesion', 0):.2%}</b>
   ({fn_result.get('total_fn_non_lesion', 0)}/{fn_result.get('total_gt_non_lesion', 0)})</p>
<h3>Top 10 漏检严重样本</h3>
<table>
<tr><th>样本</th><th>组织</th><th>病变GT</th><th>病变FN</th><th>漏检率</th></tr>
"""
    top_samples = fn_result.get("top_fn_samples", [])[:10]
    for s in top_samples:
        fn_r = s.get("fn_rate_lesion", 0)
        badge_cls = "badge-danger" if fn_r > 0.3 else ("badge-warn"
                     if fn_r > 0.1 else "badge-ok")
        html += (f"<tr><td>{s['stem']}</td><td>{s['tissue']}</td>"
                 f"<td>{s['gt_lesion']}</td><td>{s['fn_lesion']}</td>"
                 f"<td class='{badge_cls}'>{fn_r:.1%}</td></tr>")
    html += "</table></div>\n"

    if "chart_path" in tissue_result:
        html += (f'<div class="card"><h2>🧬 组织分组评估</h2>'
                 f'<img src="{_rel(tissue_result["chart_path"])}"></div>\n')

    if "chart_path" in conf_result:
        html += (f'<div class="card"><h2>📈 置信度分布</h2>'
                 f'<img src="{_rel(conf_result["chart_path"])}"></div>\n')

    if "chart_path" in bbox_result:
        html += (f'<div class="card"><h2>📐 BBox 尺寸分层 Recall</h2>'
                 f'<img src="{_rel(bbox_result["chart_path"])}"></div>\n')

    html += "</body></html>"

    report_path = out_dir / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[报告] 已生成: {report_path}")
    return str(report_path)


def run_full_analysis(
    model_path: str,
    data_yaml: str = None,
    split: str = "test",
    output_dir: str = None,
) -> dict:
    """
    运行完整的五项分析

    参数:
        model_path: 模型权重路径
        data_yaml: 数据集配置 yaml
        split: 评估划分
        output_dir: 输出根目录

    返回:
        dict: 汇总所有分析结果
    """
    root = get_project_root()
    if output_dir is None:
        output_dir = str(root / "runs" / "analysis")

    print("=" * 60)
    print("YOLOv8 模型深度分析")
    print("=" * 60)

    print("\n[1/4] 假阴性分析...")
    fn_result = analyze_false_negatives(model_path, data_yaml, split,
                                        output_dir=str(
                                            Path(output_dir) / "false_negatives"))

    print("\n[2/4] 组织分组评估...")
    tissue_result = evaluate_by_tissue(model_path, data_yaml, split,
                                       output_dir=str(
                                           Path(output_dir) / "by_tissue"))

    print("\n[3/4] 置信度分布分析...")
    conf_result = analyze_confidence_distribution(
        model_path, data_yaml, split,
        output_dir=str(Path(output_dir) / "confidence"))

    print("\n[4/4] BBox 尺寸分层评估...")
    bbox_result = analyze_bbox_size(model_path, data_yaml, split,
                                    output_dir=str(
                                        Path(output_dir) / "bbox_size"))

    simple_metrics = {
        "split": split,
        "model_path": model_path,
    }
    report_path = generate_report(simple_metrics, fn_result,
                                  tissue_result, conf_result, bbox_result,
                                  output_dir=str(Path(output_dir) / "report"))

    return {
        "status": "ok",
        "false_negatives": fn_result,
        "by_tissue": tissue_result,
        "confidence": conf_result,
        "bbox_size": bbox_result,
        "report_path": report_path,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 深度模型分析")
    parser.add_argument("--model", type=str, required=True,
                        help="模型权重路径 (.pt)")
    parser.add_argument("--data", type=str, default=None,
                        help="数据集 yaml 路径")
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "val"], help="评估划分")
    parser.add_argument("--output", type=str, default=None,
                        help="输出根目录")
    parser.add_argument("--analysis", type=str, default="all",
                        choices=["all", "fn", "tissue", "confidence",
                                 "bbox", "report"],
                        help="分析类型")

    args = parser.parse_args()

    if args.analysis == "all":
        result = run_full_analysis(args.model, args.data, args.split,
                                   args.output)
    elif args.analysis == "fn":
        result = analyze_false_negatives(args.model, args.data, args.split,
                                         output_dir=args.output)
    elif args.analysis == "tissue":
        result = evaluate_by_tissue(args.model, args.data, args.split,
                                    output_dir=args.output)
    elif args.analysis == "confidence":
        result = analyze_confidence_distribution(args.model, args.data,
                                                 args.split,
                                                 output_dir=args.output)
    elif args.analysis == "bbox":
        result = analyze_bbox_size(args.model, args.data, args.split,
                                   output_dir=args.output)
    elif args.analysis == "report":
        result = generate_report({}, {}, {}, {}, {}, output_dir=args.output)

    print("\n" + json.dumps(result, ensure_ascii=False, indent=2,
                            default=str))
