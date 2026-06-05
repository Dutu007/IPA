"""
YOLOv8 模型评估脚本
==================
功能:
  1. 在 test 集上评估训练好的模型，输出详细指标
  2. 生成混淆矩阵、PR 曲线等可视化图表
  3. 对比多个模型的指标

使用方式:
  python -m scripts.yolo_training.evaluate --model runs/detect/train/weights/best.pt

注意:
  本脚本为开发期工具，用于验证模型质量，不参与 Web 运行时。
"""

import os
import sys
import json
from pathlib import Path

from ultralytics import YOLO


def get_project_root():
    """
    获取项目根目录（IPA/）

    返回:
        Path: 项目根目录路径
    """
    return Path(__file__).resolve().parent.parent.parent


def evaluate_model(
    model_path: str,
    data_yaml: str = None,
    imgsz: int = 256,
    batch: int = 16,
    device: str = "0",
    split: str = "test",
    save_plots: bool = True,
    project: str = None,
) -> dict:
    """
    评估训练好的 YOLOv8 模型

    参数:
        model_path: 模型权重路径 (.pt)
        data_yaml: 数据集配置 yaml 路径，默认 pannuke_yolo/data.yaml
        imgsz: 输入图像尺寸
        batch: 批次大小
        device: 评估设备 ("0"=GPU0, "cpu"=CPU)
        split: 评估数据集划分 ("test" | "val")
        save_plots: 是否保存可视化图表
        project: 评估输出目录，默认 runs/detect/evaluate

    返回:
        dict: {
            "status": "ok" | "error",
            "message": "描述信息",
            "model_path": "模型路径",
            "split": "test" | "val",
            "mAP50": float,
            "mAP50_95": float,
            "precision": float,
            "recall": float,
            "lesion_recall": float,
            "lesion_precision": float,
            "lesion_mAP50": float,
            "non_lesion_recall": float,
            "non_lesion_precision": float,
            "non_lesion_mAP50": float,
        }
    """
    root = get_project_root()

    if data_yaml is None:
        data_yaml = str(root / "pannuke_yolo" / "data.yaml")
    else:
        data_yaml = str(Path(data_yaml).resolve())

    model_path = str(Path(model_path).resolve())

    if project is None:
        project = str(root / "runs" / "detect" / "evaluate")
    else:
        project = str(Path(project).resolve())

    if not Path(model_path).is_file():
        return {
            "status": "error",
            "message": f"模型文件不存在: {model_path}",
        }

    if not Path(data_yaml).is_file():
        return {
            "status": "error",
            "message": f"数据集配置文件不存在: {data_yaml}",
        }

    print(f"[评估] 模型: {model_path}")
    print(f"[评估] 数据集: {data_yaml}, 划分: {split}")
    print(f"[评估] Imgsz: {imgsz}, Batch: {batch}, Device: {device}")

    model = YOLO(model_path)

    results = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=batch,
        device=device,
        split=split,
        plots=save_plots,
        project=project,
        exist_ok=True,
    )

    metrics = _parse_val_results(results, model_path, split)

    print("\n" + "=" * 60)
    print(f"评估结果 ({split} 集):")
    print(f"  mAP@50:     {metrics.get('mAP50', 'N/A')}")
    print(f"  mAP@50-95:  {metrics.get('mAP50_95', 'N/A')}")
    print(f"  Precision:  {metrics.get('precision', 'N/A')}")
    print(f"  Recall:     {metrics.get('recall', 'N/A')}")
    print(f"  病变 Recall: {metrics.get('lesion_recall', 'N/A')}")
    print(f"  病变 mAP50:  {metrics.get('lesion_mAP50', 'N/A')}")
    print(f"  非病变 Recall: {metrics.get('non_lesion_recall', 'N/A')}")
    print(f"  非病变 mAP50:  {metrics.get('non_lesion_mAP50', 'N/A')}")
    print("=" * 60)

    return {
        "status": "ok",
        "message": "评估完成",
        **metrics,
    }


def _parse_val_results(results, model_path: str, split: str) -> dict:
    """
    解析 model.val() 的返回结果，提取结构化指标

    参数:
        results: model.val() 的返回值
        model_path: 模型路径
        split: 数据集划分名称

    返回:
        dict: 评估指标
    """
    metrics = {
        "model_path": model_path,
        "split": split,
    }

    try:
        metrics["mAP50"] = float(results.results_dict.get("metrics/mAP50(B)", 0))
        metrics["mAP50_95"] = float(results.results_dict.get("metrics/mAP50-95(B)", 0))
        metrics["precision"] = float(results.results_dict.get("metrics/precision(B)", 0))
        metrics["recall"] = float(results.results_dict.get("metrics/recall(B)", 0))
    except Exception:
        pass

    try:
        results_dict = results.results_dict
        for key in ["lesion_recall", "lesion_precision", "lesion_mAP50",
                     "non_lesion_recall", "non_lesion_precision", "non_lesion_mAP50"]:
            for rk, rv in results_dict.items():
                rk_clean = rk.replace("metrics/", "").replace("(B)", "")
                if rk_clean == key:
                    try:
                        metrics[key] = float(rv)
                    except (ValueError, TypeError):
                        metrics[key] = rv
    except Exception:
        pass

    try:
        if hasattr(results, "speed") and results.speed:
            speeds = results.speed
            metrics["preprocess_ms"] = float(speeds.get("preprocess", 0))
            metrics["inference_ms"] = float(speeds.get("inference", 0))
            metrics["postprocess_ms"] = float(speeds.get("postprocess", 0))
    except Exception:
        pass

    return metrics


def compare_models(model_paths: list, data_yaml: str = None, **kwargs) -> list:
    """
    批量对比多个模型

    参数:
        model_paths: 模型路径列表
        data_yaml: 数据集配置 yaml 路径
        **kwargs: 传递给 evaluate_model 的其他参数

    返回:
        list[dict]: 每个模型的评估结果
    """
    results = []
    for mp in model_paths:
        print(f"\n{'='*60}")
        print(f"[对比] 评估模型: {mp}")
        result = evaluate_model(model_path=mp, data_yaml=data_yaml, **kwargs)
        results.append(result)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 模型评估")
    parser.add_argument("--model", type=str, required=True,
                        help="模型权重路径 (.pt)")
    parser.add_argument("--data", type=str, default=None,
                        help="数据集 yaml 路径")
    parser.add_argument("--imgsz", type=int, default=256,
                        help="输入图像尺寸")
    parser.add_argument("--batch", type=int, default=16,
                        help="批次大小")
    parser.add_argument("--device", type=str, default="0",
                        help="评估设备")
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "val"],
                        help="评估数据集划分")
    parser.add_argument("--no-plots", action="store_true",
                        help="不保存可视化图表")
    parser.add_argument("--output", type=str, default=None,
                        help="结果 JSON 输出路径（可选）")

    args = parser.parse_args()

    result = evaluate_model(
        model_path=args.model,
        data_yaml=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,
        save_plots=not args.no_plots,
    )

    print("\n" + json.dumps(result, ensure_ascii=False, indent=2, default=str))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n结果已保存: {output_path}")

    if result["status"] == "error":
        sys.exit(1)
