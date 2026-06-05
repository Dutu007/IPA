"""
YOLOv8 病变检测模型训练脚本
===========================
功能:
  1. 加载 pannuke_yolo 数据集，训练病变/非病变二分类检测模型
  2. 训练完成后在验证集上评估，记录核心指标
  3. 可选导出 ONNX 格式供推理使用

使用方式:
  python -m scripts.yolo_training.train

注意:
  本脚本为开发期工具，产出模型文件供 Web 应用推理使用，不参与 Web 运行时。
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO


def get_project_root():
    """
    获取项目根目录（IPA/）

    返回:
        Path: 项目根目录路径
    """
    return Path(__file__).resolve().parent.parent.parent


def train_yolo(
    data_yaml: str = None,
    model_name: str = None,
    epochs: int = 50,
    imgsz: int = 256,
    batch: int = 64,
    device: str = "0",
    project: str = None,
    export_onnx: bool = False,
) -> dict:
    """
    训练 YOLOv8 是否病变检测模型

    参数:
        data_yaml: 数据集配置 yaml 路径，默认 pannuke_yolo/data.yaml
        model_name: 预训练模型路径，默认 models/yolov8m.pt
        epochs: 训练轮数
        imgsz: 输入图像尺寸
        batch: 批次大小
        device: 训练设备 ("0"=GPU0, "cpu"=CPU)
        project: 训练输出目录，默认 runs/detect
        export_onnx: 训练完成后是否导出 ONNX

    返回:
        dict: {
            "status": "ok" | "error",
            "message": "描述信息",
            "model_name": "yolov8m",
            "best_model": "模型路径",
            "onnx_model": "ONNX路径(若导出)",
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
            "epochs": int,
            "train_time": "训练耗时"
        }
    """
    root = get_project_root()

    if data_yaml is None:
        data_yaml = str(root / "pannuke_yolo" / "data.yaml")
    else:
        data_yaml = str(Path(data_yaml).resolve())

    if model_name is None:
        model_name = str(root / "models" / "yolov8m.pt")

    model_path = Path(model_name)
    if not model_path.is_file():
        model_short = model_path.stem
        print(f"[训练] 本地未找到 {model_path.name}，下载到 {model_path.parent}...")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_model = YOLO(model_path.name)
        tmp_model.save(str(model_path))
        model_name = str(model_path.resolve())
        tmp_download = Path(model_path.name)
        if tmp_download.is_file():
            tmp_download.unlink()
        print(f"[训练] 已保存: {model_name}")
    else:
        model_name = str(model_path.resolve())
        model_short = model_path.stem
        print(f"[训练] 使用本地模型: {model_name}")

    if project is None:
        project = str(root / "runs" / "detect")
    else:
        project = str(Path(project).resolve())

    if not Path(data_yaml).is_file():
        return {
            "status": "error",
            "message": f"数据集配置文件不存在: {data_yaml}",
        }

    print(f"[训练] 数据集配置: {data_yaml}")
    print(f"[训练] 模型: {model_name}")
    print(f"[训练] Epochs: {epochs}, Imgsz: {imgsz}, Batch: {batch}")
    print(f"[训练] 设备: {device}, 输出目录: {project}")

    start_time = datetime.now()

    model = YOLO(model_name)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        exist_ok=True,
        verbose=True,
    )

    end_time = datetime.now()
    train_duration = str(end_time - start_time)

    best_model_path = str(Path(results.save_dir) / "weights" / "best.pt")

    metrics = _extract_metrics(results)

    onnx_path = None
    if export_onnx and Path(best_model_path).is_file():
        onnx_path = _export_onnx(best_model_path)

    return {
        "status": "ok",
        "message": f"{model_short} 训练完成",
        "model_name": model_short,
        "best_model": best_model_path,
        "onnx_model": onnx_path,
        "mAP50": metrics.get("mAP50"),
        "mAP50_95": metrics.get("mAP50-95"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "lesion_recall": metrics.get("lesion_recall"),
        "lesion_precision": metrics.get("lesion_precision"),
        "lesion_mAP50": metrics.get("lesion_mAP50"),
        "non_lesion_recall": metrics.get("non_lesion_recall"),
        "non_lesion_precision": metrics.get("non_lesion_precision"),
        "non_lesion_mAP50": metrics.get("non_lesion_mAP50"),
        "epochs": epochs,
        "train_time": train_duration,
    }


def _extract_metrics(results) -> dict:
    """
    从训练结果中提取关键指标

    参数:
        results: model.train() 的返回值

    返回:
        dict: 包含 mAP、precision、recall 及各分类指标的字典
    """
    metrics = {}
    try:
        metrics["mAP50"] = float(results.results_dict.get("metrics/mAP50(B)", 0))
        metrics["mAP50-95"] = float(results.results_dict.get("metrics/mAP50-95(B)", 0))
        metrics["precision"] = float(results.results_dict.get("metrics/precision(B)", 0))
        metrics["recall"] = float(results.results_dict.get("metrics/recall(B)", 0))
    except Exception:
        pass

    try:
        csv_path = Path(results.save_dir) / "results.csv"
        if csv_path.is_file():
            import pandas as pd
            df = pd.read_csv(csv_path)
            last_row = df.iloc[-1]
            cols = df.columns.tolist()
            for col in cols:
                col_lower = col.strip().lower()
                if "lesion" in col_lower or "non_lesion" in col_lower:
                    val = last_row[col]
                    key = col.strip().replace("metrics/", "").replace("(B)", "")
                    try:
                        metrics[key] = float(val)
                    except (ValueError, TypeError):
                        metrics[key] = val
    except Exception:
        pass

    return metrics


def _export_onnx(model_path: str) -> str:
    """
    将训练好的模型导出为 ONNX 格式

    参数:
        model_path: .pt 模型文件路径

    返回:
        str: 导出的 ONNX 文件路径

    异常:
        RuntimeError: 导出失败时抛出
    """
    model = YOLO(model_path)
    onnx_path = model.export(format="onnx", simplify=True)
    print(f"[导出] ONNX 模型已保存: {onnx_path}")
    return str(onnx_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 病变检测模型训练")
    parser.add_argument("--data", type=str, default=None,
                        help="数据集 yaml 路径")
    parser.add_argument("--model", type=str, default=None,
                        help="预训练模型路径，默认 models/yolov8m.pt")
    parser.add_argument("--epochs", type=int, default=50,
                        help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=256,
                        help="输入图像尺寸")
    parser.add_argument("--batch", type=int, default=64,
                        help="批次大小")
    parser.add_argument("--device", type=str, default="0",
                        help="训练设备")
    parser.add_argument("--export-onnx", action="store_true",
                        help="训练完成后导出 ONNX")
    parser.add_argument("--output", type=str, default=None,
                        help="结果 JSON 输出路径（可选）")

    args = parser.parse_args()

    result = train_yolo(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        export_onnx=args.export_onnx,
    )

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"模型:      {result.get('model_name', 'N/A')}")
    print(f"最佳权重:  {result.get('best_model', 'N/A')}")
    if result.get('onnx_model'):
        print(f"ONNX导出:  {result['onnx_model']}")
    print(f"训练耗时:  {result.get('train_time', 'N/A')}")
    print()
    print(f"  mAP@50:     {result.get('mAP50', 'N/A')}")
    print(f"  mAP@50-95:  {result.get('mAP50_95', 'N/A')}")
    print(f"  Precision:  {result.get('precision', 'N/A')}")
    print(f"  Recall:     {result.get('recall', 'N/A')}")
    if result.get('lesion_recall') is not None:
        print(f"  病变 Recall: {result['lesion_recall']:.4f}")
        print(f"  病变 mAP50:  {result['lesion_mAP50']:.4f}")
    if result.get('non_lesion_recall') is not None:
        print(f"  非病变 Recall: {result['non_lesion_recall']:.4f}")
        print(f"  非病变 mAP50:  {result['non_lesion_mAP50']:.4f}")
    print("=" * 60)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n结果已保存: {output_path}")

    if result["status"] == "error":
        sys.exit(1)
