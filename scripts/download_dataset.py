"""
PanNuke 数据集下载脚本（带进度条）
=================================
功能: 自动下载 PanNuke 三个 fold，解压，处理 npy → 单张 PNG + .npy 掩膜
依赖: pip install tqdm
输出:
    {data_dir}/
        images/    ← 256×256 PNG 图像
        masks/     ← (6, 256, 256) .npy 掩膜
"""

import os
import re
import shutil
import zipfile
import warnings
from pathlib import Path

import cv2
import numpy as np
import requests
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# ==================== 配置 ====================
DATA_DIR = os.path.join(os.getcwd(), "pannuke_data")
FOLDS = [1, 2, 3]
BASE_URL = "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_{}.zip"


def _create_session(max_retries=5):
    """
    创建带重试策略的 requests Session

    参数:
        max_retries: 最大重试次数

    返回:
        requests.Session 对象
    """
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=2,
        status_forcelist=[413, 429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def download_file(url, dest_path, max_retries=100):
    """
    带进度条和断点续传的文件下载（自动重试 SSL 断连）

    参数:
        url: 下载地址
        dest_path: 保存路径
        max_retries: 最大重试次数

    异常:
        requests.RequestException: 超过最大重试次数后网络请求仍失败
    """
    filename = os.path.basename(dest_path)

    for attempt in range(max_retries):
        try:
            resume_pos = 0
            if os.path.exists(dest_path):
                resume_pos = os.path.getsize(dest_path)

            headers = {}
            if resume_pos > 0:
                headers['Range'] = f'bytes={resume_pos}-'

            session = _create_session()
            response = session.get(url, stream=True, headers=headers,
                                   timeout=(10, 60), verify=False)

            if response.status_code not in (200, 206):
                raise requests.RequestException(
                    f"HTTP {response.status_code}")

            total_size = resume_pos + int(
                response.headers.get('content-length', 0))
            mode = 'ab' if response.status_code == 206 else 'wb'

            with open(dest_path, mode) as f:
                with tqdm(initial=resume_pos, total=total_size, unit='B',
                          unit_scale=True, unit_divisor=1024, miniters=1,
                          desc=filename) as pbar:
                    for chunk in response.iter_content(
                            chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            return

        except Exception as e:
            if attempt < max_retries - 1:
                wait = 5
                tqdm.write(f"[重试 {attempt + 1}/{max_retries}] {e} "
                           f"— {wait}s 后重试...")
                import time
                time.sleep(wait)
            else:
                raise


def download_pannuke(data_dir):
    """
    下载 PanNuke 三个 fold 的 zip 文件
    
    参数:
        data_dir: 数据存放目录
    
    返回:
        下载的 zip 文件路径列表
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_paths = []
    for fold_ix in FOLDS:
        zip_path = data_dir / f"fold_{fold_ix}.zip"
        extracted_dir = data_dir / f"Fold {fold_ix}"

        if extracted_dir.is_dir():
            print(f"[跳过] Fold {fold_ix} 已存在本地文件，无需重新下载")
        else:
            url = BASE_URL.format(fold_ix)
            print(f"[下载] Fold {fold_ix} ← {url}")
            download_file(url, str(zip_path))

        zip_paths.append(str(zip_path))
    return zip_paths


def extract_zips(zip_paths):
    """
    解压所有 zip 文件
    
    参数:
        zip_paths: zip 文件路径列表
    """
    for zip_path in zip_paths:
        extract_dir = os.path.dirname(zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            file_list = zf.namelist()
            for f in tqdm(file_list, desc=f"解压 {os.path.basename(zip_path)}"):
                zf.extract(f, extract_dir)


def process_fold_npy(data_dir):
    """
    将每个 fold 的 images.npy / masks.npy 拆分为单张 PNG 和 .npy 掩膜
    逻辑等同 pathml 的 _process_downloaded_pannuke()
    
    参数:
        data_dir: 数据目录
    """
    data_dir = Path(data_dir)
    imdir = data_dir / "images"
    maskdir = data_dir / "masks"

    imdir.mkdir(exist_ok=True)
    maskdir.mkdir(exist_ok=True)

    for fold_ix in tqdm(FOLDS, desc="处理 npy 文件"):
        fold_dir = data_dir / f"Fold {fold_ix}"
        ims_path = fold_dir / "images" / f"fold{fold_ix}" / "images.npy"
        masks_path = fold_dir / "masks" / f"fold{fold_ix}" / "masks.npy"
        types_path = fold_dir / "images" / f"fold{fold_ix}" / "types.npy"

        if not all(p.is_file() for p in [ims_path, masks_path, types_path]):
            raise FileNotFoundError(f"缺少必要文件: fold {fold_ix}")

        ims_fold = np.load(ims_path, mmap_mode="r")
        masks_fold = np.load(masks_path, mmap_mode="r")
        types_fold = np.load(types_path, mmap_mode="r")

        # 转换 masks 维度: (B, H, W, C) → (B, C, H, W)
        masks_fold = np.moveaxis(masks_fold, 3, 1)

        fold_size = len(types_fold)
        for j in range(fold_size):
            im = ims_fold[j, ...]
            mask = masks_fold[j, ...]
            tissue_type = types_fold[j]
            tissue_type = re.sub(pattern="_", repl="-", string=tissue_type)

            file_basename = f"fold{fold_ix}_{j}_{tissue_type}"
            im_fname = str((imdir / f"{file_basename}.png").resolve())
            mask_fname = str((maskdir / f"{file_basename}.npy").resolve())

            cv2.imwrite(im_fname, im)
            np.save(mask_fname, mask)


# ==================== 主流程 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("PanNuke 数据集下载")
    print(f"目标目录: {DATA_DIR}")
    print(f"预计大小: ~30 GB (含 zip + 处理后文件)")
    print("=" * 60)

    # 1. 下载三个 fold 的 zip
    print("\n[1/4] 下载 fold_1 ~ fold_3 ...")
    zip_paths = download_pannuke(DATA_DIR)

    # 2. 解压
    print("\n[2/4] 解压 zip 文件 ...")
    extract_zips(zip_paths)

    # 3. 处理 npy → 单张图像/掩膜
    print("\n[3/4] 拆分 npy 为单张 PNG 与 .npy 掩膜 ...")
    process_fold_npy(DATA_DIR)

    print("\n" + "=" * 60)
    print(f"✅ 下载完成！数据已保存至: {DATA_DIR}")
    print(f"   图像: {DATA_DIR}/images/  ({len(list(Path(DATA_DIR, 'images').glob('*.png')))} 张)")
    print(f"   掩膜: {DATA_DIR}/masks/  ({len(list(Path(DATA_DIR, 'masks').glob('*.npy')))} 个)")
    print("=" * 60)