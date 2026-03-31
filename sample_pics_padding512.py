import numpy as np
from PIL import Image
from pathlib import Path

# 路径设置
RAW_TIF_DIR = Path(r'D:\Projects\P1_DNA_loci_imaging\sample pics\originals')
OUT_TIF_DIR = Path(r'D:\Projects\P1_DNA_loci_imaging\sample pics\originals_512_padding')
OUT_TIF_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 512

def pad_image_to_512(img):
    h, w = img.shape
    pad_top = (TARGET - h) // 2
    pad_bottom = TARGET - h - pad_top
    pad_left = (TARGET - w) // 2
    pad_right = TARGET - w - pad_left
    padded = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    return padded, pad_left, pad_top

# 处理所有 tif
for tif_path in RAW_TIF_DIR.glob('*.tif'):
    img = np.array(Image.open(tif_path))
    padded_img, dx, dy = pad_image_to_512(img)
    out_name = tif_path.stem + '_pad512.tif'
    Image.fromarray(padded_img).save(OUT_TIF_DIR / out_name)
    print(f'{tif_path.name} -> {out_name}, offset: dx={dx}, dy={dy}')