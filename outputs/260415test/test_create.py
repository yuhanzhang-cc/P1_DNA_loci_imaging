import numpy as np
import tifffile
from pathlib import Path

# 加载数据
data = np.load('dataset.npz', allow_pickle=True)
x_test = data['x_test']

# 创建输出文件夹
out_dir = Path('test_images_tif')
out_dir.mkdir(exist_ok=True)

# 保存每张测试图为 tif
for i, img in enumerate(x_test):
    # 确保数据类型是 uint16（如果原来是 float32，需要转换）
    if img.dtype != np.uint16:
        img = img.astype(np.uint16)
    tifffile.imwrite(out_dir / f'test_image_{i}.tif', img)
    print(f"Saved test_image_{i}.tif")

print(f"共保存 {len(x_test)} 张测试图到 {out_dir}")