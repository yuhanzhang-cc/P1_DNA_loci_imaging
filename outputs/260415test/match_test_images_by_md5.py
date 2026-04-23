import hashlib
import pandas as pd
from pathlib import Path

# 文件夹路径
test_tif_dir = Path(r'D:\Projects\P1_DNA_loci_imaging\outputs\260415test\test_images_tif')
raw_tif_dir = Path(r'D:\Projects\P1_DNA_loci_imaging\data\raw\tif')

# 存储结果
matches = []

# 1. 计算所有测试图的 MD5
print("正在计算测试图 MD5...")
test_md5_map = {}  # {md5: test_image_index}
for tif_path in sorted(test_tif_dir.glob('*.tif')):
    with open(tif_path, 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    # 提取索引，如 test_image_0.tif -> 0
    idx = int(tif_path.stem.split('_')[-1])
    test_md5_map[md5] = idx
    print(f"  {tif_path.name} -> {md5[:16]}...")

# 2. 计算所有原始图的 MD5 并比对
print("\n正在计算原始图 MD5 并比对...")
raw_md5_map = {}  # {md5: raw_filename}
for tif_path in sorted(raw_tif_dir.glob('*.tif')):
    with open(tif_path, 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    raw_md5_map[md5] = tif_path.name
    print(f"  {tif_path.name} -> {md5[:16]}...")

# 3. 比对，找出对应关系
print("\n正在比对 MD5...")
for md5, test_idx in test_md5_map.items():
    if md5 in raw_md5_map:
        matches.append({
            'test_image_index': test_idx,
            'raw_filename': raw_md5_map[md5],
            'md5': md5
        })
    else:
        matches.append({
            'test_image_index': test_idx,
            'raw_filename': 'NOT_FOUND',
            'md5': md5
        })

# 4. 保存结果
df = pd.DataFrame(matches)
df = df.sort_values('test_image_index')
df.to_csv('test_image_to_raw_mapping.csv', index=False)

print("\n=== 匹配结果 ===")
print(df.to_string())

print(f"\n✅ 结果已保存到: test_image_to_raw_mapping.csv")