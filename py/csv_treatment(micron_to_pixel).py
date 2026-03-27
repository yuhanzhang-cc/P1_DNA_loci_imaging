import csv
import os

# 路径
csv_dir = r'image_annotation\data\raw_csv'
output_dir = r'image_annotation\data\raw_csv_pixel'
os.makedirs(output_dir, exist_ok=True)

# 手动读取 pixel_size_map.csv
pixel_map_dict = {}
with open('pixel_size_map.csv', 'r', encoding='gbk') as f:
    lines = f.readlines()
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) >= 2:
            filename = parts[0].strip('"')
            pixel_width = float(parts[1])
            pixel_map_dict[filename] = pixel_width

print(f'读取到 {len(pixel_map_dict)} 个文件的像素大小')

# 遍历所有 CSV
for file in os.listdir(csv_dir):
    if not file.endswith('.csv'):
        continue
    
    base = file.replace('.csv', '')
    tif_name = base + '.tif'
    pixel_size = pixel_map_dict.get(tif_name, 0.16)
    
    print(f'处理 {file}，像素大小: {pixel_size}')
    
    input_path = os.path.join(csv_dir, file)
    output_path = os.path.join(output_dir, file)
    
    # 手动读取 CSV
    with open(input_path, 'r') as f:
        reader = csv.reader(f)
        # 跳过前三行
        for _ in range(3):
            next(reader, None)
        
        # 收集数据
        x_vals = []
        y_vals = []
        for row in reader:
            if len(row) < 6:
                continue
            try:
                # 第5列是 X，第6列是 Y（索引 4 和 5）
                x = float(row[4])
                y = float(row[5])
                x_vals.append(x)
                y_vals.append(y)
            except ValueError:
                # 跳过无法转换的行
                continue
    
    if not x_vals:
        print(f'  警告：{file} 没有有效数据')
        continue
    
    # 微米转像素
    x_px = [x / pixel_size for x in x_vals]
    y_px = [y / pixel_size for y in y_vals]
    
    # 保存为两列，无表头
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for xi, yi in zip(x_px, y_px):
            writer.writerow([xi, yi])
    
    print(f'  已保存 {output_path}，共 {len(x_vals)} 个点')

print('全部完成！')