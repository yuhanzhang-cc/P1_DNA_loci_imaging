import os
import csv

src_dir = r'image_annotation\data\raw_csv'
pixel_dir = r'image_annotation\data\raw_csv_pixel'
out_dir = r'image_annotation\data\raw_csv_pixel_replaced'

os.makedirs(out_dir, exist_ok=True)

for file in os.listdir(src_dir):
    if not file.endswith('.csv'):
        continue
    
    src_path = os.path.join(src_dir, file)
    pixel_path = os.path.join(pixel_dir, file)
    out_path = os.path.join(out_dir, file)
    
    if not os.path.exists(pixel_path):
        print(f'警告: {file} 没有像素坐标文件，跳过')
        continue
    
    # 读取原始 CSV 的所有行
    with open(src_path, 'r', encoding='utf-8') as f:
        src_lines = f.readlines()
    
    # 读取像素坐标数据（两列数字）
    with open(pixel_path, 'r') as f:
        pixel_lines = [line.strip().split(',') for line in f if line.strip()]
    
    # 找到数据开始的行（第一个以 'ID' 开头的行）
    data_start = 0
    for i, line in enumerate(src_lines):
        if line.startswith('ID'):
            data_start = i
            break
    
    # 写新文件
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        # 先写表头部分（前 data_start 行）
        for i in range(data_start):
            f.write(src_lines[i])
        
        # 写数据行，替换第5、6列
        for i, pixel_row in enumerate(pixel_lines):
            if i < len(src_lines) - data_start:
                original_line = src_lines[data_start + i]
                parts = original_line.strip().split(',')
                if len(parts) >= 6:
                    # 替换第5、6列（索引4和5）
                    parts[4] = pixel_row[0]
                    parts[5] = pixel_row[1]
                    f.write(','.join(parts) + '\n')
                else:
                    f.write(original_line)
    
    print(f'处理完成: {file}')

print('全部完成！')