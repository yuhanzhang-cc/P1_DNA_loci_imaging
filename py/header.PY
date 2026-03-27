import os
import csv

csv_dir = r'image_annotation\data\raw_csv_pixel'
output_dir = r'image_annotation\data\raw_csv_pixel_with_header'
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(csv_dir):
    if not file.endswith('.csv'):
        continue
    
    input_path = os.path.join(csv_dir, file)
    output_path = os.path.join(output_dir, file)
    
    with open(input_path, 'r') as f_in, open(output_path, 'w', newline='') as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        writer.writerow(['POSITION_X', 'POSITION_Y'])
        for row in reader:
            writer.writerow(row)
    
    print(f'处理完成: {file}')

print('全部完成！')