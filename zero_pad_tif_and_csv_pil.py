from pathlib import Path
import csv
import numpy as np
from PIL import Image

TARGET = 512
RAW_TIF_DIR = Path('image_annotation/data/raw_tif')
RAW_CSV_DIR = Path('image_annotation/data/raw_csv_pixel_replaced')
OUT_TIF_DIR = Path('image_annotation/data/padded_tif_512')
OUT_CSV_DIR = Path('image_annotation/data/padded_csv_512')
SUMMARY_CSV = Path('image_annotation/data/pad_summary.csv')

OUT_TIF_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)


def find_csv(tif_path: Path):
    candidates = [
        RAW_CSV_DIR / f'{tif_path.stem}.csv',
        RAW_CSV_DIR / f'{tif_path.stem}_allspots.csv',
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def read_tif(path: Path) -> np.ndarray:
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim != 2:
        raise ValueError(f'Only 2D grayscale tif is supported: {path.name}, got shape {arr.shape}')
    # Convert big-endian arrays from TIFF into native uint16/int32 etc. for safe padding/saving.
    if arr.dtype.byteorder == '>':
        arr = arr.astype(arr.dtype.newbyteorder('='))
    return arr


def save_tif(path: Path, arr: np.ndarray):
    Image.fromarray(arr).save(path)


def pad_image_to_512(img: np.ndarray):
    h, w = img.shape
    if h > TARGET or w > TARGET:
        return None
    pad_top = (TARGET - h) // 2
    pad_bottom = TARGET - h - pad_top
    pad_left = (TARGET - w) // 2
    pad_right = TARGET - w - pad_left
    out = np.pad(
        img,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=0,
    )
    return out, pad_left, pad_top


def pad_csv(csv_in: Path, csv_out: Path, dx: int, dy: int):
    with open(csv_in, 'r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f))

    if len(rows) < 5:
        raise ValueError(f'CSV format unexpected: {csv_in}')

    header_rows = rows[:4]
    colnames = rows[4]
    data_rows = rows[5:]

    try:
        x_idx = colnames.index('POSITION_X')
        y_idx = colnames.index('POSITION_Y')
    except ValueError as e:
        raise ValueError(f'POSITION_X / POSITION_Y not found in {csv_in}') from e

    for row in data_rows:
        if not row:
            continue
        row[x_idx] = str(float(row[x_idx]) + dx)
        row[y_idx] = str(float(row[y_idx]) + dy)

    with open(csv_out, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(header_rows)
        writer.writerow(colnames)
        writer.writerows(data_rows)


records = []
for tif_path in sorted(RAW_TIF_DIR.glob('*.tif')):
    img = read_tif(tif_path)
    result = pad_image_to_512(img)
    if result is None:
        continue

    padded_img, dx, dy = result
    raw_h, raw_w = img.shape
    new_h, new_w = padded_img.shape

    out_tif_name = tif_path.stem + '_pad512.tif'
    out_tif_path = OUT_TIF_DIR / out_tif_name
    save_tif(out_tif_path, padded_img)

    csv_path = find_csv(tif_path)
    out_csv_name = ''
    if csv_path is not None:
        out_csv_name = csv_path.stem + '_pad512.csv'
        pad_csv(csv_path, OUT_CSV_DIR / out_csv_name, dx, dy)

    records.append({
        'raw_tif_name': tif_path.name,
        'padded_tif_name': out_tif_name,
        'raw_tif_size_px': f'{raw_w}x{raw_h}',
        'padded_tif_size_px': f'{new_w}x{new_h}',
        'raw_csv_name': csv_path.name if csv_path else '',
        'padded_csv_name': out_csv_name,
    })

with open(SUMMARY_CSV, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            'raw_tif_name',
            'padded_tif_name',
            'raw_tif_size_px',
            'padded_tif_size_px',
            'raw_csv_name',
            'padded_csv_name',
        ],
    )
    writer.writeheader()
    writer.writerows(records)

print(f'Processed {len(records)} image(s).')
print(f'Padded tif dir: {OUT_TIF_DIR}')
print(f'Padded csv dir: {OUT_CSV_DIR}')
print(f'Summary csv: {SUMMARY_CSV}')
