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


def open_tif_as_array(tif_path: Path):
    with Image.open(tif_path) as im:
        return np.array(im)


def save_array_as_tif(arr: np.ndarray, out_path: Path):
    Image.fromarray(arr).save(out_path)


def pad_image_to_512(img: np.ndarray):
    if img.ndim != 2:
        raise ValueError(f'Only 2D grayscale tif is supported: {img.shape}')
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


def norm_cell(x: str) -> str:
    return x.replace('\ufeff', '').strip().upper()


def detect_header_and_data_start(rows):
    header_idx = None
    for i, row in enumerate(rows[:10]):
        norm = [norm_cell(c) for c in row]
        if 'POSITION_X' in norm and 'POSITION_Y' in norm:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError('Cannot find header row containing POSITION_X and POSITION_Y')

    # Standard TrackMate CSV: header row + 3 descriptor rows + data rows
    default_data_start = header_idx + 4
    if len(rows) > default_data_start:
        return header_idx, default_data_start
    return header_idx, header_idx + 1


def pad_csv(csv_in: Path, csv_out: Path, dx: int, dy: int):
    with open(csv_in, 'r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f))

    header_idx, data_start = detect_header_and_data_start(rows)
    colnames = [norm_cell(c) for c in rows[header_idx]]
    x_idx = colnames.index('POSITION_X')
    y_idx = colnames.index('POSITION_Y')

    out_rows = [row[:] for row in rows[:data_start]]
    for row in rows[data_start:]:
        if not row or all(c.strip() == '' for c in row):
            out_rows.append(row)
            continue
        row = row[:]
        try:
            row[x_idx] = str(float(row[x_idx]) + dx)
            row[y_idx] = str(float(row[y_idx]) + dy)
        except (ValueError, IndexError):
            pass
        out_rows.append(row)

    with open(csv_out, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)


records = []
for tif_path in sorted(list(RAW_TIF_DIR.glob('*.tif')) + list(RAW_TIF_DIR.glob('*.tiff'))):
    img = open_tif_as_array(tif_path)
    result = pad_image_to_512(img)
    if result is None:
        continue

    padded_img, dx, dy = result
    raw_h, raw_w = img.shape
    new_h, new_w = padded_img.shape

    out_tif_name = tif_path.stem + '_pad512.tif'
    out_tif_path = OUT_TIF_DIR / out_tif_name
    save_array_as_tif(padded_img, out_tif_path)

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
