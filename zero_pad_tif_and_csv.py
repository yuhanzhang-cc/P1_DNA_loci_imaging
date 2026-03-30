from pathlib import Path
import pandas as pd
import numpy as np
import tifffile as tiff

TARGET = 512
RAW_TIF_DIR = Path('image_annotation/data/raw_tif')
RAW_CSV_DIR = Path('image_annotation/data/raw_csv_pixel_replaced')
OUT_TIF_DIR = Path('image_annotation/data/padded_tif_512')
OUT_CSV_DIR = Path('image_annotation/data/padded_csv_512')
SUMMARY_CSV = Path('image_annotation/data/pad_summary.csv')

OUT_TIF_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)


def find_csv(tif_path: Path) -> Path | None:
    candidates = [
        RAW_CSV_DIR / f'{tif_path.stem}.csv',
        RAW_CSV_DIR / f'{tif_path.stem}_allspots.csv',
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def pad_image_to_512(img: np.ndarray):
    h, w = img.shape[-2], img.shape[-1]
    if h > TARGET or w > TARGET:
        return None
    pad_top = (TARGET - h) // 2
    pad_bottom = TARGET - h - pad_top
    pad_left = (TARGET - w) // 2
    pad_right = TARGET - w - pad_left
    pad_width = [(0, 0)] * img.ndim
    pad_width[-2] = (pad_top, pad_bottom)
    pad_width[-1] = (pad_left, pad_right)
    out = np.pad(img, pad_width, mode='constant', constant_values=0)
    return out, pad_left, pad_top


def pad_csv(csv_in: Path, csv_out: Path, dx: int, dy: int):
    with open(csv_in, 'r', encoding='utf-8-sig', newline='') as f:
        header_lines = [next(f) for _ in range(4)]
    df = pd.read_csv(csv_in, skiprows=4)
    if not df.empty:
        df['POSITION_X'] = df['POSITION_X'] + dx
        df['POSITION_Y'] = df['POSITION_Y'] + dy
    with open(csv_out, 'w', encoding='utf-8-sig', newline='') as f:
        f.writelines(header_lines)
        df.to_csv(f, index=False)


records = []
for tif_path in sorted(RAW_TIF_DIR.glob('*.tif')):
    img = tiff.imread(tif_path)
    result = pad_image_to_512(img)
    if result is None:
        continue
    padded_img, dx, dy = result
    raw_h, raw_w = img.shape[-2], img.shape[-1]
    new_h, new_w = padded_img.shape[-2], padded_img.shape[-1]

    out_tif_name = tif_path.stem + '_pad512.tif'
    out_tif_path = OUT_TIF_DIR / out_tif_name
    tiff.imwrite(out_tif_path, padded_img)

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

pd.DataFrame(records, columns=[
    'raw_tif_name',
    'padded_tif_name',
    'raw_tif_size_px',
    'padded_tif_size_px',
    'raw_csv_name',
    'padded_csv_name',
]).to_csv(SUMMARY_CSV, index=False, encoding='utf-8-sig')

print(f'Processed {len(records)} image(s).')
print(f'Padded tif dir: {OUT_TIF_DIR}')
print(f'Padded csv dir: {OUT_CSV_DIR}')
print(f'Summary csv: {SUMMARY_CSV}')
