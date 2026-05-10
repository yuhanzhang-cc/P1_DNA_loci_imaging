from PIL import Image

path = r"D:\Projects\P1_DNA_loci_imaging\outputs\260510test_pt\pixel_size_explore\sample_4.tif"

with Image.open(path) as im:
    desc = im.info.get("ImageDescription", "")
    print("=== ImageDescription 内容 ===")
    print(repr(desc))
    print("==========================")