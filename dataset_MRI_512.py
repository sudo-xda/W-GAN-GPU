import os
from PIL import Image

folder = r"C:\Users\SOEE\Documents\GitHub\W-GAN-GPU\dataset\MRI_kaggle_512"

saved = 0
fixed = 0
deleted = 0

for root, _, files in os.walk(folder):
    for f in files:
        if not f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            continue

        path = os.path.join(root, f)

        try:
            img = Image.open(path)
            img = img.convert("L")  # grayscale safe

            if img.size != (512, 512):
                img = img.resize((512, 512), Image.BICUBIC)
                img.save(path)
                fixed += 1
            else:
                saved += 1

        except Exception:
            os.remove(path)
            deleted += 1

print("Dataset cleaned successfully!")
print("✔ Already correct size:", saved)
print("🔧 Resized:", fixed)
print("✘ Deleted corrupted:", deleted)

