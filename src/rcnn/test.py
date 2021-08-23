from pathlib import Path
from torchvision.io import read_image

base_dir = Path('../../nih/data')

imgs = sorted(base_dir.rglob("*.png"))

for img in imgs:
    loaded = read_image(str(img))
    if loaded.shape[0] != 1:
        print(img)
        print(loaded.shape)
        break
