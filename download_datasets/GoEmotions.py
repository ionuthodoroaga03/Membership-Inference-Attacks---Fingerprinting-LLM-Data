import shutil
from pathlib import Path

import kagglehub

# Project root = parent of `download_datasets/` (this file's directory)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

dest_dir = PROJECT_ROOT / "data" / "GoEmotions"
dest_dir.mkdir(parents=True, exist_ok=True)

src = Path(kagglehub.dataset_download("debarshichanda/goemotions"))
print("Downloaded to cache:", src)

if src.is_dir():
    shutil.copytree(src, dest_dir, dirs_exist_ok=True)
else:
    shutil.copy2(src, dest_dir / src.name)

print("Copied to:", dest_dir.resolve())
