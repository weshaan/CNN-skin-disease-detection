import os
import random
import shutil
from pathlib import Path
from PIL import Image

def split_and_prepare(src_dir, dst_dir, img_size=(224,224), val_split=0.2, test_split=0.1, seed=42):
    '''
    Expects src_dir to contain class-subfolders, e.g. src_dir/benign/*.jpg, src_dir/malignant/*.jpg
    Will create dst_dir/train, dst_dir/val, dst_dir/test with the same class subfolders.
    '''
    random.seed(seed)
    src = Path(src_dir)
    dst = Path(dst_dir)
    if not src.exists():
        raise FileNotFoundError(f"Source dir {src_dir} not found.")
    # remove and recreate
    if dst.exists():
        shutil.rmtree(dst)
    for split in ('train','val','test'):
        (dst / split).mkdir(parents=True, exist_ok=True)

    classes = [d.name for d in src.iterdir() if d.is_dir()]
    for cls in classes:
        files = list((src/cls).glob('*'))
        random.shuffle(files)
        n = len(files)
        n_test = int(n * test_split)
        n_val = int(n * val_split)
        n_train = n - n_val - n_test
        groups = {
            'train': files[:n_train],
            'val': files[n_train:n_train+n_val],
            'test': files[n_train+n_val:]
        }
        for split, flist in groups.items():
            target_dir = dst / split / cls
            target_dir.mkdir(parents=True, exist_ok=True)
            for f in flist:
                # optionally resize to keep disk small; use PIL
                try:
                    im = Image.open(f)
                    im = im.convert('RGB')
                    im = im.resize(img_size)
                    im.save(target_dir / f.name, quality=90)
                except Exception as e:
                    # skip problematic files
                    print(f"Skipping {f}: {e}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--dst', required=True)
    parser.add_argument('--img-size', type=int, nargs=2, default=(224,224))
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--test-split', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    split_and_prepare(args.src, args.dst, tuple(args.img_size), args.val_split, args.test_split, args.seed)
