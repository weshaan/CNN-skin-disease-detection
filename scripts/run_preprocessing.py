from src.data.preprocess import split_and_prepare
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--dst', required=True)
    parser.add_argument('--img-size', type=int, nargs=2, default=(224,224))
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--test-split', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    split_and_prepare(args.src, args.dst, tuple(args.img_size), args.val_split, args.test_split, args.seed)
