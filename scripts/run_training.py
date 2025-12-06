import argparse
from src.models.train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, nargs=2, default=(224,224))
    parser.add_argument('--save', default='models/saved_model.h5')
    args = parser.parse_args()
    train(args.data_dir, img_size=tuple(args.img_size), batch_size=args.batch_size, epochs=args.epochs, save_path=args.save)
