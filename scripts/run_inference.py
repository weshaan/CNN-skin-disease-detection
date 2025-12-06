import argparse
from src.inference.predict import predict_image
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--img', required=True)
    args = parser.parse_args()
    res = predict_image(args.model, args.img)
    print(res)
