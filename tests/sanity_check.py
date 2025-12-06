# Quick sanity check: build model and run a forward pass on random data
import numpy as np
from src.models.cnn_model import build_simple_cnn

if __name__ == '__main__':
    model = build_simple_cnn((64,64,3))
    x = np.random.rand(1,64,64,3).astype('float32')
    y = model.predict(x)
    print('Sanity check prediction shape:', y.shape)
