from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

def predict_image(model_path, img_path, img_size=(224,224)):
    model = load_model(model_path)
    img = image.load_img(img_path, target_size=img_size)
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    prob = model.predict(arr)[0][0]
    label = 'malignant' if prob >= 0.5 else 'benign'
    return {'probability': float(prob), 'label': label}
