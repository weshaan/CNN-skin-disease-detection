from flask import Flask, request, render_template, jsonify
import os
from src.inference.predict import predict_image

app = Flask(__name__, static_folder='static', template_folder='templates')
MODEL_PATH = os.path.join('models','saved_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file'}), 400
    f = request.files['file']
    save_path = os.path.join('static', f.filename)
    f.save(save_path)
    res = predict_image(MODEL_PATH, save_path)
    return jsonify(res)

if __name__ == '__main__':
    app.run(debug=True)
