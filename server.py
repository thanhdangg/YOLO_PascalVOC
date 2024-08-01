from flask import Flask, request, jsonify
import io
from PIL import Image
import torch

app = Flask(__name__)

# Tải model YOLOv5s
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    # Lấy ảnh từ request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    img = Image.open(io.BytesIO(file.read()))

    results = model(img)

    results_json = results.pandas().xyxy[0].to_json(orient='records')
    print('results_json:', results_json)

    return jsonify({'results': results_json})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
