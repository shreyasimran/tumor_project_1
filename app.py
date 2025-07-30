import os
from flask import Flask, request, render_template
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from model import CNN_TUMOR  # We'll define the model architecture in model.py

# Define class labels
CLA_label = {
    0: 'Brain Tumor',
    1: 'Healthy'
}

# Image transform (must match training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load model
params_model = {
    "shape_in": (3, 256, 256),
    "initial_filters": 8,
    "num_fc1": 100,
    "dropout_rate": 0.25,
    "num_classes": 2
}
model = CNN_TUMOR(params_model)
model.load_state_dict(torch.load("model_weights/weights.pt", map_location='cpu'))
model.eval()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Home route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Predict
            img = Image.open(filepath).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                output = model(img_tensor)
                probs = F.softmax(output, dim=1)
                confidence, pred = torch.max(probs, 1)
                label = CLA_label[pred.item()]
                confidence = confidence.item()

            return render_template("index.html", prediction=label,
                                   confidence=f"{confidence * 100:.2f}",
                                   image_path=filepath)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
    app.run(host='0.0.0.0',port=10000)
