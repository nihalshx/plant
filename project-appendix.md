# Appendix

## Appendix A: System Architecture Diagrams

### A.1 Complete System Architecture Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    PLANT DISEASE DETECTION SYSTEM               │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│  • PlantVillage Dataset (38 Classes)                           │
│  • Train/Validation/Test Split (80/10/10)                      │
│  • Image Preprocessing & Augmentation                          │
│  • Class Mapping (JSON Format)                                 │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                       MODEL LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│  • EfficientNet-B0 Architecture                                │
│  • ImageNet Pre-trained Weights                                │
│  • Transfer Learning (38 Classes)                              │
│  • Mixed Precision Training                                    │
│  • Model Persistence (.pth format)                             │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│  • Flask Web Framework                                          │
│  • File Upload Handler                                          │
│  • Image Processing Pipeline                                    │
│  • Model Inference Engine                                       │
│  • Grad-CAM Visualization                                       │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PRESENTATION LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│  • HTML5/CSS3/Bootstrap UI                                      │
│  • Responsive Design                                            │
│  • Real-time Results Display                                    │
│  • Interactive Visualization                                    │
└─────────────────────────────────────────────────────────────────┘
```

### A.2 EfficientNet-B0 Architecture Details

```
Input Image (224×224×3)
         │
         ▼
┌─────────────────┐
│ Stem Conv 3×3   │ ────► 112×112×32
│ BN + Swish      │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ MBConv1 3×3     │ ────► 112×112×16 (1 layer)
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ MBConv6 3×3     │ ────► 112×112×24 (2 layers)
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ MBConv6 5×5     │ ────► 56×56×40 (2 layers)
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ MBConv6 3×3     │ ────► 28×28×80 (3 layers)
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ MBConv6 5×5     │ ────► 14×14×112 (3 layers)
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ MBConv6 5×5     │ ────► 14×14×192 (4 layers)
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ MBConv6 3×3     │ ────► 7×7×320 (1 layer)
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Conv1×1         │ ────► 7×7×1280
│ BN + Swish      │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Global Avg Pool │ ────► 1280
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Dropout(0.2)    │ ────► 1280
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Dense Layer     │ ────► 38 Classes
└─────────────────┘
```

## Appendix B: Training Configuration

### B.1 Hyperparameter Settings

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Optimizer** | AdamW | Better generalization with weight decay |
| **Learning Rate** | 1×10⁻³ | Balanced convergence speed |
| **Weight Decay** | 1×10⁻⁴ | L2 regularization |
| **Batch Size** | 32 | Optimal for Tesla T4 memory |
| **Epochs** | 10 (max) | Early stopping prevents overfitting |
| **Scheduler** | StepLR | Reduces LR by 0.1 every 3 epochs |
| **Early Stopping** | Patience=3 | Monitors validation loss |
| **Loss Function** | CrossEntropyLoss | Multi-class classification |
| **Dropout** | 0.2 | Regularization in classifier head |

### B.2 Hardware Specifications

```
GPU Configuration:
├── Model: Tesla T4
├── Memory: 15.7 GB GDDR6
├── CUDA Cores: 2560
├── Tensor Cores: 320 (2nd Gen)
├── Memory Bandwidth: 300 GB/s
├── FP16 Performance: 65 TFLOPS
└── Architecture: Turing

Training Environment:
├── Platform: Google Colab Pro
├── Python Version: 3.8+
├── PyTorch Version: 2.0+
├── CUDA Version: 11.8
└── cuDNN Version: 8.x
```

### B.3 Data Augmentation Pipeline

```python
# Training Transforms
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.02
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Validation/Test Transforms
valid_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

## Appendix C: Source Code Structure

### C.1 Project Directory Structure

```
plant-disease-detection/
├── app.py                      # Flask web application
├── train_model.py              # Training script
├── collect_data.py             # Data collection utility
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── model/                      # Trained model files
│   ├── efficientnet_b0_plant_disease.pth
│   ├── class_to_idx.json
│   └── training_history.json
├── static/                     # Static web assets
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── main.js
│   └── uploads/                # Uploaded images
├── templates/                  # HTML templates
│   ├── base.html
│   ├── index.html
│   └── result.html
└── data/                       # Dataset directory
    ├── train/
    ├── val/
    └── test/
```

### C.2 Model Implementation

```python
# EfficientNet-B0 Model Setup
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes=38):
        super(PlantDiseaseModel, self).__init__()
        
        # Load pre-trained EfficientNet-B0
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = efficientnet_b0(weights=weights)
        
        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

# Model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PlantDiseaseModel(num_classes=38).to(device)
```

### C.3 Training Loop Implementation

```python
def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        # Mixed precision backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Calculate accuracy
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * images.size(0)
        running_acc += torch.sum(preds == targets.data)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_acc.double() / len(train_loader.dataset)
    
    return epoch_loss, epoch_acc
```

### C.4 Grad-CAM Implementation

```python
def generate_gradcam(model, image_tensor, target_class, device):
    model.eval()
    
    # Hook functions for gradient extraction
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    # Register hooks on the last convolutional layer
    target_layer = model.backbone.features
    h_forward = target_layer.register_forward_hook(forward_hook)
    h_backward = target_layer.register_full_backward_hook(backward_hook)
    
    # Forward pass
    image_tensor = image_tensor.unsqueeze(0).to(device)
    image_tensor.requires_grad_(True)
    
    output = model(image_tensor)
    
    # Backward pass
    model.zero_grad()
    class_score = output[:, target_class]
    class_score.backward()
    
    # Generate CAM
    gradient = gradients[0].cpu().data.numpy()[0]
    activation = activations[0].cpu().data.numpy()[0]
    
    weights = np.mean(gradient, axis=(1, 2))
    cam = np.zeros(activation.shape[1:], dtype=np.float32)
    
    for i, w in enumerate(weights):
        cam += w * activation[i]
    
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam / np.max(cam) if np.max(cam) != 0 else cam
    
    # Remove hooks
    h_forward.remove()
    h_backward.remove()
    
    return cam
```

### C.5 Flask Application Routes

```python
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('predict', filename=filename))
        else:
            flash('Invalid file type')
    
    return render_template('index.html')

@app.route('/predict/<filename>')
def predict(filename):
    # Load and preprocess image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Model inference
    prediction, confidence = predict_disease(image_path)
    
    # Generate Grad-CAM
    cam_image = generate_gradcam_visualization(image_path, prediction)
    
    return render_template('result.html', 
                         filename=filename,
                         prediction=prediction,
                         confidence=confidence,
                         cam_image=cam_image)
```

## Appendix D: Additional Results

### D.1 Complete Classification Report

```
                                          precision    recall  f1-score   support

                        Apple___Apple_scab     1.00      0.98      0.99        51
                         Apple___Black_rot     1.00      1.00      1.00        51
                  Apple___Cedar_apple_rust     1.00      1.00      1.00        22
                           Apple___healthy     1.00      1.00      1.00       133
                       Blueberry___healthy     1.00      1.00      1.00       121
  Cherry_(including_sour)___Powdery_mildew     1.00      1.00      1.00        85
         Cherry_(including_sour)___healthy     1.00      1.00      1.00        70
Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot  0.90  0.90      0.90        41
               Corn_(maize)___Common_rust_     1.00      1.00      1.00        96
       Corn_(maize)___Northern_Leaf_Blight     0.95      0.95      0.95        80
                    Corn_(maize)___healthy     1.00      1.00      1.00        94
                         Grape___Black_rot     1.00      1.00      1.00        95
              Grape___Esca_(Black_Measles)     1.00      1.00      1.00       112
Grape___Leaf_blight_(Isariopsis_Leaf_Spot)     1.00      1.00      1.00        87
                           Grape___healthy     1.00      1.00      1.00        35
  Orange___Haunglongbing_(Citrus_greening)     1.00      1.00      1.00       441
                    Peach___Bacterial_spot     1.00      1.00      1.00       185
                           Peach___healthy     1.00      1.00      1.00        30
             Pepper,_bell___Bacterial_spot     0.99      0.99      0.99        81
                    Pepper,_bell___healthy     0.99      0.99      0.99       119
                     Potato___Early_blight     1.00      1.00      1.00        80
                      Potato___Late_blight     1.00      1.00      1.00        80
                          Potato___healthy     1.00      1.00      1.00        13
                       Raspberry___healthy     1.00      1.00      1.00        31
                         Soybean___healthy     1.00      1.00      1.00       408
                   Squash___Powdery_mildew     1.00      1.00      1.00       148
                  Strawberry___Leaf_scorch     1.00      1.00      1.00        90
                      Strawberry___healthy     1.00      1.00      1.00        37
                   Tomato___Bacterial_spot     1.00      1.00      1.00       171
                     Tomato___Early_blight     1.00      1.00      1.00        80
                      Tomato___Late_blight     1.00      0.99      1.00       154
                        Tomato___Leaf_Mold     1.00      1.00      1.00        77
               Tomato___Septoria_leaf_spot     0.99      1.00      0.99       143
Tomato___Spider_mites_Two-spotted_spider_mite  1.00  1.00      1.00       135
                      Tomato___Target_Spot     1.00      0.99      1.00       113
    Tomato___Tomato_Yellow_Leaf_Curl_Virus     1.00      1.00      1.00       430
              Tomato___Tomato_mosaic_virus     1.00      1.00      1.00        31
                          Tomato___healthy     1.00      1.00      1.00       128

                              accuracy                           0.997      4378
                             macro avg     0.995     0.995     0.995      4378
                          weighted avg     0.997     0.997     0.997      4378
```

### D.2 Training Metrics Summary

| Metric | Value |
|--------|-------|
| **Final Training Accuracy** | 99.84% |
| **Final Validation Accuracy** | 99.79% |
| **Test Accuracy** | 99.70% |
| **Training Loss (Final)** | 0.0061 |
| **Validation Loss (Final)** | 0.0060 |
| **Total Parameters** | 5.3M |
| **Training Time** | ~2 hours (Tesla T4) |
| **Inference Time** | ~150ms per image |

### D.3 Performance Benchmarks

```
Model Comparison:
┌─────────────────┬──────────┬────────────┬──────────┬─────────┐
│ Architecture    │ Accuracy │ Parameters │ FLOPs    │ Size    │
├─────────────────┼──────────┼────────────┼──────────┼─────────┤
│ EfficientNet-B0 │ 99.70%   │ 5.3M       │ 0.39B    │ 20.5MB  │
│ ResNet-50       │ 96.20%   │ 25.6M      │ 4.1B     │ 98MB    │
│ VGG-16          │ 93.80%   │ 138M       │ 15.3B    │ 528MB   │
│ MobileNet-v2    │ 95.10%   │ 3.4M       │ 0.3B     │ 14MB    │
└─────────────────┴──────────┴────────────┴──────────┴─────────┘
```

### D.4 Deployment Requirements

```
System Requirements:
├── Minimum RAM: 4GB
├── Recommended RAM: 8GB
├── Python Version: 3.8+
├── GPU Memory: 2GB (optional, for faster inference)
└── Disk Space: 1GB

Python Dependencies:
├── torch>=2.0.0
├── torchvision>=0.15.0
├── flask>=2.3.0
├── pillow>=9.0.0
├── opencv-python>=4.5.0
├── numpy>=1.21.0
├── matplotlib>=3.5.0
└── werkzeug>=2.3.0

Installation Commands:
$ pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
$ pip install flask pillow opencv-python numpy matplotlib werkzeug
$ git clone https://github.com/username/plant-disease-detection.git
$ cd plant-disease-detection
$ python app.py
```

### D.5 API Usage Examples

```python
# Example 1: Single Image Prediction
import requests

url = "http://localhost:5000/api/predict"
files = {'file': open('plant_leaf.jpg', 'rb')}
response = requests.post(url, files=files)
result = response.json()

print(f"Disease: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")

# Example 2: Batch Processing
import os
from pathlib import Path

image_folder = Path("test_images/")
results = []

for image_path in image_folder.glob("*.jpg"):
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
        result = response.json()
        results.append({
            'filename': image_path.name,
            'prediction': result['prediction'],
            'confidence': result['confidence']
        })

# Save results to CSV
import pandas as pd
df = pd.DataFrame(results)
df.to_csv('batch_predictions.csv', index=False)
```