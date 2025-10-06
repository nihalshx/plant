# Appendix

## Appendix A: System Architecture Diagrams

### A.1 Overall System Architecture
The complete system architecture demonstrates a three-tier design pattern consisting of presentation, application, and data layers. The Flask web application serves as the primary interface, handling user interactions and orchestrating the entire disease detection workflow from image upload to result presentation.

### A.2 EfficientNet-B0 Model Architecture  
The EfficientNet-B0 architecture employs compound scaling methodology to optimize depth, width, and resolution simultaneously. The model consists of a stem convolution layer, followed by seven stages of Mobile Inverted Bottleneck (MBConv) blocks with squeeze-and-excitation attention, culminating in global average pooling and classification layers.

### A.3 Data Flow Diagram
The data processing pipeline illustrates the complete journey from raw image input through preprocessing, model inference, Grad-CAM generation, and result visualization, ensuring seamless user experience and reliable disease classification.

---

## Appendix B: Source Code Highlights

### B.1 EfficientNet-B0 Model Implementation

```python
# Core EfficientNet-B0 Model Setup for Plant Disease Detection
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def create_model(num_classes=38):
    """
    Create EfficientNet-B0 model with transfer learning
    Args:
        num_classes: Number of plant disease classes (38)
    Returns:
        Modified EfficientNet-B0 model
    """
    # Load ImageNet pre-trained weights
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    
    # Replace final classification layer
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model

# Data preprocessing transformations
def get_transforms():
    """
    Define data augmentation and preprocessing pipelines
    """
    from torchvision import transforms
    
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                             saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    valid_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, valid_transforms
```

### B.2 Advanced Training Implementation

```python
# Mixed Precision Training with Early Stopping
import torch
from torch.cuda.amp import GradScaler, autocast

def train_model(model, train_loader, val_loader, device, epochs=10):
    """
    Advanced training function with mixed precision and early stopping
    """
    # Initialize mixed precision scaler
    scaler = GradScaler()
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Accumulate metrics
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_acc += torch.sum(preds == targets).item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_acc += torch.sum(preds == targets).item()
        
        # Calculate epoch metrics
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = running_acc / len(train_loader.dataset)
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = val_acc / len(val_loader.dataset)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}')
        print(f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}')
        
        # Early stopping logic
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break
        
        scheduler.step()
    
    # Load best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model
```

### B.3 Grad-CAM Implementation

```python
# Grad-CAM Visualization for Model Interpretability
import torch
import numpy as np
import cv2
from PIL import Image

def gradcam_efficientnet(model, image_tensor, target_class=None):
    """
    Generate Grad-CAM heatmap for EfficientNet-B0
    Args:
        model: Trained EfficientNet-B0 model
        image_tensor: Input image tensor
        target_class: Target class for visualization
    Returns:
        CAM heatmap as numpy array
    """
    model.eval()
    
    # Hook variables
    conv_output = None
    gradients = None
    
    def forward_hook(module, input, output):
        nonlocal conv_output
        conv_output = output
    
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]
    
    # Register hooks on the last convolutional layer
    target_layer = model.features
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)
    
    # Forward pass
    image_tensor.requires_grad_(True)
    output = model(image_tensor)
    
    if target_class is None:
        target_class = torch.argmax(output, dim=1).item()
    
    # Backward pass
    model.zero_grad()
    class_loss = output[0, target_class]
    class_loss.backward()
    
    # Generate CAM
    if gradients is not None and conv_output is not None:
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        
        # Weighted combination of feature maps
        cam = torch.sum(weights * conv_output, dim=1, keepdim=True)
        cam = torch.relu(cam)  # ReLU to keep positive values
        
        # Normalize CAM
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    else:
        cam = np.zeros((7, 7))  # Default size for EfficientNet-B0
    
    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()
    
    return cam

def overlay_cam_on_image(original_image, cam, alpha=0.4):
    """
    Overlay CAM heatmap on original image
    Args:
        original_image: PIL Image
        cam: CAM heatmap array
        alpha: Overlay transparency
    Returns:
        PIL Image with CAM overlay
    """
    # Resize CAM to match image size
    cam_resized = cv2.resize(cam, (original_image.width, original_image.height))
    
    # Convert to heatmap
    cam_colored = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), 
        cv2.COLORMAP_JET
    )
    cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay on original image
    original_array = np.array(original_image)
    overlayed = cv2.addWeighted(original_array, 1-alpha, cam_colored, alpha, 0)
    
    return Image.fromarray(overlayed)
```

### B.4 Flask Web Application Core

```python
# Flask Application for Plant Disease Detection
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import torch
import json
from PIL import Image
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load trained model and class mappings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model(num_classes=38).to(device)
model.load_state_dict(torch.load('model/efficientnet_b0_plant_disease.pth', 
                                 map_location=device))
model.eval()

# Load class mappings
with open('model/class_to_idx.json', 'r') as f:
    class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

def allowed_file(filename):
    """Check if uploaded file is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_disease(image_path):
    """
    Predict plant disease from image
    Args:
        image_path: Path to uploaded image
    Returns:
        Dictionary with prediction results
    """
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        _, valid_transforms = get_transforms()
        image_tensor = valid_transforms(image).unsqueeze(0).to(device)
        
        # Model prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_class = idx_to_class[predicted_idx.item()]
            confidence_score = confidence.item()
        
        # Generate Grad-CAM
        cam = gradcam_efficientnet(model, image_tensor, predicted_idx.item())
        cam_overlay = overlay_cam_on_image(image, cam)
        
        # Save CAM overlay
        cam_filename = f"cam_{os.path.basename(image_path)}"
        cam_path = os.path.join(app.config['UPLOAD_FOLDER'], cam_filename)
        cam_overlay.save(cam_path)
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence_score,
            'cam_filename': cam_filename
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main page with file upload"""
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Predict disease
        results = predict_disease(filepath)
        if results:
            return render_template('result.html', 
                                 filename=filename,
                                 predicted_class=results['predicted_class'],
                                 confidence=results['confidence'],
                                 cam_filename=results['cam_filename'])
        else:
            return "Error processing image", 500
    
    return render_template('index.html')

@app.route('/result/<filename>')
def result(filename):
    """Display prediction results"""
    return render_template('result.html', filename=filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
```

### B.5 Data Collection and Preprocessing

```python
# Data preprocessing and augmentation pipeline
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import os

def create_data_loaders(data_dir, batch_size=32, num_workers=2):
    """
    Create data loaders for training and validation
    Args:
        data_dir: Root directory containing train/val/test folders
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
    Returns:
        Dictionary containing train, validation, and test loaders
    """
    # Define transforms
    train_transforms, valid_transforms = get_transforms()
    
    # Create datasets
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=train_transforms
    )
    
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'val'),
        transform=valid_transforms
    )
    
    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'test'),
        transform=valid_transforms
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'classes': train_dataset.classes,
        'class_to_idx': train_dataset.class_to_idx
    }

def evaluate_model(model, test_loader, device):
    """
    Comprehensive model evaluation
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Computing device (CPU/GPU)
    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    # Calculate metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
    
    # Classification report
    class_names = test_loader.dataset.classes
    report = classification_report(all_targets, all_preds, 
                                 target_names=class_names, 
                                 output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'targets': all_targets
    }
```

---

## Appendix C: HTML Templates and User Interface

### C.1 Base Template (base.html)

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Plant Disease Detection System</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .gradient-bg {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .card-custom {
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        .upload-area {
            border: 2px dashed #ddd;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .upload-area:hover {
            border-color: #667eea;
            background-color: #f8f9ff;
        }
    </style>
</head>
<body class="gradient-bg">
    <nav class="navbar navbar-expand-lg navbar-dark" style="background-color: rgba(0,0,0,0.1);">
        <div class="container">
            <a class="navbar-brand" href="/">
                <i class="fas fa-leaf"></i> Plant Disease Detection
            </a>
        </div>
    </nav>
    
    <div class="container mt-4">
        {% block content %}
        {% endblock %}
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    {% block scripts %}
    {% endblock %}
</body>
</html>
```

### C.2 Main Upload Page (index.html)

```html
{% extends "base.html" %}

{% block content %}
<div class="row justify-content-center">
    <div class="col-md-8">
        <div class="card card-custom">
            <div class="card-body p-5">
                <div class="text-center mb-4">
                    <i class="fas fa-microscope fa-3x text-primary mb-3"></i>
                    <h1 class="h3">Plant Disease Detection System</h1>
                    <p class="text-muted">Upload a plant leaf image to detect diseases using AI</p>
                </div>
                
                <form method="POST" enctype="multipart/form-data" id="uploadForm">
                    <div class="upload-area mb-4" id="uploadArea">
                        <i class="fas fa-cloud-upload-alt fa-3x text-muted mb-3"></i>
                        <h5>Drop your image here or click to browse</h5>
                        <p class="text-muted">Supports: JPG, PNG, BMP, WebP</p>
                        <input type="file" name="file" id="fileInput" accept="image/*" 
                               style="display: none;" required>
                    </div>
                    
                    <div id="imagePreview" class="text-center mb-4" style="display: none;">
                        <img id="preview" src="" alt="Preview" class="img-fluid rounded" 
                             style="max-height: 300px;">
                        <p class="mt-2">Selected file: <span id="fileName"></span></p>
                    </div>
                    
                    <div class="text-center">
                        <button type="submit" class="btn btn-primary btn-lg px-5" id="submitBtn">
                            <i class="fas fa-search"></i> Analyze Image
                        </button>
                    </div>
                </form>
            </div>
        </div>
        
        <div class="card card-custom mt-4">
            <div class="card-body">
                <h5 class="card-title">How it works</h5>
                <div class="row">
                    <div class="col-md-4 text-center">
                        <i class="fas fa-camera fa-2x text-primary mb-2"></i>
                        <h6>1. Upload Image</h6>
                        <small class="text-muted">Take or upload a clear photo of the plant leaf</small>
                    </div>
                    <div class="col-md-4 text-center">
                        <i class="fas fa-brain fa-2x text-primary mb-2"></i>
                        <h6>2. AI Analysis</h6>
                        <small class="text-muted">Our EfficientNet-B0 model analyzes the image</small>
                    </div>
                    <div class="col-md-4 text-center">
                        <i class="fas fa-clipboard-list fa-2x text-primary mb-2"></i>
                        <h6>3. Get Results</h6>
                        <small class="text-muted">Receive diagnosis with confidence score and visual explanation</small>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
    // File upload handling
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    const preview = document.getElementById('preview');
    const fileName = document.getElementById('fileName');
    
    uploadArea.addEventListener('click', () => fileInput.click());
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#667eea';
        uploadArea.style.backgroundColor = '#f8f9ff';
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = '#ddd';
        uploadArea.style.backgroundColor = '';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect(files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
    
    function handleFileSelect(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            preview.src = e.target.result;
            fileName.textContent = file.name;
            imagePreview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
    
    // Form submission with loading state
    document.getElementById('uploadForm').addEventListener('submit', function() {
        const submitBtn = document.getElementById('submitBtn');
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
        submitBtn.disabled = true;
    });
</script>
{% endblock %}
```

### C.3 Results Page (result.html)

```html
{% extends "base.html" %}

{% block content %}
<div class="row justify-content-center">
    <div class="col-md-10">
        <div class="card card-custom">
            <div class="card-body p-5">
                <div class="text-center mb-4">
                    <i class="fas fa-check-circle fa-3x text-success mb-3"></i>
                    <h1 class="h3">Analysis Complete</h1>
                </div>
                
                <div class="row">
                    <div class="col-md-6">
                        <h5 class="mb-3">Original Image</h5>
                        <img src="{{ url_for('static', filename='uploads/' + filename) }}" 
                             alt="Original" class="img-fluid rounded shadow">
                    </div>
                    <div class="col-md-6">
                        <h5 class="mb-3">AI Analysis (Grad-CAM)</h5>
                        <img src="{{ url_for('static', filename='uploads/' + cam_filename) }}" 
                             alt="Grad-CAM" class="img-fluid rounded shadow">
                        <small class="text-muted d-block mt-2">
                            Red areas indicate regions the AI focused on for diagnosis
                        </small>
                    </div>
                </div>
                
                <div class="row mt-4">
                    <div class="col-md-12">
                        <div class="alert alert-info">
                            <h4 class="alert-heading">
                                <i class="fas fa-microscope"></i> Diagnosis Results
                            </h4>
                            <hr>
                            <div class="row">
                                <div class="col-md-6">
                                    <h5>Detected Disease:</h5>
                                    <h3 class="text-primary">{{ predicted_class.replace('_', ' ').title() }}</h3>
                                </div>
                                <div class="col-md-6">
                                    <h5>Confidence Score:</h5>
                                    <div class="progress mb-2" style="height: 25px;">
                                        <div class="progress-bar" role="progressbar" 
                                             style="width: {{ confidence * 100 }}%">
                                            {{ "%.1f"|format(confidence * 100) }}%
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="text-center mt-4">
                    <a href="{{ url_for('index') }}" class="btn btn-primary btn-lg px-5">
                        <i class="fas fa-upload"></i> Analyze Another Image
                    </a>
                </div>
            </div>
        </div>
        
        <div class="card card-custom mt-4">
            <div class="card-body">
                <h5 class="card-title">Understanding the Results</h5>
                <div class="row">
                    <div class="col-md-6">
                        <h6><i class="fas fa-eye text-primary"></i> Visual Explanation</h6>
                        <p class="small text-muted">
                            The Grad-CAM visualization shows which parts of the leaf 
                            the AI model focused on to make its prediction. Red/warm colors 
                            indicate areas of high importance for the diagnosis.
                        </p>
                    </div>
                    <div class="col-md-6">
                        <h6><i class="fas fa-percentage text-primary"></i> Confidence Score</h6>
                        <p class="small text-muted">
                            The confidence score indicates how certain the model is about 
                            its prediction. Higher scores (>90%) indicate more reliable 
                            diagnoses, while lower scores may require expert consultation.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}
```

---

## Appendix D: Additional Technical Documentation

### D.1 Model Performance Metrics Summary

| **Metric** | **Value** |
|------------|-----------|
| Test Accuracy | 99.70% |
| Training Time | ~45 minutes (Tesla T4) |
| Model Size | 23 MB |
| Inference Time | <200ms per image |
| GPU Memory Usage | <2GB |
| Total Parameters | 5.3M |

### D.2 PlantVillage Dataset Statistics

- **Total Images**: 54,305
- **Number of Classes**: 38
- **Plant Species**: 14
- **Image Resolution**: Variable (resized to 224×224)
- **Data Split**: 80% Train, 10% Validation, 10% Test
- **Color Mode**: RGB

### D.3 Hardware and Software Requirements

**Development Environment:**
- OS: Windows 11 / Linux Ubuntu 20.04+
- Python: 3.8 or higher
- CUDA: 11.0+ (for GPU training)
- RAM: 8GB minimum, 16GB recommended
- Storage: 10GB free space

**Production Deployment:**
- CPU: 2+ cores
- RAM: 4GB minimum
- GPU: Optional (NVIDIA GTX 1060 or equivalent)
- Network: Stable internet connection
- Browser: Chrome, Firefox, Safari (latest versions)