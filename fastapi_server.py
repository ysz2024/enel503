from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import cv2
import pickle
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS to allow frontend on localhost:3000 to call API on localhost:8000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, or specify like ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Global constants
IMG_SIZE = 128  # Standard image size for all models
MAX_KEYPOINTS = 75  # Number of SIFT keypoints for model 3

# ------------------------------- LOAD MODELS -------------------------------
# Define CNN architecture - same structure for both models 1 and 2
class SimpleCNN(nn.Module):
    """
    Simple CNN architecture with two convolutional layers
    Used for both baseline (model 1) and improved (model 2) models
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)  # First conv layer: 3 input channels, 32 output
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)  # Second conv layer: 32 input, 64 output
        self.pool = nn.MaxPool2d(2)  # Max pooling with 2x2 kernel
        self.dropout = nn.Dropout(0.3)  # Dropout layer for model 2 (ignored in model 1)
        self.fc1 = nn.Linear(64*(IMG_SIZE//4)*(IMG_SIZE//4), 64)  # Fully connected layer
        self.fc2 = nn.Linear(64, 1)  # Output layer (1 for binary classification)
        
    def forward(self, x):
        """Forward pass through the network"""
        x = self.pool(nn.functional.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool(nn.functional.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        x = x.view(x.size(0), -1)  # Flatten tensor
        x = self.dropout(x)  # Apply dropout (only effective for model 2)
        x = nn.functional.relu(self.fc1(x))  # FC1 -> ReLU
        x = self.fc2(x)  # Final output
        return x

# Load Model 1 (baseline CNN)
print("Loading Model 1 (Baseline CNN)...")
model1 = SimpleCNN()
model1.load_state_dict(torch.load("results/model1_cnn.pt", map_location="cpu"))
model1.eval()  # Set model to evaluation mode

# Load Model 2 (Improved CNN with enhancements)
print("Loading Model 2 (Improved CNN)...")
model2 = SimpleCNN()
model2.load_state_dict(torch.load("results/model2_cnn.pt", map_location="cpu"))
model2.eval()  # Set model to evaluation mode

# Load Model 3 (SIFT+HOG+SVM with PCA)
print("Loading Model 3 (SIFT+HOG+SVM)...")
with open("results/model3_svm.pkl", "rb") as f:
    # Updated to load three components: classifier, scaler, and PCA
    svm_model, scaler, pca = pickle.load(f)

# Load Model 4 (Haar + AdaBoost)
print("Loading Model 4 (Haar+AdaBoost)...")
with open("results/model4_haar_adaboost.pkl", "rb") as f:
    adaboost_model = pickle.load(f)

# ------------------------------- UTILITY FUNCTIONS -------------------------------
# Transform for CNN models (resize and convert to tensor)
transform_cnn = T.Compose([
    T.ToPILImage(),  # Convert numpy array to PIL Image
    T.Resize((IMG_SIZE, IMG_SIZE)),  # Resize to standard size
    T.ToTensor()  # Convert to tensor and normalize to [0,1]
])

def predict_cnn(model, img_bytes):
    """
    Predict using CNN models (Model 1 or 2)
    
    Args:
        model: PyTorch CNN model
        img_bytes: Raw image bytes
        
    Returns:
        tuple: (label, probability of fake)
    """
    # Decode image bytes to array
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None  # Return None if image decode fails
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    x = transform_cnn(img).unsqueeze(0)  # Apply transform and add batch dimension
    
    # Forward pass without gradient computation
    with torch.no_grad():
        logits = model(x)
        prob_real = torch.sigmoid(logits).item()  # Convert to probability
    
    prob_fake = 1.0 - prob_real  # Probability of fake = 1 - probability of real
    
    # Decide label: if prob_fake >=0.5 => "Fake", else "Real"
    # For model 2, we use a slightly lower threshold (0.45)
    threshold = 0.45 if model is model2 else 0.5
    label = "Fake" if prob_fake >= threshold else "Real"
    
    return label, float(prob_fake)

def extract_sift(img, max_kp=MAX_KEYPOINTS):
    """
    Extract SIFT features from an image
    
    Args:
        img: Input image
        max_kp: Maximum number of keypoints to extract
        
    Returns:
        numpy.ndarray: Flattened SIFT features
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply contrast enhancement to highlight features
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Extract SIFT features with optimized parameters
    sift = cv2.SIFT_create(nfeatures=max_kp, contrastThreshold=0.02)
    kp, desc = sift.detectAndCompute(gray, None)
    
    # Handle empty descriptors
    if desc is None:
        desc = np.zeros((1, 128), dtype=np.float32)
    
    # Ensure we have exactly max_kp descriptors
    if len(desc) > max_kp:
        desc = desc[:max_kp]
    if len(desc) < max_kp:
        pad = np.zeros((max_kp - len(desc), 128), dtype=np.float32)
        desc = np.vstack((desc, pad))
    
    return desc.flatten()

def extract_hog(img):
    """
    Extract HOG features for additional texture information
    
    Args:
        img: Input image
        
    Returns:
        numpy.ndarray: Flattened HOG features
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # HOG parameters
    win_size = (128, 128)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    
    # Create HOG descriptor
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    
    # Compute HOG features
    hog_features = hog.compute(gray)
    
    return hog_features.flatten()

def predict_svm(img_bytes):
    """
    Predict using the SIFT+HOG+SVM model (Model 3)
    
    Args:
        img_bytes: Raw image bytes
        
    Returns:
        tuple: (label, probability of fake)
    """
    # Decode image bytes to array
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None  # Return None if image decode fails
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to standard size
    
    # Extract combined features (SIFT + HOG)
    sift_features = extract_sift(img)
    hog_features = extract_hog(img)
    combined_features = np.concatenate([sift_features, hog_features])
    
    # Standardize features
    feat_sc = scaler.transform(combined_features.reshape(1, -1))
    
    # Apply PCA dimensionality reduction
    feat_pca = pca.transform(feat_sc)
    
    # Make prediction
    # For SciKit-Learn models: 0 => Fake, 1 => Real
    if hasattr(svm_model, 'predict_proba'):
        proba = svm_model.predict_proba(feat_pca)[0]  # [pFake, pReal]
        prob_fake = float(proba[0])
        pred = 0 if prob_fake >= 0.5 else 1
    else:
        # If model doesn't support probability, use decision function
        pred = svm_model.predict(feat_pca)[0]
        decision = svm_model.decision_function(feat_pca)[0]
        # Convert decision score to probability (approximate)
        prob_fake = 1.0 / (1.0 + np.exp(-decision)) if pred == 0 else 1.0 - (1.0 / (1.0 + np.exp(-decision)))
    
    label = "Fake" if pred == 0 else "Real"
    return label, prob_fake

def extract_haar_features(img):
    """
    Extract Haar-like features from an image
    
    Args:
        img: Grayscale input image
        
    Returns:
        numpy.ndarray: Extracted Haar features
    """
    features = []
    
    # Define different region sizes for feature extraction
    sizes = [(2, 2), (4, 4), (8, 8)]
    
    for size in sizes:
        height, width = size
        step_y = img.shape[0] // height
        step_x = img.shape[1] // width
        
        # Extract vertical edge features (difference between adjacent rectangles)
        for y in range(0, img.shape[0] - step_y, step_y):
            for x in range(0, img.shape[1] - 2*step_x, step_x):
                rect1 = img[y:y+step_y, x:x+step_x].mean()
                rect2 = img[y:y+step_y, x+step_x:x+2*step_x].mean()
                features.append(rect1 - rect2)
        
        # Extract horizontal edge features
        for y in range(0, img.shape[0] - 2*step_y, step_y):
            for x in range(0, img.shape[1] - step_x, step_x):
                rect1 = img[y:y+step_y, x:x+step_x].mean()
                rect2 = img[y+step_y:y+2*step_y, x:x+step_x].mean()
                features.append(rect1 - rect2)
        
        # Extract checkerboard pattern features
        for y in range(0, img.shape[0] - 2*step_y, step_y):
            for x in range(0, img.shape[1] - 2*step_x, step_x):
                rect1 = img[y:y+step_y, x:x+step_x].mean()
                rect2 = img[y:y+step_y, x+step_x:x+2*step_x].mean()
                rect3 = img[y+step_y:y+2*step_y, x:x+step_x].mean()
                rect4 = img[y+step_y:y+2*step_y, x+step_x:x+2*step_x].mean()
                features.append(rect1 + rect4 - rect2 - rect3)
    
    # Add LBP-like features (comparing center pixel with surrounding regions)
    for y in range(step_y, img.shape[0] - step_y, step_y):
        for x in range(step_x, img.shape[1] - step_x, step_x):
            center = img[y:y+step_y, x:x+step_x].mean()
            neighbors = [
                img[y-step_y:y, x-step_x:x].mean(),
                img[y-step_y:y, x:x+step_x].mean(),
                img[y-step_y:y, x+step_x:x+2*step_x].mean(),
                img[y:y+step_y, x+step_x:x+2*step_x].mean(),
                img[y+step_y:y+2*step_y, x+step_x:x+2*step_x].mean(),
                img[y+step_y:y+2*step_y, x:x+step_x].mean(),
                img[y+step_y:y+2*step_y, x-step_x:x].mean(),
                img[y:y+step_y, x-step_x:x].mean()
            ]
            for n in neighbors:
                features.append(center - n)
    
    return np.array(features)

def predict_haar_adaboost(img_bytes):
    """
    Predict using the Haar+AdaBoost model (Model 4)
    
    Args:
        img_bytes: Raw image bytes
        
    Returns:
        tuple: (label, probability of fake)
    """
    # Decode image bytes to array
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None  # Return None if image decode fails
    
    # Convert to grayscale and resize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Extract Haar features
    features = extract_haar_features(img).reshape(1, -1)
    
    # Predict using AdaBoost
    pred = adaboost_model.predict(features)[0]
    label = "Fake" if pred == 0 else "Real"
    
    # Get probability
    proba = adaboost_model.predict_proba(features)[0]  # [pFake, pReal]
    prob_fake = float(proba[0])
    
    return label, prob_fake

# ------------------------------- FASTAPI ROUTES -------------------------------
@app.post("/predict/{model_id}")
async def predict(model_id: int, file: UploadFile = File(...)):
    """
    Endpoint to predict fake/real face using the specified model
    
    Args:
        model_id: Model to use (1-4)
        file: Uploaded image file
    
    Returns:
        JSONResponse with prediction and probability
    """
    # Read uploaded file bytes
    img_bytes = await file.read()
    
    # Choose model based on model_id
    if model_id == 1:
        label, prob = predict_cnn(model1, img_bytes)
    elif model_id == 2:
        label, prob = predict_cnn(model2, img_bytes)
    elif model_id == 3:
        label, prob = predict_svm(img_bytes)
    elif model_id == 4:
        label, prob = predict_haar_adaboost(img_bytes)
    else:
        return JSONResponse({"error": "Invalid model_id"}, status_code=400)

    # Check if image processing was successful
    if label is None:
        return JSONResponse({"error": "Invalid image or decode failure"}, status_code=400)
    
    # Return prediction results
    return JSONResponse({
        "prediction": label,
        "probability": prob
    })

# Start server when running this script directly
if __name__=="__main__":
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)