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

# CORS so frontend on localhost:3000 can call localhost:8000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IMG_SIZE = 128

# ------------------------------- LOAD MODELS -------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64*(IMG_SIZE//4)*(IMG_SIZE//4), 64)
        self.fc2 = nn.Linear(64, 1)
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Model 1 (baseline)
model1 = SimpleCNN()
model1.load_state_dict(torch.load("results/model1_cnn.pt", map_location="cpu"))
model1.eval()

# Model 2 (HistEq + Aug)
model2 = SimpleCNN()
model2.load_state_dict(torch.load("results/model2_cnn.pt", map_location="cpu"))
model2.eval()

# Model 3 (SIFT + SVM)
with open("results/model3_svm.pkl", "rb") as f:
    svm_model, scaler = pickle.load(f)

# Model 4 (Haar + AdaBoost)
with open("results/model4_haar_adaboost.pkl", "rb") as f:
    adaboost_model = pickle.load(f)

# ------------------------------- UTILS -------------------------------
transform_cnn = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor()
])

def predict_cnn(model, img_bytes):
    # Preprocess
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = transform_cnn(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        prob_real = torch.sigmoid(logits).item()
    prob_fake = 1.0 - prob_real
    # Decide label: if prob_fake >=0.5 => "Fake", else "Real"
    label = "Fake" if prob_fake >= 0.5 else "Real"
    return label, float(prob_fake)

def predict_svm(img_bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    kp, desc = sift.detectAndCompute(gray, None)
    if desc is None:
        desc = np.zeros((1,128), dtype=np.float32)
    if len(desc) > 100:
        desc = desc[:100]
    if len(desc) < 100:
        pad = np.zeros((100-len(desc), 128), dtype=np.float32)
        desc = np.vstack((desc, pad))
    feat = desc.flatten().reshape(1,-1)
    feat_sc = scaler.transform(feat)
    # scikit-learn returns 0 => Fake, 1 => Real
    pred = svm_model.predict(feat_sc)[0]
    label = "Fake" if pred == 0 else "Real"
    # If using probability:
    proba = svm_model.predict_proba(feat_sc)[0]  # [pFake, pReal]
    prob_fake = float(proba[0])
    return label, prob_fake

def extract_haar_features(img):
    """提取Haar特征"""
    features = []
    
    # 定义不同尺寸的矩形区域进行特征提取
    sizes = [(2, 2), (4, 4), (8, 8)]
    
    for size in sizes:
        height, width = size
        step_y = img.shape[0] // height
        step_x = img.shape[1] // width
        
        # 提取垂直边缘特征 (2邻接矩形差值)
        for y in range(0, img.shape[0] - step_y, step_y):
            for x in range(0, img.shape[1] - 2*step_x, step_x):
                rect1 = img[y:y+step_y, x:x+step_x].mean()
                rect2 = img[y:y+step_y, x+step_x:x+2*step_x].mean()
                features.append(rect1 - rect2)
        
        # 提取水平边缘特征
        for y in range(0, img.shape[0] - 2*step_y, step_y):
            for x in range(0, img.shape[1] - step_x, step_x):
                rect1 = img[y:y+step_y, x:x+step_x].mean()
                rect2 = img[y+step_y:y+2*step_y, x:x+step_x].mean()
                features.append(rect1 - rect2)
        
        # 提取棋盘模式特征
        for y in range(0, img.shape[0] - 2*step_y, step_y):
            for x in range(0, img.shape[1] - 2*step_x, step_x):
                rect1 = img[y:y+step_y, x:x+step_x].mean()
                rect2 = img[y:y+step_y, x+step_x:x+2*step_x].mean()
                rect3 = img[y+step_y:y+2*step_y, x:x+step_x].mean()
                rect4 = img[y+step_y:y+2*step_y, x+step_x:x+2*step_x].mean()
                features.append(rect1 + rect4 - rect2 - rect3)
    
    # 附加LBP类型特征 (对比中心像素与周围区域)
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
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    
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
    img_bytes = await file.read()
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

    if label is None:
        return JSONResponse({"error": "Invalid image or decode failure"}, status_code=400)
    return JSONResponse({
        "prediction": label,
        "probability": prob
    })

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)