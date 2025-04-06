import os, glob, cv2, time
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

DATA_DIR = "dataset"
IMG_SIZE = (128, 128)
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data(folder):
    X, y = [], []
    for cls, label in [("Fake", 0), ("Real", 1)]:
        fdir = os.path.join(folder, cls)
        for f in glob.glob(fdir + "/*.*"):
            img = cv2.imread(f)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Haar works on grayscale
                img = cv2.resize(img, IMG_SIZE)
                X.append(img)
                y.append(label)
    return np.array(X), np.array(y)

def extract_haar_features(img):
    """Extract Haar features"""
    features = []
    
    # Define different rectangle sizes for feature extraction
    sizes = [(2, 2), (4, 4), (8, 8)]
    
    for size in sizes:
        height, width = size
        step_y = img.shape[0] // height
        step_x = img.shape[1] // width
        
        # Extract vertical edge features (difference between two adjacent rectangles)
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
    
    # Add LBP-like features (comparing center region with surrounding regions)
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

print("Loading data...")
X_raw, y = load_data(DATA_DIR)

# Extract Haar features
print("Extracting Haar features...")
features = []
for img in X_raw:
    features.append(extract_haar_features(img))
features = np.array(features)

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(
    features, y, test_size=0.2, random_state=42
)

# Train AdaBoost classifier
print("Training AdaBoost classifier...")
clf = AdaBoostClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save model
MODEL4_PATH = os.path.join(RESULTS_DIR, "model4_haar_adaboost.pkl")
with open(MODEL4_PATH, "wb") as f:
    pickle.dump(clf, f)
print(f"Model saved: {MODEL4_PATH}")

# Evaluation
print("Evaluating model...")
start_time = time.time()
y_pred = clf.predict(X_test)
end_time = time.time()
inference_time_per_image = (end_time - start_time) / len(X_test)

cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred, target_names=["Fake", "Real"])

# Save confusion matrix
plt.figure()
plt.imshow(cm, cmap='Blues')
plt.title("Model 4 (Haar+AdaBoost) Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i,j]), ha="center", va="center")
plt.colorbar()
cm_path = os.path.join(RESULTS_DIR, "model4_confmat.png")
plt.savefig(cm_path)
plt.close()
print("Confusion matrix saved:", cm_path)

# Model size
model_size_bytes = os.path.getsize(MODEL4_PATH)
model_size_mb = model_size_bytes / (1024*1024)

# Number of parameters (number of weak classifiers in AdaBoost * number of features)
num_params = clf.n_estimators * X_train.shape[1]

# Save statistics
stats_path = os.path.join(RESULTS_DIR, "model4_stats.txt")
with open(stats_path, "w") as f:
    f.write("=== Model 4 (Haar+AdaBoost) Metrics ===\n\n")
    f.write("Classification Report:\n")
    f.write(cr + "\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")
    f.write(f"Inference time (sec/image): {inference_time_per_image:.6f}\n")
    f.write(f"Model size on disk (MB): {model_size_mb:.2f}\n")
    f.write(f"Number of parameters (est): {num_params}\n")

print("Statistics written to:", stats_path)
print("Done.")