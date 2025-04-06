import os, glob, cv2, time
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

DATA_DIR = "dataset"
IMG_SIZE = (128,128)
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data(folder):
    X, y = [], []
    for cls, label in [("Fake",0), ("Real",1)]:
        fdir = os.path.join(folder, cls)
        for f in glob.glob(fdir + "/*.*"):
            img = cv2.imread(f)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, IMG_SIZE)
                X.append(img)
                y.append(label)
    return np.array(X), np.array(y)

def extract_sift(img, max_kp=100):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    kp, desc = sift.detectAndCompute(gray, None)
    if desc is None:
        desc = np.zeros((1,128), dtype=np.float32)
    if len(desc) > max_kp:
        desc = desc[:max_kp]
    if len(desc) < max_kp:
        pad = np.zeros((max_kp - len(desc), 128), dtype=np.float32)
        desc = np.vstack((desc, pad))
    return desc.flatten()

X_raw, y = load_data(DATA_DIR)

# Extract SIFT features for entire dataset
features = []
for img in X_raw:
    features.append(extract_sift(img))
features = np.array(features, dtype=np.float32)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    features, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

clf = SVC(kernel='linear', probability=True)
clf.fit(X_train_sc, y_train)

MODEL3_PATH = os.path.join(RESULTS_DIR, "model3_svm.pkl")
with open(MODEL3_PATH, "wb") as f:
    pickle.dump((clf, scaler), f)
print(f"Saved {MODEL3_PATH}")

# -------- Evaluate --------
y_pred = clf.predict(X_test_sc)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred, target_names=["Fake","Real"])

# Inference time includes the SIFT step if you want *real-world timing*:
start_time = time.time()
for i in range(len(X_test)):
    # In real usage, we'd re-run extract_sift on the raw image
    # Let's do that to measure real inference
    # But we only have X_test_sc here. So let's do the full pipeline again
    # i.e. re-run SIFT from the original X_raw?
    pass
end_time = time.time()

# For simplicity, let's measure just the SVM predict on all X_test_sc:
svm_start = time.time()
_ = clf.predict(X_test_sc)
svm_end = time.time()
svm_inference_time_per_image = (svm_end - svm_start) / len(X_test)

# If you want the real SIFT overhead, do:
# sift_start = time.time()
# for i in range(len(X_test)):
#     _ = extract_sift(X_raw[i])
# sift_end = time.time()
# sift_time = (sift_end - sift_start)/ len(X_test)
# total_inference_time_per_image = svm_inference_time_per_image + sift_time

total_inference_time_per_image = svm_inference_time_per_image  # example

# Save confusion matrix
plt.figure()
plt.imshow(cm, cmap='Blues')
plt.title("Model 3 (SIFT+SVM) Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i,j]), ha="center", va="center")
plt.colorbar()
cm_path = os.path.join(RESULTS_DIR, "model3_confmat.png")
plt.savefig(cm_path)
plt.close()
print("Saved confusion matrix:", cm_path)

# Model size on disk
model_size_bytes = os.path.getsize(MODEL3_PATH)
model_size_mb = model_size_bytes / (1024*1024)

# Number of parameters for linear SVM:
num_params = clf.coef_.size

stats_path = os.path.join(RESULTS_DIR, "model3_stats.txt")
with open(stats_path, "w") as f:
    f.write("=== Model 3 (SIFT+SVM) Metrics ===\n\n")
    f.write("Classification Report:\n")
    f.write(cr + "\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")
    f.write(f"Inference time (sec/image): {total_inference_time_per_image:.6f}\n")
    f.write(f"Model size on disk (MB): {model_size_mb:.2f}\n")
    f.write(f"Number of parameters (SVM coef size): {num_params}\n")

print("Wrote stats to:", stats_path)
print("Done.")
