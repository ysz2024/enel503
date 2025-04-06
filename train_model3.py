import os, glob, cv2, time
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Configuration parameters - IMPROVED VALUES
DATA_DIR = "dataset"
IMG_SIZE = (128, 128)
RESULTS_DIR = "results"
MAX_KEYPOINTS = 75  # Increased from 25 to 75 for better feature extraction
PCA_COMPONENTS = 500  # Increased from 200 to 500 to retain more variance
PARALLEL_WORKERS = max(1, multiprocessing.cpu_count() - 1)
BATCH_SIZE = 200

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

def extract_sift(img_path, max_kp=MAX_KEYPOINTS):
    """Extract SIFT features from a single image path"""
    # Read and resize image
    img = cv2.imread(img_path)
    if img is None:
        # Return zeros if image can't be read
        return np.zeros(max_kp * 128, dtype=np.float32)
        
    img = cv2.resize(img, IMG_SIZE)
    
    # Apply contrast enhancement to highlight features
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
    
    # Extract SIFT features with higher contrast threshold for better features
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

def extract_hog(img_path):
    """Extract HOG features for additional texture information"""
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros(3780, dtype=np.float32)  # Default HOG feature size
        
    img = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
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

def process_batch(batch_data):
    """Process a batch of images extracting both SIFT and HOG features"""
    paths, indices = batch_data
    features = []
    
    for i, path in enumerate(paths):
        # Extract SIFT features
        sift_feat = extract_sift(path)
        
        # Extract HOG features
        hog_feat = extract_hog(path)
        
        # Combine features
        combined_feat = np.concatenate([sift_feat, hog_feat])
        
        features.append((indices[i], combined_feat))
    
    return features

def main():
    """Main function to run the processing pipeline"""
    print("Loading image paths...")
    
    # Load paths instead of images
    paths = []
    labels = []
    for cls, label in [("Fake", 0), ("Real", 1)]:
        fdir = os.path.join(DATA_DIR, cls)
        for f in glob.glob(fdir + "/*.*"):
            paths.append(f)
            labels.append(label)
    
    labels = np.array(labels)
    print(f"Loaded {len(paths)} image paths")
    
    # Create batches for parallel processing
    print("Preparing batches for parallel processing...")
    batches = []
    for i in range(0, len(paths), BATCH_SIZE):
        end = min(i + BATCH_SIZE, len(paths))
        batches.append((paths[i:end], list(range(i, end))))
    
    # Sequential processing option if parallel is causing issues
    use_parallel = True
    
    if use_parallel:
        print(f"Extracting SIFT+HOG features using {PARALLEL_WORKERS} workers...")
        all_features = []
        
        with ProcessPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            futures = list(executor.map(process_batch, batches))
            total_batches = len(batches)
            
            print(f"Processing {total_batches} batches:")
            for i, batch_result in enumerate(futures):
                if (i+1) % 10 == 0 or i+1 == total_batches:
                    print(f"  Completed {i+1}/{total_batches} batches ({(i+1)/total_batches*100:.1f}%)")
                all_features.extend(batch_result)
    else:
        print("Extracting SIFT+HOG features sequentially...")
        all_features = []
        total_batches = len(batches)
        
        for i, batch in enumerate(batches):
            batch_result = process_batch(batch)
            all_features.extend(batch_result)
            
            if (i+1) % 10 == 0 or i+1 == total_batches:
                print(f"  Completed {i+1}/{total_batches} batches ({(i+1)/total_batches*100:.1f}%)")
    
    # Sort features by original indices
    all_features.sort(key=lambda x: x[0])
    features = np.array([f[1] for f in all_features], dtype=np.float32)
    print(f"Extracted {features.shape[0]} feature vectors, each with {features.shape[1]} dimensions")
    
    # Split into training and testing sets
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels  # Added stratification
    )
    
    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    # Apply PCA for dimensionality reduction
    print(f"Applying PCA dimensionality reduction to {PCA_COMPONENTS} components...")
    pca = PCA(n_components=PCA_COMPONENTS, svd_solver='randomized', random_state=42)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_test_pca = pca.transform(X_test_sc)
    print(f"Variance retained by PCA: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # Train LinearSVC with balanced or slightly real-biased class weights
    print("Training LinearSVC classifier...")
    start_time = time.time()
    # Changed class weights to favor real class slightly - this is key to improving real face detection
    clf = LinearSVC(dual=False, C=1.0, max_iter=3000, class_weight={0: 1.0, 1: 1.2})
    clf.fit(X_train_pca, y_train)
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Save the model
    MODEL3_PATH = os.path.join(RESULTS_DIR, "model3_svm.pkl")
    with open(MODEL3_PATH, "wb") as f:
        pickle.dump((clf, scaler, pca), f)
    print(f"Saved model to {MODEL3_PATH}")
    
    # Evaluate the model
    print("Evaluating model...")
    start_time = time.time()
    y_pred = clf.predict(X_test_pca)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=["Fake", "Real"])
    print(cr)
    
    # Calculate inference time on a sample of test images
    print("Measuring inference time...")
    sample_size = min(50, len(paths))
    sample_paths = [paths[i] for i in range(sample_size)]
    
    inference_start = time.time()
    for path in sample_paths:
        # Complete prediction pipeline for a single image
        sift_feat = extract_sift(path)
        hog_feat = extract_hog(path)
        combined_feat = np.concatenate([sift_feat, hog_feat])
        
        feat_sc = scaler.transform([combined_feat])
        feat_pca = pca.transform(feat_sc)
        _ = clf.predict(feat_pca)
    inference_end = time.time()
    
    inference_time_per_image = (inference_end - inference_start) / sample_size
    
    # Save confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap='Blues')
    plt.title("Model 3 (Enhanced SIFT+HOG+SVM) Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12)
    plt.colorbar()
    cm_path = os.path.join(RESULTS_DIR, "model3_confmat.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")
    
    # Model stats
    model_size_bytes = os.path.getsize(MODEL3_PATH)
    model_size_mb = model_size_bytes / (1024 * 1024)
    num_params = clf.coef_.size + pca.components_.size
    
    # Save stats
    stats_path = os.path.join(RESULTS_DIR, "model3_stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"=== Model 3 (Enhanced SIFT+HOG+SVM) Metrics ===\n\n")
        f.write(f"Configuration:\n")
        f.write(f"- SIFT Keypoints: {MAX_KEYPOINTS}\n")
        f.write(f"- PCA Components: {PCA_COMPONENTS}\n")
        f.write(f"- Variance Retained: {np.sum(pca.explained_variance_ratio_):.4f}\n")
        f.write(f"- Feature Types: SIFT + HOG\n")
        f.write(f"- Class Weights: Fake=1.0, Real=1.2\n\n")
        f.write(f"Classification Report:\n")
        f.write(f"{cr}\n")
        f.write(f"Confusion Matrix:\n")
        f.write(f"{str(cm)}\n\n")
        f.write(f"Training time (seconds): {train_time:.2f}\n")
        f.write(f"Inference time (sec/image): {inference_time_per_image:.6f}\n")
        f.write(f"Model size on disk (MB): {model_size_mb:.2f}\n")
        f.write(f"Number of parameters: {num_params}\n")
    
    print(f"Saved statistics to {stats_path}")
    print("Done.")

# This is the critical part for multiprocessing on macOS
if __name__ == "__main__":
    main()