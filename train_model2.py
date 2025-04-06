import os, glob, cv2, time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Configuration parameters
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = "dataset"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def hist_eq_rgb(img):
    """
    Apply histogram equalization to the Y channel in YUV color space.
    This enhances contrast while preserving color information.
    """
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

def edge_enhance(img):
    """
    Enhance edges in the image to emphasize potential artifacts in fake images.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply Laplacian edge detection
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Normalize to 0-255 range
    laplacian = np.uint8(np.absolute(laplacian))
    
    # Merge back with original image
    result = img.copy()
    
    # Subtly enhance edges in all channels
    for i in range(3):
        result[:,:,i] = cv2.addWeighted(result[:,:,i], 0.9, laplacian, 0.1, 0)
    
    return result

class MyDataset(Dataset):
    """Dataset class for loading real and fake face images with enhanced preprocessing"""
    def __init__(self, folder, transform=None, transform_test=None, apply_preprocessing=False, class_weights=None):
        self.paths = []
        self.labels = []
        for cls, label in [("Fake", 0), ("Real", 1)]:
            fdir = os.path.join(folder, cls)
            for f in glob.glob(fdir + "/*.*"):
                self.paths.append(f)
                self.labels.append(label)
        self.transform = transform
        self.transform_test = transform_test
        self.apply_preprocessing = apply_preprocessing
        self.class_weights = class_weights

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        label = self.labels[idx]
        
        # Apply selective preprocessing based on image class
        if self.apply_preprocessing:
            if label == 0:  # Fake image
                # Enhance edges to make artifacts more visible
                img = edge_enhance(img)
            else:  # Real image 
                # Apply histogram equalization to real images
                img = hist_eq_rgb(img)
        
        # Apply appropriate transforms
        if self.transform:
            img = self.transform(img)
        elif self.transform_test:
            img = self.transform_test(img)
        
        # Get sample weight if class_weights is provided
        sample_weight = 1.0
        if self.class_weights is not None:
            sample_weight = self.class_weights[label]
            
        return img, label, sample_weight

# Define transforms for model 2
augment_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(p=0.3),  # Reduce flip probability
    T.RandomRotation(10),  # Less aggressive rotation
    T.ColorJitter(brightness=0.1, contrast=0.1),  # Subtle color variations
    T.ToTensor(),
])

test_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
])

# Define class weights to focus more on fake detection
# This is key to improving fake image detection
class_weights = {0: 1.5, 1: 1.0}  # Weight fake images (0) higher than real images (1)

# Create dataset with preprocessing and class weights
full_dataset = MyDataset(DATA_DIR, apply_preprocessing=True, class_weights=class_weights)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_ds, test_ds = torch.utils.data.random_split(full_dataset, [train_size, test_size])

# Assign transforms
train_ds.dataset.transform = augment_transform
test_ds.dataset.transform_test = test_transform

# Custom collate function to handle sample weights
def collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    weights = torch.tensor([item[2] for item in batch])
    return imgs, labels, weights

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

class SimpleCNN(nn.Module):
    """
    Simple CNN architecture - keeping the same structure as model 1 
    for compatibility, but with improved training
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.2)  # Add dropout for regularization
        self.fc1 = nn.Linear(64 * (IMG_SIZE // 4) * (IMG_SIZE // 4), 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)  # Apply dropout
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

criterion = nn.BCEWithLogitsLoss(reduction='none')  # Use 'none' to apply per-sample weights
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)  # Add weight decay

# Learning rate scheduler to improve training
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# ------- TRAIN -------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for imgs, labels, weights in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        weights = weights.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        
        # Apply sample weights to the loss
        losses = criterion(outputs, labels)
        weighted_loss = (losses * weights).mean()
        
        weighted_loss.backward()
        optimizer.step()
        running_loss += weighted_loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")
    
    # Update learning rate based on loss
    scheduler.step(avg_loss)

MODEL2_PATH = os.path.join(RESULTS_DIR, "model2_cnn.pt")
torch.save(model.state_dict(), MODEL2_PATH)
print(f"Saved {MODEL2_PATH}")

# ------- EVALUATE -------
model.eval()
all_preds = []
all_targets = []
class_probs = {0: [], 1: []}  # Store probabilities for each class

start_time = time.time()
with torch.no_grad():
    for imgs, labels, _ in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = model(imgs)

        prob_real = torch.sigmoid(outputs).squeeze()
        prob_fake = 1.0 - prob_real

        # Store probabilities for analysis
        for i, label in enumerate(labels.cpu().numpy()):
            if i < len(prob_fake):
                class_probs[label].append(prob_fake[i].item() if label == 0 else prob_real[i].item())

        # If prob_fake >= 0.5 => label=0(Fake), else 1(Real)
        # Use a slightly lower threshold for fake detection to improve recall
        fake_threshold = 0.45  # Adjust threshold to favor detecting fakes
        preds = torch.where(
            prob_fake >= fake_threshold,
            torch.zeros_like(prob_fake).long(),
            torch.ones_like(prob_fake).long()
        )

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

end_time = time.time()
num_images = len(test_ds)
total_time = end_time - start_time
inference_time_per_image = total_time / num_images

# Generate and visualize confusion matrix
cm = confusion_matrix(all_targets, all_preds)
cr = classification_report(all_targets, all_preds, target_names=["Fake","Real"])

plt.figure()
plt.imshow(cm, cmap='Blues')
plt.title("Model 2 Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i,j]), ha="center", va="center")
plt.colorbar()
cm_path = os.path.join(RESULTS_DIR, "model2_confmat.png")
plt.savefig(cm_path)
plt.close()

# Calculate model statistics
model_size_bytes = os.path.getsize(MODEL2_PATH)
model_size_mb = model_size_bytes / (1024*1024)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Calculate average confidence for each class
avg_fake_conf = np.mean(class_probs[0]) if class_probs[0] else 0
avg_real_conf = np.mean(class_probs[1]) if class_probs[1] else 0

# Save statistics to file
stats_path = os.path.join(RESULTS_DIR, "model2_stats.txt")
with open(stats_path, "w") as f:
    f.write("=== Model 2 (Improved CNN with Focus on Fake Detection) ===\n\n")
    f.write("Classification Report:\n")
    f.write(cr + "\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")
    f.write(f"Inference time (sec/image): {inference_time_per_image:.6f}\n")
    f.write(f"Model size on disk (MB): {model_size_mb:.2f}\n")
    f.write(f"Number of parameters: {num_params}\n")
    f.write(f"Average confidence for fake images: {avg_fake_conf:.4f}\n")
    f.write(f"Average confidence for real images: {avg_real_conf:.4f}\n")
    f.write(f"Using adjusted threshold for fake detection: {fake_threshold}\n")

print("Saved confusion matrix:", cm_path)
print("Wrote stats to:", stats_path)
print("Done.")