import os, glob, cv2, time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = "dataset"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def hist_eq_rgb(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

class MyDataset(Dataset):
    def __init__(self, folder, transform=None, transform_test=None):
        self.paths = []
        self.labels = []
        for cls, label in [("Fake", 0), ("Real", 1)]:
            fdir = os.path.join(folder, cls)
            for f in glob.glob(fdir + "/*.*"):
                self.paths.append(f)
                self.labels.append(label)
        self.transform = transform
        self.transform_test = transform_test

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Always do histogram eq
        img = hist_eq_rgb(img)

        # If self.transform is set, assume training
        # If self.transform_test is set, assume test
        if self.transform:
            img = self.transform(img)
        elif self.transform_test:
            img = self.transform_test(img)

        label = self.labels[idx]
        return img, label

augment_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(20),
    T.ToTensor()
])

test_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor()
])

full_dataset = MyDataset(DATA_DIR)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_ds, test_ds = torch.utils.data.random_split(full_dataset, [train_size, test_size])

# Assign transforms
train_ds.dataset.transform = augment_transform
test_ds.dataset.transform_test = test_transform

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ------- TRAIN -------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

MODEL2_PATH = os.path.join(RESULTS_DIR, "model2_cnn.pt")
torch.save(model.state_dict(), MODEL2_PATH)
print(f"Saved {MODEL2_PATH}")

# ------- EVALUATE -------
model.eval()
all_preds = []
all_targets = []

start_time = time.time()
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = model(imgs)

        prob_real = torch.sigmoid(outputs).squeeze()
        prob_fake = 1.0 - prob_real

        # If prob_fake >= 0.5 => label=0(Fake), else 1(Real)
        preds = torch.where(
            prob_fake >= 0.5,
            torch.zeros_like(prob_fake).long(),
            torch.ones_like(prob_fake).long()
        )

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

end_time = time.time()
num_images = len(test_ds)
total_time = end_time - start_time
inference_time_per_image = total_time / num_images

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

model_size_bytes = os.path.getsize(MODEL2_PATH)
model_size_mb = model_size_bytes / (1024*1024)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

stats_path = os.path.join(RESULTS_DIR, "model2_stats.txt")
with open(stats_path, "w") as f:
    f.write("=== Model 2 (HistEq + Aug CNN) Metrics ===\n\n")
    f.write("Classification Report:\n")
    f.write(cr + "\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")
    f.write(f"Inference time (sec/image): {inference_time_per_image:.6f}\n")
    f.write(f"Model size on disk (MB): {model_size_mb:.2f}\n")
    f.write(f"Number of parameters: {num_params}\n")

print("Saved confusion matrix:", cm_path)
print("Wrote stats to:", stats_path)
print("Done.")
