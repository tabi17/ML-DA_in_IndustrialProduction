import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim


class CityScapesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: folderul cu subfoldere pentru fiecare clasa
        transform: torchvision.transforms
        """
        self.images = []
        self.labels = []
        self.transform = transform

        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        for cls in classes:
            cls_folder = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_folder):
                if fname.endswith(('.png', '.jpg', '.jpeg')):  # doar imagini
                    self.images.append(os.path.join(cls_folder, fname))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")  # asigură 3 canale
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


transform = transforms.Compose([
    transforms.Resize((128, 128)),        # sau 299x299 pentru Inception
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


dataset_path = "/home/tabita17/Documents/SCHOOL/M1/IAADPI/iad_doc/ML&DA-lab/date/gtFine_trainvaltest/cityscapes/train"  # înlocuiește cu path-ul tău
dataset = CityScapesDataset(dataset_path, transform=transform)

# Split 70% train / 30% test
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Model CNN (ResNet50)
num_classes = len(os.listdir(dataset_path))
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)


# Loss și optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")


# Evaluare
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
