import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

# -------------------------------
# Device
# -------------------------------
device = torch.device("cpu")
print("Using device:", device)

# -------------------------------
# Transforms
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# Load CityScapes subset
# -------------------------------
dataset_path = "/home/tabita17/Documents/SCHOOL/M1/IAADPI/iad_doc/ML&DA-lab/date/gtFine_trainvaltest/cityscapes/train"
full_dataset = datasets.ImageFolder(dataset_path, transform=transform)

# Folosim doar primele 100 imagini pentru demo
subset_indices = list(range(min(100, len(full_dataset))))
dataset = Subset(full_dataset, subset_indices)

train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# -------------------------------
# Model
# -------------------------------
num_classes = len(full_dataset.classes)
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

# -------------------------------
# Loss & Optimizer
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# Train Minimal Epoch
# -------------------------------
epochs = 1   # pentru demo, CPU
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i+1) % 10 == 0:
            print(f"Batch {i+1}, Loss: {running_loss/10:.4f}")
            running_loss = 0.0

print("Training finished on subset!")
