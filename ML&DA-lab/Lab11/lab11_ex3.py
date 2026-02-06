import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


device = torch.device("cpu")
print("Using device:", device)


# DATASET
class CityScapesDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, max_images=20):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = sorted([
            f for f in os.listdir(img_dir)
            if f.endswith("_leftImg8bit.png")
        ])[:max_images]  # LIMITARE pt viteză

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.img_dir, img_name)
        mask_name = img_name.replace("_leftImg8bit.png",
                                     "_gtFine_labelIds.png")
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)

        mask = torch.from_numpy(np.array(mask)).long()
        return image, mask




transform = transforms.Compose([
    transforms.Resize((128, 128)), #248
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


images_dir = "/home/tabita17/Documents/SCHOOL/M1/IAADPI/iad_doc/ML&DA-lab/date/gtFine_trainvaltest/cityscapes/"
masks_dir  = "/home/tabita17/Documents/SCHOOL/M1/IAADPI/iad_doc/ML&DA-lab/date/gtInd/cityscapes/"

dataset = CityScapesDataset(images_dir, masks_dir, transform)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

print("Images loaded:", len(dataset))


# U-NET
class UNet(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU()
        )

        self.pool = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()
        )

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.out = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool(x1)
        x3 = self.enc2(x2)
        x4 = self.up(x3)
        return self.out(x4)


# TRAIN
model = UNet(num_classes=20).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 3

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(loader):.4f}")

print("✅ Training finished successfully!")
