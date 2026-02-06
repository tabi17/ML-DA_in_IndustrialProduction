import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.Compose([
    transforms.Resize((299, 299)), # neaparat pentru InceptionV3
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


dataset_path = "/home/tabita17/Documents/SCHOOL/M1/IAADPI/iad_doc/ML&DA-lab/date/caltech-101/101/caltech101"
dataset = datasets.ImageFolder(dataset_path, transform=transform)

num_classes = len(dataset.classes)
print("Number of classes:", num_classes)


train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Subset (30% din training) pentru niste teste mai rapide
# subset_train_size = int(0.3 * len(train_dataset))
# train_dataset_small = Subset(train_dataset, range(subset_train_size))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)


def train_and_evaluate(model, model_name, epochs=10):  #
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"\nTraining {model_name}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"{model_name} Accuracy on CalTech101: {accuracy:.2f}%")


# ResNet18

resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
train_and_evaluate(resnet, "ResNet18")


# ResNet50
# resnet = models.resnet50(pretrained=True)
# resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
# train_and_evaluate(resnet, "ResNet50")


# InceptionV3

inception = models.inception_v3(pretrained=True, aux_logits=True)  # aux_logits=False
inception.fc = nn.Linear(inception.fc.in_features, num_classes)

# Dezactivăm calculul auxiliar în timpul training-ului pentru CPU/test rapid
inception.aux_logits = False

train_and_evaluate(inception, "InceptionV3")
