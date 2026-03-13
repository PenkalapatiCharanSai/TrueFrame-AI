import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

DEVICE = torch.device("cpu")
DATA_DIR = "balanced_dataset"   # IMPORTANT
MODEL_PATH = "models/best_model.pth"

BATCH_SIZE = 32   # Bigger batch = faster CPU training
EPOCHS = 15
PATIENCE = 4
LR = 0.0001

# -------- TRANSFORMS --------
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR,"train"), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR,"val"), transform=val_transform)

print("Train samples:", len(train_dataset))
print("Val samples:", len(val_dataset))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# -------- MODEL --------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze everything
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last block only (faster training)
for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 2)
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    patience=2,
    factor=0.5,
    verbose=True
)

best_acc = 0
trigger = 0

print("\n===== TRAINING START =====")

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    # ---- TRAIN ----
    model.train()
    train_correct = 0
    train_total = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs,1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_acc = train_correct / train_total
    print("Train Acc:", round(train_acc,4))

    # ---- VALIDATION ----
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs,1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    print("Val Acc:", round(val_acc,4))

    scheduler.step(val_acc)

    if val_acc > best_acc:
        best_acc = val_acc
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        trigger = 0
        print("✅ Model Saved")
    else:
        trigger += 1
        print(f"No improvement ({trigger}/{PATIENCE})")

    if trigger >= PATIENCE:
        print("⛔ Early stopping")
        break

print("\n🎯 Training Complete")
print("Best Validation Accuracy:", best_acc)
