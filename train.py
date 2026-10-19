import math
import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

use_pin = device.type == "cuda"

N_COPIES = 3
NUM_EPOCHS = 30
WARMUP_EPOCHS = 3

AUG_DIR = "./augmented_data"

base_resize = transforms.Resize((224, 224))

aug_original = base_resize

aug_color = transforms.Compose(
    [
        base_resize,
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    ]
)

aug_geometric = transforms.Compose(
    [
        base_resize,
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
    ]
)


augmentations_by_copy = [aug_original, aug_color, aug_geometric]

raw_dataset = datasets.OxfordIIITPet(
    root="./data", split="trainval", transform=None, download=True
)

labels_path = os.path.join(AUG_DIR, "labels.txt")
if not os.path.exists(labels_path):
    if os.path.exists(AUG_DIR):
        shutil.rmtree(AUG_DIR)
    
    os.makedirs(AUG_DIR, exist_ok=True)

    labels_list = []
    idx = 0
    total = len(raw_dataset) * N_COPIES

    for i in range(len(raw_dataset)):
        img, label = raw_dataset[i]
        for copy_idx in range(N_COPIES):
            aug_func = augmentations_by_copy[copy_idx]
            aug_img = aug_func(img)
            aug_img.save(os.path.join(AUG_DIR, f"{idx:06d}.jpg"))
            labels_list.append(label)
            idx += 1

        if (i + 1) % 500 == 0:
            print(
                f"  Processed {i + 1}/{len(raw_dataset)} originals "
                f"({idx}/{total} total)"
            )

    with open(labels_path, "w") as f:
        for label in labels_list:
            f.write(f"{label}\n")
    print(f"Done. Generated {idx} images.")
else:
    print(f"Using existing augmented dataset at {AUG_DIR}")


class AugmentedPetDataset(Dataset):
    def __init__(self, aug_dir, transform=None):
        self.aug_dir = aug_dir
        self.transform = transform
        with open(os.path.join(aug_dir, "labels.txt"), "r") as f:
            self.labels = [int(line.strip()) for line in f]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.aug_dir, f"{idx:06d}.jpg")
        img = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.18)),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_data = AugmentedPetDataset(AUG_DIR, transform=train_transform)
train_data_eval = AugmentedPetDataset(AUG_DIR, transform=eval_transform)

print(f"Training dataset size: {len(train_data)}")

train_loader = DataLoader(
    train_data, batch_size=64, shuffle=True, num_workers=0, pin_memory=use_pin
)
train_eval_loader = DataLoader(
    train_data_eval, batch_size=64, shuffle=False, num_workers=0, pin_memory=use_pin
)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class PetResNet(nn.Module):
    def __init__(self, num_classes=37):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.stage1 = nn.Sequential(ResBlock(64, 64), ResBlock(64, 64))
        self.stage2 = nn.Sequential(ResBlock(64, 128, stride=2), ResBlock(128, 128))
        self.stage3 = nn.Sequential(ResBlock(128, 256, stride=2), ResBlock(256, 256))
        self.stage4 = nn.Sequential(ResBlock(256, 512, stride=2), ResBlock(512, 512))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        return self.classifier(x)


model = PetResNet(num_classes=37).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.SGD(
    model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True
)




def get_lr(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    else:
        progress = (epoch - WARMUP_EPOCHS) / (NUM_EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1 + math.cos(math.pi * progress))


scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)

MODEL_PATH = "./model.pth"

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    scheduler.step()
    train_acc = 100 * correct / total if total > 0 else 0.0
    avg_loss = total_loss / max(1, (i + 1))
    lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
        f"Loss: {avg_loss:.4f} | "
        f"Accuracy (with aug): {train_acc:.2f}% | "
        f"LR: {lr:.5f}"
    )

torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in train_eval_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

final_train_acc = 100 * correct / total if total > 0 else 0.0
print(f"Final training accuracy: {final_train_acc:.2f}%")
