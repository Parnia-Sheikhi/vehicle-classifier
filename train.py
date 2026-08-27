import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# for training data I'm adding some augmentation (flip, rotate, color change)
# so the model doesn't just memorize the exact images and actually generalizes better
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

# no augmentation for validation, I want to see real performance on unmodified images
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root="train", transform=train_transform)
val_dataset = datasets.ImageFolder(root="val", transform=val_transform)
class_names = train_dataset.classes

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# function to build the model, so I don't repeat this code if I need it again
def build_model(num_classes):
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

    # freezing the pretrained layers, I only want to train my own last layer
    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model.to(device)


# runs one full pass over the training data
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


# checks how the model does on data it hasn't trained on
# using this both during training (for validation) and at the end (for the final report)
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy, all_preds, all_labels


# ------------------- actual training starts here -------------------

model = build_model(len(class_names))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
num_epochs = 10

# keeping track of everything so I can plot it later
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc = 0.0

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

    # only saving the model if it's better than what I had before
    # (last epoch isn't always the best one)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "vehicle_classifier.pth")
        print(f"  -> new best model, saved it (val acc: {val_acc:.2f}%)")

print(f"\nBest validation accuracy I got: {best_val_acc:.2f}%")

# ------------------- phase 2: fine-tuning -------------------
# now unlocking the last few conv blocks of the backbone and training
# them too, but with a much smaller learning rate so we don't wreck
# the pretrained weights, just nudge them toward our data

print("\nStarting fine-tuning phase...")

for param in model.features[-4:].parameters():
    param.requires_grad = False
    param.requires_grad = True

finetune_optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001
)

finetune_epochs = 5

for epoch in range(finetune_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, finetune_optimizer)
    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Fine-tune Epoch [{epoch+1}/{finetune_epochs}] "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "vehicle_classifier.pth")
        print(f"  -> new best model, saved it (val acc: {val_acc:.2f}%)")

print(f"\nBest validation accuracy after fine-tuning: {best_val_acc:.2f}%")

# plotting loss and accuracy so I can see how training went
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(history["train_loss"], label="Train Loss")
axes[0].plot(history["val_loss"], label="Val Loss")
axes[0].set_title("Loss over epochs")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history["train_acc"], label="Train Accuracy")
axes[1].plot(history["val_acc"], label="Val Accuracy")
axes[1].set_title("Accuracy over epochs")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plt.savefig("training_curves.png")
print("saved training_curves.png")

# loading back the best model (not necessarily the last epoch) for the final report
model.load_state_dict(torch.load("vehicle_classifier.pth", map_location=device))
_, _, all_preds, all_labels = evaluate(model, val_loader, criterion)

# confusion matrix to see which classes get mixed up
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, xticks_rotation=45, cmap="Blues")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("saved confusion_matrix.png")

# precision/recall/f1 per class, more detailed than just accuracy
report = classification_report(all_labels, all_preds, target_names=class_names)
print(report)
with open("classification_report.txt", "w") as f:
    f.write(report)
print("saved classification_report.txt")