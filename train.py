import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ------------------------------
    # Transforms
    # ------------------------------
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ------------------------------
    # Dataset
    # ------------------------------
    data_path = r"D:\New folder\human_rec\Structured\train"
    dataset = datasets.ImageFolder(data_path, transform=train_transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    val_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    # ------------------------------
    # Model
    # ------------------------------
    weights = EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    # Freeze all features initially
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(dataset.classes)
    )

    model.to(device)

    # ------------------------------
    # Loss & Optimizer
    # ------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=3e-4)

    # ------------------------------
    # Training Loop
    # ------------------------------
    epochs = 30

    for epoch in range(epochs):
        # ===== TRAIN =====
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100 * correct / total

        # ===== VALIDATION =====
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100 * correct / total

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% || "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

    # ------------------------------
    # Save the trained model
    # ------------------------------
    save_path = r"D:\New folder\human_rec\efficientnet_action_model.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": dataset.classes
    }, save_path)

    print("\n✅ Model saved successfully at:", save_path)


if __name__ == "__main__":
    main()
