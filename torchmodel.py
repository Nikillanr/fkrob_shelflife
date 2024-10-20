import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# 1. Define data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# 2. Load the dataset
data_dir = './dataset'  # Replace with your dataset path

# Use ImageFolder for training and validation
full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])

# Split dataset into training and validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Update the transform for validation dataset
val_dataset.dataset.transform = data_transforms['val']

# Create dataloaders
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4),
    'val': DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
}

# Get class names
class_names = full_dataset.classes
print(f"Classes: {class_names}")

# 3. Define the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = models.efficientnet_b0(pretrained=True)

# Freeze the base model
for param in model.parameters():
    param.requires_grad = False

# Modify the classifier
num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(num_ftrs, 1),
    nn.Sigmoid()
)

model = model.to(device)
print(model)

# 4. Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# 5. Training loop
def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=10):
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = (outputs > 0.5).float()

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    print(f'Best Validation Accuracy: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

# Train the model
model, history = train_model(model, dataloaders, criterion, optimizer, device, num_epochs=10)

# 6. Evaluate the model
def evaluate_model(model, dataloaders, device):
    model.eval()
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(inputs)
            preds = (outputs > 0.5).float()

            running_corrects += torch.sum(preds == labels)

    accuracy = running_corrects.double() / len(dataloaders['val'].dataset)
    print(f'Validation Accuracy: {accuracy:.4f}')

evaluate_model(model, dataloaders, device)

# 7. Plot training history
def plot_training_history(history, num_epochs):
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), history['train_loss'], label='Train Loss')
    plt.plot(range(1, num_epochs+1), history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), history['train_acc'], label='Train Acc')
    plt.plot(range(1, num_epochs+1), history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    plt.show()

plot_training_history(history, num_epochs=10)

# 8. Estimate Shelf Life
def estimate_shelf_life(model, image_path, baseline_shelf_life, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        freshness_prob = model(image).item()

    initial_rot = 0.5
    rot_percent = initial_rot - (freshness_prob - initial_rot)
    rot_percent = max(0.0, min(1.0, rot_percent))
    estimated_shelf_life = baseline_shelf_life * (1 - rot_percent)

    return estimated_shelf_life, freshness_prob

# Example usage
sample_image_path = './test.jpg'  # Replace with your image path
baseline_shelf_life = 5  # Example: 10 days
shelf_life, freshness_prob = estimate_shelf_life(model, sample_image_path, baseline_shelf_life, device)
#round up to the nearest integer
shelf_life = np.ceil(shelf_life)

print(f"Estimated Shelf Life: {shelf_life:.2f} days")
print(f"Freshness Probability: {freshness_prob*100:.2f}%")
print(f"Rot Percentage: {(0.5 - freshness_prob)*100:.2f}%")
