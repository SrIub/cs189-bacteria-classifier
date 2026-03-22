"""This model is basically the same as the Roboflow model just with DIBaS augmented images"""
"""Please run this file in the parent directory"""
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import os
from pathlib import Path
import seaborn as sns

class EqualizeGrayscale:
    """Transforms image to grayscale and applies histogram equalization"""
    def __call__(self, img):
        img = img.convert("L")
        img_np = np.array(img)
        equalized = cv2.equalizeHist(img_np)
        return Image.fromarray(equalized)

def augment_extra_images(input_folder, output_folder, target_count=500):
    """Augment starting 20 images per class to 500 per class with random flipping"""
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1))
    ])
    image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg")) + \
                  list(input_path.glob("*.png")) + list(input_path.glob("*.tif"))
    num_originals = len(image_files)
    augmentations_per_image = target_count // num_originals + 1
    count = 0
    for img_file in image_files:
        image = Image.open(img_file)
        for i in range(augmentations_per_image):
            if count >= target_count:
                break
            augmented = transform(image)
            save_path = output_path / f"{img_file.stem}_aug{i}.png"
            augmented.save(save_path)
            count += 1
    print(f"Saved {count} augmented images to {output_path}")

def preprocess_data(image_folder_path):
    """Define data augmentation and preprocessing pipeline for training(No rotation because of augmented images)"""
    transform_pipeline = transforms.Compose([
        EqualizeGrayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = datasets.ImageFolder(root=image_folder_path, transform=transform_pipeline)
    return dataset

def visualize_batch(dataset, class_names, num_images_to_display=8):
    """Visualize the image after grayscale (doesn't work after other augmentations)"""
    data_loader = DataLoader(dataset, batch_size=num_images_to_display, shuffle=True)
    image_batch, label_batch = next(iter(data_loader))
    image_batch = image_batch * 0.5 + 0.5
    fig, axes = plt.subplots(1, num_images_to_display, figsize=(15, 3))
    for i in range(num_images_to_display):
        image = image_batch[i].squeeze(0).numpy()
        axes[i].imshow(image, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(class_names[label_batch[i]])
    plt.tight_layout()
    plt.show()

def build_model():
    """Use pretrained ResNet18 model. Accepts grayscaled image (1 channel) and outputs 2 classes"""
    model = models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

def setup_training(model, learning_rate=0.001):
    """Setup variables for training. Device is cpu, loss function is CrossEntropyLoss(), optimizer is Adam"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, loss_function, optimizer, device

def train_model(model, train_data, loss_function, optimizer, device, num_epochs=5):
    """Train the model with 5 epochs"""
    model.train()
    losses = []
    for epoch_index in range(num_epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        for images, labels in train_data:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            _, predicted_classes = torch.max(outputs, 1)
            correct_predictions += (predicted_classes == labels).sum().item()
            total_predictions += labels.size(0)
        training_accuracy = correct_predictions / total_predictions * 100
        print(f"Epoch {epoch_index+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Train Accuracy: {training_accuracy:.2f}%")
        losses.append(epoch_loss)
    return losses

def evaluate_model(model, data_loader, device):
    """Evaluate the model on test data (20%)"""
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for image_batch, label_batch in data_loader:
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            outputs = model(image_batch)
            _, predicted_classes = torch.max(outputs, 1)
            correct_predictions += (predicted_classes == label_batch).sum().item()
            total_predictions += label_batch.size(0)
    return correct_predictions / total_predictions * 100

def k_fold_cross_validation(dataset, k=5, batch_size=32, num_epochs=5, learning_rate=0.001):
    """Do 5 fold cross validation and save the best model based on accuracy"""
    kfold = KFold(n_splits=k, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_accuracies = []
    all_loss_curves = []
    for fold, (train_indices, val_indices) in enumerate(kfold.split(dataset)):
        print(f"\nFold {fold + 1}/{k}")
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        train_data = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_data = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        model = build_model()
        model, loss_function, optimizer, device = setup_training(model, learning_rate)
        epoch_losses = train_model(model, train_data, loss_function, optimizer, device, num_epochs)
        val_accuracy = evaluate_model(model, test_data, device)
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
        # Save model per fold
        model_path = "best_model_DIBaS.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Saved model for Fold {fold + 1} to {model_path}")
        fold_accuracies.append(val_accuracy)
        all_loss_curves.append(epoch_losses)
    avg_accuracy = np.mean(fold_accuracies)
    print(f"\nAverage K-Fold Validation Accuracy: {avg_accuracy:.2f}%")
    return all_loss_curves

def real_evaluation(folder_path, true_label_name):
    """Evaluate a given folder with directory structure Label/images.jpg"""
    class_names = ["E. Coli", "Staphylococcus"]
    true_label_index = class_names.index(true_label_name)
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    model = build_model()
    model.load_state_dict(torch.load("best_model_DIBaS.pth", map_location=torch.device("cpu")))
    model.eval()
    transform = transforms.Compose([
        EqualizeGrayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif")):
            continue
        image_path = os.path.join(folder_path, filename)
        print(f"Evaluating: {filename}")
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(image)
            probs = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probs).item()
        y_true.append(true_label_index)
        y_pred.append(predicted_class)
        is_correct = predicted_class == true_label_index
        correct += int(is_correct)
        total += 1
        print(f"→ Predicted: {class_names[predicted_class]}, Correct: {is_correct}, Confidence: {probs[0][predicted_class]:.2%}\n")
    accuracy = correct / total * 100
    f1 = f1_score(y_true, y_pred, average="binary", pos_label=true_label_index)
    print(f"Accuracy for {true_label_name}: {correct}/{total} = {accuracy:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    return y_true, y_pred

def plot_training_loss(losses, model_label="Model"):
    plt.figure(figsize=(6, 4))
    if isinstance(losses[0], list):
        for i, curve in enumerate(losses):
            plt.plot(range(1, len(curve)+1), curve, label=f"Fold {i+1}")
    else:
        plt.plot(range(1, len(losses)+1), losses, label=model_label)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(f"Training Loss Curve - {model_label}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def learning_rate_sweep_from_dataset(dataset, learning_rates, batch_size=32, num_batches=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    results = {}
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for lr in learning_rates:
        print(f"\nTesting learning rate: {lr}")
        model = build_model().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        batch_losses = []
        model.train()

        for i, (images, labels) in enumerate(train_loader):
            if i >= num_batches:
                break
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        results[lr] = batch_losses

    return results

def plot_learning_rate(lr_results):
    plt.figure(figsize=(7, 5))
    for lr, losses in lr_results.items():
        plt.plot(range(1, len(losses) + 1), losses, label=f"lr={lr}")
    plt.xlabel("Batch")
    plt.ylabel("Training Loss")
    plt.title("Learning Rate Sweep - Training Loss per Batch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def main():
    dataset = preprocess_data("DIBas")
    """This line of code is for training with cross validation (basically gets 100% training and testing accuracy for each fold)"""
    #k_fold_cross_validation(dataset)

    """These two lines of code are for testing the model on Roboflow dataset"""
    y_true_1, y_pred_1 = real_evaluation("Microbes/E. Coli", "E. Coli")
    y_true_2, y_pred_2 = real_evaluation("Microbes/Staphylococcus", "Staphylococcus")
    plot_confusion_matrix(y_true_1 + y_true_2, y_pred_1 + y_pred_2, class_names=["E. Coli", "Staphylococcus"], title="DIBaS Evaluation Confusion Matrix")

    """This was for augmenting the images"""
    #augment_extra_images("Staphylococcus.aureus", "Augmented_Staphylococcus")
    #augment_extra_images("Escherichia.coli", "Augmented_E.Coli")


    #losses = k_fold_cross_validation(dataset)
    #plot_training_loss(losses)
    #real_evaluation("Microbes/E. Coli/", "E. Coli")
    #real_evaluation("Microbes/Staphylococcus", "Staphylococcus")


    #class_names = dataset.classes
    #visualize_batch(dataset, class_names)
    #learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    #learning_rate_results = learning_rate_sweep(train_loader, learning_rates)
    #plot_learning_rate_losses(learning_rate_results)
    #model = build_model()
    #model, loss_function, optimizer, device = setup_training(model)
    #train_model(model, train_loader, test_loader, loss_function, optimizer, device, num_epochs=10)

main()
