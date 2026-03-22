"""This simple CNN model can run evaluations for bot the Roboflow and DIBaS trained models"""
"""Please run this file in the parent directory"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
from pathlib import Path
import numpy as np
import cv2
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import seaborn as sns

class EqualizeGrayscale:
    """Transforms image to grayscale and applies histogram equalization"""
    def __call__(self, img):
        img = img.convert("L")
        img_np = np.array(img)
        equalized = cv2.equalizeHist(img_np)
        return Image.fromarray(equalized)

class SimpleCNN(nn.Module):
    """A simple CNN with two convolution layers and three fully connected layers"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):   #Forward pass applies ReLU for each layer
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def preprocess_data(image_folder_path):
    """Define data augmentation with random flipping only for the Roboflow dataset"""
    transform_pipeline = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        EqualizeGrayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = datasets.ImageFolder(root=image_folder_path, transform=transform_pipeline)
    return dataset

def compute_accuracy(model, dataloader, device):
    """Compute model accuracy on a dataloader"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total * 100

def real_evaluation(folder_path, answer, model_path):
    """Evaluate all images in a folder and print prediction and accuracy, default model is DIBaS"""
    class_names = ["E. Coli", "Staphylococcus"]
    if answer not in class_names:
        raise ValueError("Answer must be 'E. Coli' or 'Staphylococcus'.")
    true_index = class_names.index(answer)
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    transform = transforms.Compose([
        EqualizeGrayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    folder = Path(folder_path)
    for image_file in folder.glob("*"):
        if image_file.suffix.lower() not in [".jpg", ".png", ".jpeg", ".tif"]:
            continue
        image = Image.open(image_file)
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image)
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            is_correct = pred_class == true_index
            y_true.append(true_index)
            y_pred.append(pred_class)
            correct += int(is_correct)
            total += 1
            print(f"{image_file.name} → {class_names[pred_class]} ({probs[0][pred_class]:.2%}) | Correct: {is_correct}")
    accuracy = correct / total * 100 if total > 0 else 0
    f1 = f1_score(y_true, y_pred, average="binary", pos_label=true_index) if total > 0 else 0
    print(f"\nAccuracy for {answer}: {correct}/{total} = {accuracy:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    return y_true, y_pred

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix", save_path=None):
    """Plot a confusion matrix using multiple real_evaluation folder outputs"""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def train_model(trainloader, testloader, num_epochs=5, learning_rate=0.001):
    """Train and save the Simple CNN model with optimized learning rate"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)  #Initialize model
    entropy_loss = nn.CrossEntropyLoss()  #Define loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  #Use Adam optimizer with optimized learning rate
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()  #Set model to training mode
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  #Reset gradients to zero before each step
            outputs = model(inputs)  #make the predictions
            loss = entropy_loss(outputs, labels)  #calculate loss between predicted and true
            loss.backward()  #calculate the gradients for the weights
            optimizer.step()  #update weights based on gradients
            running_loss += loss.item()  #track total loss for this epoch
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss:.4f}")
    train_acc = compute_accuracy(model, trainloader, device)    #Evaluate
    test_acc = compute_accuracy(model, testloader, device)
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    torch.save(model.state_dict(), "simple_cnn_flipped.pth")  # Save trained model weights
    print("Model saved to simple_cnn.pth")
    return model

def k_fold_cross_validation_simple_cnn(dataset, k=5, batch_size=32, num_epochs=5, learning_rate=0.001):
    """Run 5-fold validation and save best model by validation accuracy"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kfold = KFold(n_splits=k, shuffle=True)
    fold_losses = []
    best_val_accuracy = 0.0
    best_model_state = None
    for fold, (train_indices, val_indices) in enumerate(kfold.split(dataset)):
        print(f"\nFold {fold+1}/{k}")
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        model = SimpleCNN().to(device)
        entropy_loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        epoch_losses = []
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = entropy_loss(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(train_loader)
            epoch_losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
        val_accuracy = compute_accuracy(model, val_loader, device)
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
        fold_losses.append(epoch_losses)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()
    if best_model_state is not None:    #Save the highest accuracy model (the last one with 100%)
        torch.save(best_model_state, "simple_cnn.pth")
        print(f"\nBest model saved with validation accuracy: {best_val_accuracy:.2f}%")
    return fold_losses

def learning_rate_sweep_simple_cnn(train_loader, learning_rates, num_batches=100):
    """Test different learning rates on a training batches until the data runs out (around 20 not 100)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    entropy_loss = nn.CrossEntropyLoss()
    for lr in learning_rates:
        print(f"\nTesting learning rate: {lr}")
        model = SimpleCNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        batch_losses = []
        for i, (images, labels) in enumerate(train_loader):
            if i >= num_batches:
                break
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = entropy_loss(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        results[lr] = batch_losses
    return results

def plot_learning_rate(lr_results, clip_max=10):
    """Plot training loss for different learning rates with a max of 10 so the plot looks ok"""
    plt.figure(figsize=(7, 5))
    for lr, losses in lr_results.items():
        if clip_max:
            losses = [min(loss, clip_max) for loss in losses]
        plt.plot(range(1, len(losses) + 1), losses, label=f"lr={lr}")
    plt.xlabel("Batch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss per Fold - Simple CNN with DIBaS")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_training_loss(losses, model_label="Model"):
    """Plot training loss for one or more folds"""
    plt.figure(figsize=(6, 4))
    if isinstance(losses[0], list):  # multiple folds
        for i, curve in enumerate(losses):
            plt.plot(range(1, len(curve)+1), curve, label=f"Fold {i+1}")
    else:  # single run
        plt.plot(range(1, len(losses)+1), losses, label=model_label)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(f"Training Loss Curve - {model_label}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    dataset_path = "Microbes"
    dataset = preprocess_data(dataset_path)

    """These two lines will run evaluation with simple CNN trained on DIBaS dataset"""
    y_true_1, y_pred_1 = real_evaluation("Microbes/E. Coli", "E. Coli", model_path="simple_cnn_DIBaS.pth")
    y_true_2, y_pred_2 = real_evaluation("Microbes/Staphylococcus", "Staphylococcus", model_path="simple_cnn_DIBaS.pth")

    """These two lines will run evaluation with simiple CNN trained on Roboflow dataset"""
    #y_true_1, y_pred_1 = real_evaluation("DIBaS/Augmented_Staphylococcus/","Staphylococcus", model_path="simple_cnn_Roboflow.pth")
    #y_true_2, y_pred_2 = real_evaluation("DIBaS/Augmented_E.Coli/", "E. Coli", model_path="simple_cnn_Roboflow.pth")
    
    y_true_all = y_true_1 + y_true_2
    y_pred_all = y_pred_1 + y_pred_2
    class_names = ["E. Coli", "Staphylococcus"]
    plot_confusion_matrix(y_true_all, y_pred_all, class_names, title="Confusion Matrix - Simple CNN-DIBaS")

    # Split into train (80%) and test (20%)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    """This line of code is for training the model with cross validation"""
    #losses = k_fold_cross_validation_simple_cnn(dataset)

    #plot_training_loss(losses, model_label="Simple_CNN")
    #learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    #lr_results = learning_rate_sweep_simple_cnn(train_loader,learning_rates)
    #plot_learning_rate(lr_results)
    # Train and evaluate
    #model = train_model(train_loader, test_loader, num_epochs=10, learning_rate=0.001)

main()