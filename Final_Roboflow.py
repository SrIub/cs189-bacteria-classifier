"""Please run this file in the parent directory"""
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import numpy as np
import cv2
import os
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns


class EqualizeGrayscale:
    """Transforms image to remove color, brightness, etc. biases"""
    def __call__(self, img):
        img = img.convert("L")  # First change to grayscale for different dyes
        img_np = np.array(img)
        equalized = cv2.equalizeHist(img_np)  # Use openCV to do histogram equalization and normalize image brightness
        return Image.fromarray(equalized)
    
def preprocess_data(image_folder_path):
    """Define data augmentation and preprocessing pipeline for training"""
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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  #Best learning rate was found using learning rate graph
    return model, loss_function, optimizer, device


def train_model(model, train_data, loss_function, optimizer, device, num_epochs=5):
    """Train the model with 5 epochs"""
    model.train()
    epoch_losses = []
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
            predicted_logits, predicted_classes = torch.max(outputs, 1)     #This is the highest confidence prediction
            correct_predictions += (predicted_classes == labels).sum().item()
            total_predictions += labels.size(0)
        training_accuracy = correct_predictions / total_predictions * 100
        avg_loss = epoch_loss / len(train_data)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch_index+1}/{num_epochs} - Loss: {avg_loss:.4f} - Train Accuracy: {training_accuracy:.2f}%")
    return epoch_losses

def plot_all_folds(loss_curves, model_label="Model"):
    """Plot training loss for each fold"""
    plt.figure(figsize=(7, 5))
    for i, losses in enumerate(loss_curves):
        plt.plot(range(1, len(losses)+1), losses, label=f"Fold {i+1}")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(f"Training Loss per Fold - {model_label}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_model(model, data_loader, device):
    """Evaluate the model on test data (20%)"""
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():   #no_grad() because no need to adjust weights in evaluation
        for image_batch, label_batch in data_loader:
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            outputs = model(image_batch)
            _, predicted_classes = torch.max(outputs, 1)    #Again, best prediction
            correct_predictions += (predicted_classes == label_batch).sum().item()
            total_predictions += label_batch.size(0)
    return correct_predictions / total_predictions * 100

def learning_rate_sweep(train_data, learning_rates, num_batches=100):
    """Do learning rate sweep comparing loss per batch until the dataset runs out of batches (around 20 not 100)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entropy_loss = nn.CrossEntropyLoss()
    results = {}
    for lr in learning_rates:
        print(f"Testing learning rate: {lr}")
        model = build_model().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        batch_losses = []
        model.train()
        for i, (images, labels) in enumerate(train_data):
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
        fold_losses = train_model(model, train_data, loss_function, optimizer, device, num_epochs)
        all_loss_curves.append(fold_losses)
        val_accuracy = evaluate_model(model, test_data, device)
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
        fold_accuracies.append(val_accuracy)
        if val_accuracy == max(fold_accuracies):
            torch.save(model.state_dict(), "best_model_Roboflow.pth")
    avg_accuracy = np.mean(fold_accuracies)
    print(f"\nAverage K-Fold Validation Accuracy: {avg_accuracy:.2f}%")
    return all_loss_curves

def predict_image(image_path, model_path="best_model_Roboflow.pth"):
    """Use saved model to predict the label of a single image (from online)"""
    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()    #Load the model
    transform = transforms.Compose([
        EqualizeGrayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])      #Use the same transforms as the training data
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)   #Best prediction
    class_names = ["E. Coli", "Staphylococcus"]
    print(f"Prediction: {class_names[predicted_class.item()]}")
    print(torch.softmax(output, dim=1))

def real_evaluation(folder_path, true_label_name):
    """Evaluate a given folder with directory structure Label/images.jpg"""
    class_names = ["E. Coli", "Staphylococcus"]
    true_label_index = class_names.index(true_label_name)
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    model = build_model()
    model.load_state_dict(torch.load("best_model_Roboflow.pth", map_location=torch.device("cpu")))
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
        print(f"Predicted: {class_names[predicted_class]}, Correct: {is_correct}, Confidence: {probs[0][predicted_class]:.2%}\n")
    accuracy = correct / total * 100
    f1 = f1_score(y_true, y_pred, average="binary", pos_label=true_label_index)
    print(f"Accuracy for {true_label_name}: {correct}/{total} = {accuracy:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    return y_true, y_pred

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix", save_path=None):
    """Use real_evaluation to plot a confusion matrix of multiple labels"""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved confusion matrix to {save_path}")
    plt.show()

def main():
    dataset = preprocess_data("Microbes")

    #Run this two lines of code in order to evaluate this ResNet18 Roboflow model on the DIBaS dataset
    y_true_1, y_pred_1 = real_evaluation("DIBaS/Augmented_Staphylococcus/","Staphylococcus")
    y_true_2, y_pred_2 = real_evaluation("DIBaS/Augmented_E.Coli/", "E. Coli")
    
    #plot_confusion_matrix(y_true_1 + y_true_2, y_pred_1 + y_pred_2, class_names=["E. Coli", "Staphylococcus"], title="Roboflow Evaluation Confusion Matrix")
    
    """This code is for evaluating on a subset of the training dataset that its not trained on"""
    #real_evaluation("Roboflow Validation/E. Coli", "E. Coli")
    #real_evaluation("Roboflow Validation/Staphylococcus", "Staphylococcus")

    """This is for training or other stuff"""
    #loss_curves = k_fold_cross_validation(dataset, k=5, batch_size=32, num_epochs=5, learning_rate=0.001)
    #plot_all_folds(loss_curves, model_label="ResNet18 with Roboflow")

    #class_names = dataset.classes
    #visualize_batch(dataset, class_names)
    #learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    #learning_rate_results = learning_rate_sweep(train_loader, learning_rates)
    #plot_learning_rate_losses(learning_rate_results)
    #model = build_model()
    #model, loss_function, optimizer, device = setup_training(model)
    #losses = train_model(model, train_loader, test_loader, loss_function, optimizer, device, num_epochs=5)

main()
