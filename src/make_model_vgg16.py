# src/make_model_vgg16.py

import os
import sys
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import vgg16, VGG16_Weights

# Permet d'importer eval_metrics.py depuis le dossier src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from eval_metrics import eval_metrics


def create_data_loaders_for_vgg16(x_train, x_val, x_test, y_train, y_val, y_test, config):
    """
    Convertit les tableaux numpy en Tensors PyTorch, permute les dimensions,
    et crée des DataLoaders pour l'entraînement, validation et test.
    """
    # Convert one-hot labels to class indices si besoin
    y_train = np.argmax(y_train, axis=1) if len(y_train.shape) > 1 else y_train
    y_val = np.argmax(y_val, axis=1) if len(y_val.shape) > 1 else y_val
    y_test = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test

    # Conversion en tenseurs PyTorch et permutation des dimensions (B, H, W, C) -> (B, C, H, W)
    x_train = torch.tensor(x_train, dtype=torch.float32).permute(0, 3, 1, 2)
    x_val = torch.tensor(x_val, dtype=torch.float32).permute(0, 3, 1, 2)
    x_test = torch.tensor(x_test, dtype=torch.float32).permute(0, 3, 1, 2)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Création des datasets
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)

    # Création des DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    print(f"✔️ DataLoaders créés: Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    return train_loader, val_loader, test_loader


def create_vgg16_model(config):
    """
    Crée un modèle VGG16 pré-entraîné avec une nouvelle couche finale adaptée au nombre de classes.
    """
    weights = VGG16_Weights.IMAGENET1K_V1 if config.get("pretrained", False) else None
    model = vgg16(weights=weights)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, config["num_classes"])
    return model


def train_vgg16_model(model, train_loader, val_loader, config):
    """
    Entraîne le modèle VGG16 sur les données fournies, avec affichage des métriques à chaque époque.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        avg_train_loss = epoch_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct / total

        # Phase validation
        model.eval()
        val_loss = 0.0
        all_labels, all_preds, all_probs = [], [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader.dataset)
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        metrics = eval_metrics(all_labels, all_preds, all_probs)

        print(f"📦 Epoch {epoch+1}/{config['epochs']} - "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}")
        print(f"📊 Val Metrics - Acc: {metrics['accuracy']:.4f} | Prec: {metrics['precision']:.4f} | "
              f"Rec: {metrics['recall']:.4f} | LogLoss: {metrics['log_loss']:.4f} | "
              f"AUC: {metrics['mean_roc_auc']:.4f}")

        # Scheduler step
        scheduler.step(avg_val_loss)

    return model


def test_model_vgg16(model, test_loader):
    """
    Évalue le modèle sur l'ensemble de test et retourne les probabilités et métriques calculées.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_pred_proba = np.array(all_probs)

    metrics = eval_metrics(y_true, y_pred, y_pred_proba)
    return y_pred_proba, metrics
