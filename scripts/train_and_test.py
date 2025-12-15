import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
from torch.utils.data import DataLoader

import os
os.chdir("..") 
from scripts.classifiers import *

def load_and_preprocess_data(data_path, test_size=0.2, random_state=42):
    """
    Args:
        data_path: Path to the CSV file
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        (X_train_scaled, X_test_scaled, y_train, y_test, n_features, n_classes, label_encoder)
    """
    df = pd.read_csv(data_path)
    
    feature_cols = [col for col in df.columns if col not in ['material', 'location', 'distance']]
    X = df[feature_cols].values
    y = df['material'].values
    
    # Remove rows with NaN material labels
    valid_mask = ~pd.isna(y)
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"Data shape: {X.shape}")
    print(f"Num classes: {len(np.unique(y))}")
    print(f"Classes: {np.unique(y)}")
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTrain set: {X_train_scaled.shape[0]} samples")
    print(f"Test set: {X_test_scaled.shape[0]} samples")
    print()
    
    n_features = X_train_scaled.shape[1]
    n_classes = len(np.unique(y_encoded))
    
    return X_train_scaled, X_test_scaled, y_train, y_test, n_features, n_classes, label_encoder

def train_sklearn_classifiers(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Train sklearn classifiers and return accuracy results.
    
    Args:
        X_train_scaled: Scaled training features
        X_test_scaled: Scaled test features
        y_train: Training labels
        y_test: Test labels
    
    Returns:
        Dictionary mapping classifier names to (accuracy, confusion_matrix)
    """
    classifiers = get_sklearn_classifiers()
    results = {}
    
    for name, clf in classifiers.items():
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        results[name] = {"accuracy": accuracy, "confusion_matrix": cm}
        print(f"{name} Accuracy: {accuracy:.4f}")
    
    return results

def train_pytorch_classifiers(X_train_scaled, X_test_scaled, y_train, y_test, n_features, n_classes, epochs=50, batch_size=32):
    """
    Train PyTorch neural network classifiers and return accuracy results.
    
    Args:
        X_train_scaled: Scaled training features
        X_test_scaled: Scaled test features
        y_train: Training labels
        y_test: Test labels
        n_features: Number of input features
        n_classes: Number of output classes
        epochs: Number of training epochs
        batch_size: Batch size for training
    
    Returns:
        Dictionary mapping model names to {"accuracy", "confusion_matrix", "loss_history"}
    """
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)
    
    train_dataset = MaterialDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    nn_models = get_pytorch_models(n_features, n_classes)
    results = {}
    
    for name, model in nn_models.items():
        model, loss_history = train_pytorch_model(model, train_loader, epochs=epochs)
        accuracy, y_pred = evaluate_pytorch_model(model, X_test_tensor, y_test_tensor)
        cm = confusion_matrix(y_test, y_pred)
        results[name] = {
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "loss_history": loss_history,
        }
        print(f"{name} Accuracy: {accuracy:.4f}")
    
    return results

def print_results(results):
    print("\nModel Accuracies:")
    for name, metrics in sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True):
        print(f"  {name:30s}: {metrics['accuracy']:.4f}")

def main():
    data_path = "features_dataset.csv"
    
    try:
        # Load and preprocess data
        X_train_scaled, X_test_scaled, y_train, y_test, n_features, n_classes, label_encoder = load_and_preprocess_data(data_path)
        
        # Train sklearn classifiers
        sklearn_results = train_sklearn_classifiers(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Train PyTorch classifiers
        pytorch_results = train_pytorch_classifiers(
            X_train_scaled, X_test_scaled, y_train, y_test, n_features, n_classes
        )
        
        # Combine results and print
        all_results = {**sklearn_results, **pytorch_results}
        print_results(all_results)
        
        # Print confusion matrices (labels are encoded integers)
        for name, metrics in all_results.items():
            cm = metrics.get("confusion_matrix")
            if cm is not None:
                print(f"\nConfusion matrix for {name}:")
                print(cm)
        
        # Example: decode label integers back to original class names for later plotting
        class_names = label_encoder.classes_
        print("\nClass order for confusion matrices:", class_names)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
