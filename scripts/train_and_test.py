import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
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

    unique_classes, counts = np.unique(y, return_counts=True)
    print("Num samples per class:")
    for cls, cnt in zip(unique_classes, counts):
        print(f"{cls}: {cnt}")
    
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

def train_sklearn_classifiers(X_train_scaled, X_test_scaled, y_train, y_test, pca=False, n_components=None, cv=10):
    """
    Train sklearn classifiers using k-fold cross-validation and return accuracy results.
    
    Args:
        X_train_scaled: Scaled training features
        X_test_scaled: Scaled test features
        y_train: Training labels
        y_test: Test labels
        pca: Boolean flag to apply PCA dimensionality reduction (default: False)
        n_components: Number of components to keep. If float (0 < n_components < 1), 
                     it's treated as the variance ratio to retain. If int, it's the 
                     number of components. If None and pca=True, keeps all components.
        cv: Number of folds for cross-validation (default: 10)
    
    Returns:
        Dictionary mapping classifier names to {"accuracy", "confusion_matrix"}
    """
    # Combine train and test data for cross-validation
    X_all = np.vstack([X_train_scaled, X_test_scaled])
    y_all = np.hstack([y_train, y_test])
    
    # Apply PCA if requested
    if pca:
        pca_transformer = PCA(n_components=n_components)
        X_all = pca_transformer.fit_transform(X_all)
        
        if n_components is None or (isinstance(n_components, float) and 0 < n_components < 1):
            actual_components = X_all.shape[1]
            explained_variance = pca_transformer.explained_variance_ratio_.sum()
            print(f"PCA applied: {actual_components} components, {explained_variance:.4f} variance explained")
        else:
            print(f"PCA applied: {X_all.shape[1]} components")
    
    # Use stratified k-fold for cross-validation
    cv_fold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    classifiers = get_sklearn_classifiers()
    results = {}
    
    for name, clf in classifiers.items():
        # Perform cross-validation to get accuracy scores
        cv_scores = cross_val_score(clf, X_all, y_all, cv=cv_fold, scoring='accuracy')
        accuracy = cv_scores.mean()
        accuracy_std = cv_scores.std()
        
        # Get cross-validation predictions for confusion matrix
        y_pred = cross_val_predict(clf, X_all, y_all, cv=cv_fold)
        cm = confusion_matrix(y_all, y_pred)
        
        results[name] = {"accuracy": accuracy, "confusion_matrix": cm}
        print(f"{name} Accuracy: {accuracy:.4f} (+/- {accuracy_std:.4f})")
    
    return results

def train_pytorch_classifiers(X_train_scaled, X_test_scaled, y_train, y_test, n_features, n_classes, epochs=200, batch_size=32, pca=False, n_components=None):
    """
    Train PyTorch neural network classifiers and return accuracy results.
    
    Args:
        X_train_scaled: Scaled training features
        X_test_scaled: Scaled test features
        y_train: Training labels
        y_test: Test labels
        n_features: Number of input features (before PCA if applied)
        n_classes: Number of output classes
        epochs: Number of training epochs
        batch_size: Batch size for training
        pca: Boolean flag to apply PCA dimensionality reduction (default: False)
        n_components: Number of components to keep. If float (0 < n_components < 1), 
                     it's treated as the variance ratio to retain. If int, it's the 
                     number of components. If None and pca=True, keeps all components.
    
    Returns:
        Dictionary mapping model names to {"accuracy", "confusion_matrix", "loss_history"}
    """
    # Apply PCA if requested
    if pca:
        pca_transformer = PCA(n_components=n_components)
        X_train_scaled = pca_transformer.fit_transform(X_train_scaled)
        X_test_scaled = pca_transformer.transform(X_test_scaled)
        
        # Update n_features to match the reduced dimensionality
        n_features = X_train_scaled.shape[1]
        
        if n_components is None or (isinstance(n_components, float) and 0 < n_components < 1):
            actual_components = X_train_scaled.shape[1]
            explained_variance = pca_transformer.explained_variance_ratio_.sum()
            print(f"PCA applied: {actual_components} components, {explained_variance:.4f} variance explained")
        else:
            print(f"PCA applied: {X_train_scaled.shape[1]} components")
    
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

def load_multilabel_data(data_path, label_columns=['material', 'location'], test_size=0.2, random_state=42):
    """
    Load and preprocess data for multilabel classification (2 labels).
    
    Args:
        data_path: Path to the CSV file
        label_columns: List of column names to use as labels (default: ['material', 'location'])
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        (X_train_scaled, X_test_scaled, y_train, y_test, n_features, label_encoders, scaler)
        where y_train and y_test are 2D arrays with shape (n_samples, n_labels)
    """
    df = pd.read_csv(data_path)
    
    feature_cols = [col for col in df.columns if col not in label_columns + ['distance']]
    X = df[feature_cols].values
    
    # Get labels - ensure all labels are valid (no NaN) for each sample
    valid_mask = np.ones(len(df), dtype=bool)
    for label_col in label_columns:
        valid_mask = valid_mask & (~pd.isna(df[label_col].values))
    
    # Filter X and labels to only include rows where all labels are valid
    X = X[valid_mask]
    
    y_labels = []
    label_encoders = []
    
    for label_col in label_columns:
        y = df[label_col].values[valid_mask]
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        y_labels.append(y_encoded)
        label_encoders.append(label_encoder)
    
    # Stack labels into 2D array: (n_samples, n_labels)
    y_multilabel = np.column_stack(y_labels)
    
    print(f"Data shape: {X.shape}")
    print(f"Number of labels: {len(label_columns)}")
    for i, (label_col, le) in enumerate(zip(label_columns, label_encoders)):
        unique_classes = le.classes_
        print(f"  {label_col}: {unique_classes}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_multilabel, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTrain set: {X_train_scaled.shape[0]} samples")
    print(f"Test set: {X_test_scaled.shape[0]} samples")
    print()
    
    n_features = X_train_scaled.shape[1]
    
    return X_train_scaled, X_test_scaled, y_train, y_test, n_features, label_encoders, scaler

def train_multilabel_classifiers(X_train_scaled, X_test_scaled, y_train, y_test, label_names=['material', 'location'], pca=False, n_components=None, cv=10):
    """
    Train multilabel classifiers using k-fold cross-validation and return accuracy results.
    
    Args:
        X_train_scaled: Scaled training features
        X_test_scaled: Scaled test features
        y_train: Training labels (2D array: n_samples x n_labels)
        y_test: Test labels (2D array: n_samples x n_labels)
        label_names: List of label names for reporting (default: ['material', 'location'])
        pca: Boolean flag to apply PCA dimensionality reduction (default: False)
        n_components: Number of components to keep. If float (0 < n_components < 1), 
                     it's treated as the variance ratio to retain. If int, it's the 
                     number of components. If None and pca=True, keeps all components.
        cv: Number of folds for cross-validation (default: 10)
    
    Returns:
        Dictionary mapping classifier names to {"accuracy", "hamming_loss", "jaccard_score", 
        "per_label_accuracy", "confusion_matrices"}
    """
    # Combine train and test data for cross-validation
    X_all = np.vstack([X_train_scaled, X_test_scaled])
    y_all = np.vstack([y_train, y_test])
    
    # Apply PCA if requested
    if pca:
        pca_transformer = PCA(n_components=n_components)
        X_all = pca_transformer.fit_transform(X_all)
        
        if n_components is None or (isinstance(n_components, float) and 0 < n_components < 1):
            actual_components = X_all.shape[1]
            explained_variance = pca_transformer.explained_variance_ratio_.sum()
            print(f"PCA applied: {actual_components} components, {explained_variance:.4f} variance explained")
        else:
            print(f"PCA applied: {X_all.shape[1]} components")
    
    # Use regular k-fold for cross-validation
    # (StratifiedKFold doesn't support multilabel directly)
    cv_fold = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    multilabel_classifiers = get_multilabel_classifiers()
    results = {}
    
    for name, clf in multilabel_classifiers.items():
        # Get cross-validation predictions
        y_pred = cross_val_predict(clf, X_all, y_all, cv=cv_fold)
        
        # Calculate exact match accuracy manually using cross-validation
        # (all labels must match for a sample to be considered correct)
        cv_scores = []
        for train_idx, val_idx in cv_fold.split(X_all, y_all):
            y_val_true = y_all[val_idx]
            y_val_pred = y_pred[val_idx]
            
            # Calculate exact match accuracy (all labels must match)
            exact_match = np.all(y_val_pred == y_val_true, axis=1)
            accuracy_fold = np.mean(exact_match)
            cv_scores.append(accuracy_fold)
        
        accuracy = np.mean(cv_scores)
        accuracy_std = np.std(cv_scores)
        
        # Calculate per-label accuracy
        per_label_acc = []
        confusion_matrices = []
        for i, label_name in enumerate(label_names):
            label_acc = accuracy_score(y_all[:, i], y_pred[:, i])
            per_label_acc.append(label_acc)
            cm = confusion_matrix(y_all[:, i], y_pred[:, i])
            confusion_matrices.append(cm)
        
        results[name] = {
            "accuracy": accuracy,
            "accuracy_std": accuracy_std,
            "per_label_accuracy": dict(zip(label_names, per_label_acc)),
            "confusion_matrices": dict(zip(label_names, confusion_matrices))
        }
        
        print(f"{name}")
        print(f"  Overall Accuracy: {accuracy:.4f} (+/- {accuracy_std:.4f})")
        for label_name, acc in zip(label_names, per_label_acc):
            print(f"  {label_name} Accuracy: {acc:.4f}")
        print()
    
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
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
