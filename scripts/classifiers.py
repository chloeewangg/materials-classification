from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

def get_sklearn_classifiers():
    return {
        'SVM': SVC(
            kernel='rbf',
            C=1,
            gamma='scale',
            class_weight='balanced'),
        'KNN': KNeighborsClassifier(
            n_neighbors=7,
            weights='distance',
            p=2),
        # 'Logistic Regression': LogisticRegression(
        #     penalty='l2',
        #     C=0.1,
        #     class_weight='balanced'),
        'Random Forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight='balanced',
            random_state=42),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=12,
            class_weight='balanced',
            random_state=42)
    }

class MaterialDataset(Dataset):
    """PyTorch Dataset for Dataloader"""
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SimpleNN(nn.Module):
    """One hidden layer"""
    def __init__(self, input_dim, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.fc2(x)
        return x

class MediumNN(nn.Module):
    """Two hidden layers and dropout"""
    def __init__(self, input_dim, num_classes):
        super(MediumNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class DeepNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.15),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.15),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),

            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def get_pytorch_models(input_dim, num_classes):
    """
    Returns a dictionary of fresh PyTorch model instances.
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
    
    Returns:
        Dictionary mapping model names to fresh model instances
    """
    return {
        'Simple NN (64)': SimpleNN(input_dim, num_classes),
        'Medium NN (128-64)': MediumNN(input_dim, num_classes),
        'Deep NN (256-128-64)': DeepNN(input_dim, num_classes)
    }

def train_pytorch_model(model, train_loader, epochs=100):
    """
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        epochs: Number of training epochs
    
    Returns:
        (trained_model, loss_history)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_history = []
    
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        loss_history.append(loss.item())
    
    return model, loss_history

def evaluate_pytorch_model(model, X_test, y_test):
    """
    Args:
        model: PyTorch model to evaluate
        X_test: Test features tensor
        y_test: Test labels tensor
    
    Returns:
        (accuracy, y_pred)
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).float().mean().item()
    return accuracy, predicted.cpu().numpy()

