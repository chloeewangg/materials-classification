import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

data_path = "features_dataset.csv"

if __name__ == "__main__":
    try:
        df = pd.read_csv(data_path)

        feature_cols = [col for col in df.columns if col not in ['material', 'location', 'distance']]
        X = df[feature_cols].values
        y = df['material'].values

        # Remove rows with NaN material labels
        valid_mask = ~pd.isna(y)
        X = X[valid_mask]
        y = y[valid_mask]

        print(f"Dataset shape: {X.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        print(f"Classes: {np.unique(y)}")

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"\nTrain set: {X_train_scaled.shape[0]} samples")
        print(f"Test set: {X_test_scaled.shape[0]} samples")

        classifiers = {
            'SVM': SVC(random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(random_state=42)
        }

        results = {}
        for name, clf in classifiers.items():
            print(f"\nTraining {name}...")
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
            print(f"{name} Accuracy: {accuracy:.4f}")

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_train_tensor = torch.LongTensor(y_train)
        y_test_tensor = torch.LongTensor(y_test)

        # Dataset class
        class MaterialDataset(Dataset):
            def __init__(self, X, y):
                self.X = X
                self.y = y
            
            def __len__(self):
                return len(self.X)
            
            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]

        train_dataset = MaterialDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        n_features = X_train_scaled.shape[1]
        n_classes = len(np.unique(y_encoded))

        # Neural Network Architectures
        class SimpleNN(nn.Module):
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
                super(DeepNN, self).__init__()
                self.fc1 = nn.Linear(input_dim, 256)
                self.relu1 = nn.ReLU()
                self.dropout1 = nn.Dropout(0.3)
                self.fc2 = nn.Linear(256, 128)
                self.relu2 = nn.ReLU()
                self.dropout2 = nn.Dropout(0.3)
                self.fc3 = nn.Linear(128, 64)
                self.relu3 = nn.ReLU()
                self.dropout3 = nn.Dropout(0.3)
                self.fc4 = nn.Linear(64, num_classes)
            
            def forward(self, x):
                x = self.relu1(self.fc1(x))
                x = self.dropout1(x)
                x = self.relu2(self.fc2(x))
                x = self.dropout2(x)
                x = self.relu3(self.fc3(x))
                x = self.dropout3(x)
                x = self.fc4(x)
                return x

        def train_pytorch_model(model, train_loader, epochs=50):
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            model.train()
            for epoch in range(epochs):
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            return model

        def evaluate_pytorch_model(model, X_test, y_test):
            model.eval()
            with torch.no_grad():
                outputs = model(X_test)
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == y_test).float().mean().item()
            return accuracy

        nn_models = {
            'Simple NN (64)': SimpleNN(n_features, n_classes),
            'Medium NN (128-64)': MediumNN(n_features, n_classes),
            'Deep NN (256-128-64)': DeepNN(n_features, n_classes)
        }

        for name, model in nn_models.items():
            print(f"\nTraining {name}...")
            model = train_pytorch_model(model, train_loader, epochs=50)
            accuracy = evaluate_pytorch_model(model, X_test_tensor, y_test_tensor)
            results[name] = accuracy
            print(f"{name} Accuracy: {accuracy:.4f}")

        print("\nModel Accuracies:")
        for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name:30s}: {acc:.4f}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
