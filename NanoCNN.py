import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import sys
sys.stdout.reconfigure(encoding='utf-8')



# --- 1. GLOBALNA KONFIGURACJA ---
@dataclass
class Config:
    """Parametry sterujące procesem uczenia."""
    seed: int = 42
    epochs: int = 100
    batch_size: int = 256
    lr: float = 0.00025
    num_classes: int = 10
    train_dir: str = "train"
    test_dir: str = "test"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fast_end: bool = False
    target_loss: float = 0.005

def init_environment(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- 2. UNIWERSALNY ŁADOWACZ OBRAZÓW ---
class ImageFolderDataset(Dataset):
    """Generyczny ładowacz obrazów z podziałem na podfoldery klas."""
    def __init__(self, root):
        self.samples = []
        self.labels = []

        print(f"Indeksowanie danych w: {root}")

        for label in range(10):
            class_path = os.path.join(root, str(label))
            if not os.path.exists(class_path):
                continue

            filenames = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png'))]

            for name in filenames:
                path = os.path.join(class_path, name)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

                if img is not None:
                    # Normalizacja i zmiana wymiarów do standardu wejściowego
                    img = cv2.resize(img, (28, 28))
                    tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)

                    self.samples.append(tensor)
                    self.labels.append(label)

        print(f"Zakończono: {len(self.samples)} próbek w pamięci.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

# --- 3. ARCHITEKTURA NANO-CNN ---
class NanoEncoder(nn.Module):
    """Lekki koder cech oparty na konwolucjach i uśrednianiu globalnym."""
    def __init__(self, out_features=10):
        super(NanoEncoder, self).__init__()

        self.activation = nn.GELU()

        # Bloki ekstrakcji (Feature Extractors)
        self.stage1 = self._make_block(1, 8, stride=2)
        self.stage2 = self._make_block(8, 16, stride=2)
        self.stage3 = self._make_block(16, 32, stride=1)

        # Agregacja przestrzenna i klasyfikacja
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(32, out_features)

    def _make_block(self, in_c, out_c, stride):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=0, stride=stride),
            nn.BatchNorm2d(out_c),
            self.activation
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)

# --- 4. ANALIZA WYNIKÓW ---
def plot_results(history, model):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Wykres MSE
    axes[0,0].plot(history['train_mse'], label='Train MSE')
    axes[0,0].plot(history['test_mse'], label='Test MSE')
    axes[0,0].set_title("Błąd Średniokwadratowy (MSE)")
    axes[0,0].legend()

    # 2. Wykres Błędu Klasyfikacji (1 - Accuracy)
    axes[0,1].plot(history['train_err'], label='Train Error')
    axes[0,1].plot(history['test_err'], label='Test Error')
    axes[0,1].set_title("Błąd Klasyfikacji (0-1)")
    axes[0,1].legend()

    # 3. Histogram wag - Pierwsza warstwa (Conv1)
    weights_conv = model.stage1[0].weight.detach().cpu().numpy().flatten()
    axes[1,0].hist(weights_conv, bins=30, color='skyblue', edgecolor='black')
    axes[1,0].set_title("Rozkład wag: Warstwa Konwolucyjna 1")

    # 4. Histogram wag - Ostatnia warstwa (Linear)
    weights_fc = model.head.weight.detach().cpu().numpy().flatten()
    axes[1,1].hist(weights_fc, bins=30, color='salmon', edgecolor='black')
    axes[1,1].set_title("Rozkład wag: Warstwa Wyjściowa (FC)")

    plt.tight_layout()
    plt.show()

# --- 5. PĘTLA GŁÓWNA ---
def main():
    cfg = Config()
    init_environment(cfg.seed)

    train_loader = DataLoader(ImageFolderDataset("train"), batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(ImageFolderDataset("test"), batch_size=cfg.batch_size)

    model = NanoEncoder(cfg.num_classes).to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    # Do celów MSE potrzebujemy One-Hot encodingu
    history = {'train_mse': [], 'test_mse': [], 'train_err': [], 'test_err': []}

    for epoch in range(cfg.epochs):
        model.train()
        train_mse, train_correct = 0, 0

        for x, y in train_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            optimizer.zero_grad()

            outputs = model(x)
            # CrossEntropy do optymalizacji
            loss = F.cross_entropy(outputs, y)
            loss.backward()
            optimizer.step()

            # Obliczanie MSE (potrzebne prawdopodobieństwa vs One-Hot)
            with torch.no_grad():
                probs = F.softmax(outputs, dim=1)
                y_onehot = F.one_hot(y, cfg.num_classes).float()
                train_mse += F.mse_loss(probs, y_onehot).item()
                train_correct += (outputs.argmax(1) == y).sum().item()

        # Ewaluacja
        model.eval()
        test_mse, test_correct = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(cfg.device), y.to(cfg.device)
                outputs = model(x)
                probs = F.softmax(outputs, dim=1)
                y_onehot = F.one_hot(y, cfg.num_classes).float()
                test_mse += F.mse_loss(probs, y_onehot).item()
                test_correct += (outputs.argmax(1) == y).sum().item()

        # Metryki
        epoch_train_mse = train_mse / len(train_loader)
        epoch_test_mse = test_mse / len(test_loader)
        train_error = 1 - (train_correct / len(train_loader.dataset))
        test_error = 1 - (test_correct / len(test_loader.dataset))

        history['train_mse'].append(epoch_train_mse)
        history['test_mse'].append(epoch_test_mse)
        history['train_err'].append(train_error)
        history['test_err'].append(test_error)

        print(f"Epoka {epoch+1} | Test MSE: {epoch_test_mse:.4f} | Błąd kl.: {test_error*100:.2f}%")


        if cfg.fast_end and epoch_train_mse <= cfg.target_loss: break

    plot_results(history, model)




if __name__ == "__main__":
    main()