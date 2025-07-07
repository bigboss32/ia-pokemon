import os
import csv
import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from uno import PokemonMultiLabelDataset  # Cambia 'uno' si tu archivo se llama diferente

from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# 📂 Directorio raíz
root_dir = r"D:\ia-pokemon-workflow\imagen\pokemon_images_by_type"

# ✅ Tipos de Pokémon
all_types = [
    'normal', 'fire', 'water', 'electric', 'grass', 'ice',
    'fighting', 'poison', 'ground', 'flying', 'psychic',
    'bug', 'rock', 'ghost', 'dragon', 'dark', 'steel', 'fairy'
]

# ✅ Transforms con augmentations + normalización
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 📚 Dataset
dataset = PokemonMultiLabelDataset(root_dir, all_types, transform)
print(f"🔍 Total de imágenes: {len(dataset)}")

# 🔀 Train / Val split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ✅ Modelo base
model = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(all_types))  # SIN Sigmoid, usamos BCEWithLogitsLoss

# ✅ Fine-tuning completo
for param in model.parameters():
    param.requires_grad = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ⚙️ Loss + Optimizador
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# 📊 Registro de métricas
num_epochs = 50
best_val_loss = float('inf')

all_train_losses = []
all_val_losses = []
all_val_accuracies = []
all_precisions = []
all_recalls = []
all_f1s = []

threshold = 0.5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / train_size

    # 🔍 Validación + Accuracy + Métricas
    model.eval()
    val_loss = 0.0
    total_acc = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            preds = torch.sigmoid(outputs) > threshold

            acc = (preds == labels.bool()).float().mean(dim=1)
            total_acc += acc.sum().item()

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    val_loss = val_loss / val_size
    val_acc = total_acc / val_size

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()

    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    all_train_losses.append(epoch_loss)
    all_val_losses.append(val_loss)
    all_val_accuracies.append(val_acc)
    all_precisions.append(precision)
    all_recalls.append(recall)
    all_f1s.append(f1)

    print(f"📊 Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - Prec: {precision:.4f} - Rec: {recall:.4f} - F1: {f1:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "pokemon_multi_label.pth")
        print("✅ Nuevo mejor modelo guardado.")

# 🎉 Entrenamiento terminado
print("🏆 Mejor Val Loss:", best_val_loss)

# 📁 Guardar métricas CSV
with open("training_metrics.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Train Loss", "Val Loss", "Val Accuracy", "Precision", "Recall", "F1"])
    for i in range(num_epochs):
        writer.writerow([i+1, all_train_losses[i], all_val_losses[i], all_val_accuracies[i], all_precisions[i], all_recalls[i], all_f1s[i]])

print("✅ CSV de métricas guardado: training_metrics.csv")

# 📊 Gráficas
plt.figure(figsize=(10,5))
plt.plot(all_train_losses, label="Train Loss")
plt.plot(all_val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Val Loss")
plt.legend()
plt.savefig("loss_curve.png")
plt.close()

plt.figure(figsize=(10,5))
plt.plot(all_val_accuracies, label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.legend()
plt.savefig("accuracy_curve.png")
plt.close()

print("✅ Gráficas guardadas: loss_curve.png, accuracy_curve.png")

# 📷 Exportar 5 predicciones
os.makedirs("predictions", exist_ok=True)
saved = 0
model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.sigmoid(outputs) > threshold

        for i in range(images.size(0)):
            img = images[i].cpu()
            true_labels = [all_types[j] for j, val in enumerate(labels[i]) if val == 1]
            pred_labels = [all_types[j] for j, val in enumerate(preds[i].cpu()) if val == 1]

            filename = f"predictions/img_{saved}_true_{'_'.join(true_labels)}_pred_{'_'.join(pred_labels)}.png"
            save_image(img * 0.5 + 0.5, filename)

            saved += 1
            if saved >= 5:
                break
        if saved >= 5:
            break

print("✅ Ejemplos guardados en carpeta predictions/")
