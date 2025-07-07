import os
import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from uno import PokemonMultiLabelDataset  # Ajusta si tu archivo se llama diferente

# 📂 Directorio raíz
root_dir = r"D:\ia-pokemon-workflow\imagen\pokemon_images_by_type"

# ✅ Todos los tipos
all_types = [
    'normal', 'fire', 'water', 'electric', 'grass', 'ice',
    'fighting', 'poison', 'ground', 'flying', 'psychic',
    'bug', 'rock', 'ghost', 'dragon', 'dark', 'steel', 'fairy'
]

# ✅ Transforms con augmentations + NORMALIZACIÓN (imagenNet)
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

# ✅ Modelo: ResNet18 con pesos ImageNet
model = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features

# 📌 Sustituir FC: Linear SIN Sigmoid
model.fc = nn.Linear(num_ftrs, len(all_types))

# ✅ Descongelar TODO para fine-tuning completo
for param in model.parameters():
    param.requires_grad = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ⚙️ Loss multilabel
criterion = nn.BCEWithLogitsLoss()

# ⚙️ Optimizador: tasa baja para todo
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# 🧩 Entrenamiento
num_epochs = 100
best_val_loss = float('inf')

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

    # Validación
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

    val_loss = val_loss / val_size

    print(f"📊 Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}")

    # Guardar mejor modelo
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "pokemon_multi_label.pth")
        print("✅ Nuevo mejor modelo guardado.")

print("🎉 Entrenamiento terminado.")
print(f"🏆 Mejor Val Loss: {best_val_loss:.4f}")
