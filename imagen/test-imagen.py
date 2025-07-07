import torch
from torchvision import models, transforms
from PIL import Image

# 🎯 Etiquetas
all_types = [
    'normal', 'fire', 'water', 'electric', 'grass', 'ice',
    'fighting', 'poison', 'ground', 'flying', 'psychic',
    'bug', 'rock', 'ghost', 'dragon', 'dark', 'steel', 'fairy'
]

# 🚀 Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🧩 Modelo base - igual al entrenado
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(all_types))

# ✅ Cargar pesos entrenados
model.load_state_dict(torch.load("pokemon_multi_label.pth", map_location=device))
model = model.to(device)
model.eval()

# 📏 Transform igual al entrenamiento
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 🔮 Función de predicción
def predict_types(image_path, threshold=0.7):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]

    print(f"🔍 Probabilidades: {[round(p, 3) for p in probs]}")

    predicted_types = [all_types[i] for i, p in enumerate(probs) if p >= threshold]
    return predicted_types

# 📸 Prueba con tu imagen
test_image = r"D:\ia-pokemon-workflow\imagen\pokemon_images_by_type\electric\emolga_artwork.png"

predicted = predict_types(test_image, threshold=0.3)
print(f"✅ Tipos predichos: {predicted}")
