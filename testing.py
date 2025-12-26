import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image

# -----------------------------
# 1. Device
# -----------------------------


#print("CUDA Available:", torch.cuda.is_available())
#print("CUDA Version:", torch.version.cuda)
#print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# 2. Image Transform (same as training)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# 3. Load Saved Model
# -----------------------------
model_path = r"D:\New folder\human_rec\efficientnet_action_model.pth"
checkpoint = torch.load(model_path, map_location=device, weights_only=True)

classes = checkpoint["classes"]

weights = EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights)

# Replace classifier
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(classes)
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

print("✅ Model loaded successfully")
print("Classes:", classes)

# -----------------------------
# 4. Predict Function
# -----------------------------
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return classes[predicted.item()], confidence.item() * 100

# -----------------------------
# 5. Test Prediction
# -----------------------------
if __name__ == "__main__":
    img_path = r"D:\New folder\human_rec\Structured\test\laughing\Image_12124.jpg"# <-- Replace with your image path
    pred_class, conf = predict_image(img_path)

    print("\nPrediction Result")
    print("------------------")
    print(f"Predicted Class : {pred_class}")
    print(f"Confidence      : {conf:.2f}%")
