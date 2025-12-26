🧍 Human Action Recognition using EfficientNet (PyTorch)

This project performs image-based Human Action Recognition (HAR) using deep learning and PyTorch.
It identifies human actions such as laughing, running, sitting, eating, etc., and outputs the predicted action and confidence score.

Supports GPU acceleration (CUDA) and CPU fallback.

🚀 Features

✅ Image-based Human Action Recognition
✅ Pretrained EfficientNet-B0 backbone
✅ Transfer Learning for classifier layer
✅ Predicts action label and confidence score
✅ Supports GPU (CUDA) and CPU fallback
✅ Image preprocessing with PIL + torchvision transforms
✅ Easy to extend with more action classes

🗂️ Project Structure

human_rec/
│
├── Structured/                  # Dataset folders
│   ├── train/
│   │   ├── laughing/
│   │   ├── running/
│   │   └── ...
│   └── test/
│       └── laughing/
│
├── train.py                     # Script to train the model
├── testing.py                   # Script for image prediction
├── efficientnet_action_model.pth # Saved trained model
├── README.md
└── requirements.txt             # Dependencies

🧠 How It Works

1.Dataset Preparation
    Organize images into folders by action class
    Example: Structured/train/laughing/

2.Preprocessing
    Convert images to RGB
    Resize to 224×224
    Normalize using ImageNet mean & std

3.Convert to PyTorch tensor
    Feature Extraction (CNN)
    Pass images through EfficientNet-B0
    Extract features like body posture, hand movements, object interaction

4.Classification
    Fine-tuned classifier layer predicts action class

5.Output
    Predicted class
    Confidence score (%)

⚙️ Installation
1️⃣ Clone the repository
     git clone https://github.com/yourusername/human-action-recognition.git
     cd human-action-recognition

2️⃣ Create Virtual Environment
     python -m venv venv

3️⃣ Activate Virtual Environment

    Windows (PowerShell):
    venv\Scripts\activate

4️⃣ Install Dependencies
CPU-only
pip install torch torchvision pillow

GPU-enabled (CUDA)

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

Install other dependencies:
pip install pillow

🧪 Train the Model

Organize your dataset in:
Structured/train/action_name/image.png

Run training script:
python train.py
Training uses 80% of dataset
Validation uses 20%

Model saved as:
efficientnet_action_model.pth

📷 Run Prediction / Test
python testing.py

✅ Sample Output (GPU Inference)
Using device: cuda
✅ Model loaded successfully
Classes: ['calling', 'clapping', 'cycling', 'dancing', 'drinking',
          'eating', 'fighting', 'hugging', 'laughing',
          'listening_to_music', 'running', 'sitting',
          'sleeping', 'texting', 'using_laptop']

Prediction Result
------------------
Predicted Class : laughing
Confidence      : 97.34%

📌 Requirements

Python 3.10+

PyTorch (CPU or GPU-enabled)

torchvision

pillow (PIL)

NVIDIA GPU recommended for faster inference (optional)

⭐ Future Enhancements

Video-based human action recognition

Real-time webcam predictions

Add more action classes

Edge/mobile deployment


Performance evaluation metrics

🧑‍💻 Author

Shobha Jangade
B.Tech – CSE (AI), CSVTU
Skills: Python, PyTorch, Deep Learning, Computer Vision#