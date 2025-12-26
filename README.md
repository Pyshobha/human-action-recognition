# 🧍 Human Action Recognition using EfficientNet (PyTorch)

This project performs **image-based Human Action Recognition (HAR)** using deep learning and PyTorch.
It identifies human actions such as **laughing, running, sitting, eating**, etc., and outputs the predicted action along with a **confidence score**. 🎯

The system supports **GPU acceleration (CUDA)** with **CPU fallback** ⚡.

---

## Features ✨

* Image-based Human Action Recognition
* Pretrained EfficientNet-B0 backbone
* Transfer Learning for classifier layer
* Predicts action label and confidence score
* Supports GPU (CUDA) and CPU fallback
* Image preprocessing with PIL and torchvision transforms
* Easy to extend with more action classes

---

## Project Structure 📂

```
human_rec/
├── train.py                       # Script to train the model
├── testing.py                     # Script for image prediction
├── README.md
├── requirements.txt               # Dependencies
├── .gitignore
├── models/                        # Pretrained model (if available locally)
│   └── efficientnet_action_model.pth
└── data/                          # Dataset (to be added locally)
    ├── train/
    │   ├── laughing/
    │   ├── running/
    │   └── ...
    └── test/
        └── laughing/
```

---

## Installation 🛠️

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/human-action-recognition.git
cd human-action-recognition
```

2. **Create virtual environment**

```bash
python -m venv venv
```

3. **Activate virtual environment**

```powershell
# Windows (PowerShell)
venv\Scripts\activate
```

4. **Install dependencies**

**CPU-only**

```bash
pip install torch torchvision pillow
```

**GPU-enabled (CUDA)**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pillow
```

---

## Train the Model 🔧

Ensure your dataset is structured as:

```
data/train/action_name/image.png
```

Run the training script:

```bash
python train.py
```

* 80% of data used for training
* 20% of data used for validation

The trained model will be saved as:

```
models/efficientnet_action_model.pth
```

---

## Run Prediction / Testing 🔍

```bash
python testing.py
```

**Sample Output (GPU Inference):**

```
Using device: cuda
Model loaded successfully
Classes: ['calling', 'clapping', 'cycling', 'dancing', 'drinking',
          'eating', 'fighting', 'hugging', 'laughing',
          'listening_to_music', 'running', 'sitting',
          'sleeping', 'texting', 'using_laptop']

Prediction Result
------------------
Predicted Class : laughing
Confidence      : 97.34%
```

---

## Requirements 📋

* Python 3.10+
* PyTorch (CPU or GPU-enabled)
* torchvision
* pillow (PIL)

> NVIDIA GPU is recommended for faster inference (optional)

---

## Future Enhancements 🚀

* Video-based human action recognition
* Real-time webcam predictions
* Add more action classes
* Edge / mobile deployment
* Performance evaluation metrics
