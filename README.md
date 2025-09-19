# 🖼️ CIFAR-10 Image Classification using CNN

This project implements a **Convolutional Neural Network (CNN)** to classify images from the **CIFAR-10 dataset** into 10 classes:

- ✈️ Airplane  
- 🚗 Automobile  
- 🐦 Bird  
- 🐱 Cat  
- 🦌 Deer  
- 🐶 Dog  
- 🐸 Frog  
- 🐴 Horse  
- 🚢 Ship  
- 🚚 Truck  

---

## 📂 Dataset

The [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) contains:  
- **50,000 training images** (32×32 pixels, RGB)  
- **10,000 test images**  
- **10 classes**  

It is automatically downloaded with TensorFlow/Keras:

```python
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

---

## 🏗️ Model Architecture

The CNN is built using **Keras Sequential API**:

- `Conv2D` → `BatchNormalization` → `MaxPooling2D` → `Dropout`  
- Another `Conv2D` block with pooling + dropout  
- `Flatten` → `Dense (128, relu)` → `Dense (10, softmax)`  

**Optimizer:** Adam  
**Loss:** SparseCategoricalCrossentropy  
**Metric:** Accuracy  

---

## ⚡ Training

Train for 10 epochs:

```python
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
```

---

## 📊 Results

- Test accuracy: **~70–80%** (varies with epochs)  
- Example output:

```
Prediction: cat , True: cat
Prediction: dog , True: cat
```

---

## 🖼️ Predict with Your Own Image

You can test with your own image:

```python
import cv2, numpy as np
img = cv2.imread("cat.jpg")
img = cv2.resize(img, (32,32))
img = img.astype('float32') / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
print("Predicted Class:", class_names[np.argmax(prediction)])
```

⚠️ CIFAR-10 is low-res (32×32). Real-world photos may not classify correctly.  

---

## 📌 Requirements

- Python 3.x  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- OpenCV (for custom image testing)  

Install dependencies:

```bash
pip install tensorflow numpy matplotlib opencv-python
```

---

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/Arghadeepkar85/cifar10-cnn.git
   cd cifar10-cnn
   ```

2. Run training:
   ```bash
   python cifar10_cnn.py
   ```

3. (Optional) Test with your own image:
   ```bash
   python cifar10_cnn.py --image cat.jpg
   ```

---

## 🔮 Future Improvements

- Add **data augmentation**  
- Try deeper models (ResNet, VGG, etc.)  
- Use higher-resolution datasets (CIFAR-100, ImageNet)  

---

## 📜 License
MIT License  
