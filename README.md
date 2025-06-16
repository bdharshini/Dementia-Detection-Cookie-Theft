# 🧠 Dementia Detection Using the Cookie Theft Dataset

Detecting early signs of dementia, especially Alzheimer’s disease, through **language-based analysis** of patient speech describing a visual scene—the Cookie Theft image.

---

## 🖼️ The Cookie Theft Picture

Participants are asked to describe the image below. Their spoken descriptions are recorded and analyzed to assess signs of cognitive decline.

<p align="center">
  <img src="images/cookie_theft.jpg" alt="Cookie Theft Image" width="400"/>
</p>

---

## 📚 Dataset Overview

* **Source**: [Hugging Face – MearaHe/dementiabank](https://huggingface.co/datasets/MearaHe/dementiabank)
* **Samples**: 498
* **Columns**: 3
* **Labels**: `Dementia` / `Healthy`
* **Balance**: The dataset is **balanced**, so **accuracy** is used as the evaluation metric.
* **Limitation**: The dataset is relatively small, which may affect the generalization of ML models.

### 📊 Class Distribution

<p align="center">
  <img src="images/class_distribution.png" alt="Class Distribution" width="400"/>
</p>

---

## 🤖 Model Training & Evaluation

All models were trained on **TF-IDF vectorized** transcriptions of the participants' descriptions.

| Model                               | Accuracy |
| ----------------------------------- | -------- |
| **SVC (Support Vector Classifier)** | **0.88** |
| Logistic Regression                 | 0.82     |
| XGBoost                             | 0.80     |
| Decision Tree (Gini)                | 0.71     |
| Decision Tree (Entropy)             | 0.67     |

> ✅ **SVC** was selected as the final model due to its superior performance.

### 📈 Model Accuracy Comparison

<p align="center">
  <img src="images/model_accuracies.png" alt="Model Accuracy Chart" width="500"/>
</p>

---

## 🔄 End-to-End Prediction Pipeline

1. **🗣️ Audio Input**
   User describes the Cookie Theft image.

2. **📝 Speech-to-Text**
   Audio is transcribed into text using a speech recognition module.

3. **🧼 Text Preprocessing**
   Text is cleaned and transformed using **TF-IDF vectorization**.

4. **📊 Prediction**
   The trained **SVC model** predicts whether the speech indicates signs of dementia.

---

## 🚧 Limitations & Future Work

* **Small Dataset**: Limits generalization to larger populations.
* **Text Only**: Currently, only text features are used.

Future improvements may include:

* Integration of **acoustic features** (e.g., pitch, pauses).
* Development of a **UI**.
* Expansion using **multimodal data** (audio + text).

---
