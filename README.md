🧠 Dementia Detection using Cookie Theft Dataset
This project focuses on detecting early signs of dementia, particularly Alzheimer's disease, by analyzing speech descriptions of the "Cookie Theft" picture—a widely used diagnostic tool in cognitive assessments

📊 Dataset
Source: Hugging Face - MearaHe/dementiabank

Description:
The dataset contains 498 samples with 3 columns. Each sample includes a patient's speech transcription describing the Cookie Theft image. Based on the description, each subject is labeled as either having dementia or being cognitively healthy.

Challenge:
The relatively small dataset size poses a limitation to the model’s generalization capabilities.

Balance:
The dataset is balanced across classes, so accuracy is used as the primary evaluation metric.

🧪 Models and Results
The following models were trained using TF-IDF vectorized speech text:

Model	Accuracy
Support Vector Classifier (SVC)	0.88
Logistic Regression	0.82
XGBoost	0.80
Decision Tree (Gini)	0.71
Decision Tree (Entropy)	0.67

👉 SVC was selected as the final model for prediction due to its superior performance.

🔊 Real-Time Prediction Pipeline
Audio Input: User records their speech describing the Cookie Theft image.

Speech-to-Text: The audio is transcribed into text using a speech recognition library.

Preprocessing & Encoding: The text is cleaned and transformed using TF-IDF vectorization.

Prediction: The pre-trained SVC model classifies the sample as either "Dementia" or "Healthy".

🚧 Limitations & Future Work
Dataset Size: The limited number of samples restricts the model's potential.

Modalities: Currently, only transcribed text is used. Future work can integrate acoustic features from the audio.

UI/UX: Future versions may include a user-friendly interface for clinical or self-screening use.
